"""Hybrid algorithm: Top-K expert placement -> location_aware grouping -> backhaul repair."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from physical_layer import NOISE_PSD_W_PER_HZ, TX_PSD_W_PER_HZ

from rsma_integration import build_scheduler_inputs  # noqa: E402
from utils.formulation import (  # noqa: E402
    AssignmentKey,
    DeviceId,
    EvaluationResult,
    ExpertId,
    FormulationConfig,
    FormulationEvaluator,
    GroupId,
    GroupSpec,
    InfeasibleBandwidthBudget,
    ServerId,
    SubtaskSpec,
    normalize_devices,
    normalize_experts,
    normalize_servers,
    normalize_tasks,
)

BackhaulEdge = Tuple[ServerId, GroupId, ServerId]
GroupKey = Tuple[ServerId, GroupId]
FeatureDemandKey = Tuple[str, str, ServerId, str]
FeatureSource = Tuple[ServerId, GroupId]


@dataclass
class ChainedBackhaulDecision:
    source_server: ServerId
    group_id: GroupId
    target_server: ServerId
    features: List[str]


@dataclass
class ChainedPipelineResult:
    expert_placement: Dict[ServerId, List[ExpertId]]
    subtask_assignment: Dict[AssignmentKey, List[Tuple[ServerId, ExpertId]]]
    subtask_data_requirements: Dict[str, Dict[ServerId, List[str]]]
    server_required_features: Dict[ServerId, List[str]]
    rsma_groups: Dict[ServerId, List[List[DeviceId]]]
    group_required_features: Dict[str, List[str]]
    group_time_budgets: Dict[str, float]
    rsma_group_bandwidths: Dict[ServerId, List[float]]
    backhaul_plan: List[ChainedBackhaulDecision]
    selected_probability: Dict[str, float]
    required_probability: Dict[str, float]
    reconstruction_loss: Dict[str, float]
    performance_loss: Dict[str, float]
    total_cost: float
    activation_cost: float
    bandwidth_cost: float
    forwarding_cost: float
    inference_cost: float = 0.0
    backhaul: Set[BackhaulEdge] = field(default_factory=set)
    violations: List[str] = field(default_factory=list)
    evaluation: Optional[EvaluationResult] = None
    feature_source_map: Dict[FeatureDemandKey, FeatureSource] = field(
        default_factory=dict
    )


class ChainedComparisonPipeline:
    allow_expert_loss_repair: bool = True
    activate_all_candidate_groups: bool = False

    def __init__(
        self,
        servers: Iterable[Any],
        devices: Iterable[Any],
        experts: Mapping[str, Any],
        tasks: Iterable[Any],
        top_k: int = 2,
        rank_by: str = "gating",
        clusters_per_server: Optional[int] = None,
        kmeans_iterations: int = 20,
        config: Optional[FormulationConfig] = None,
        placement_random_seed: Optional[int] = None,
    ):
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if rank_by not in {"gating", "contribution"}:
            raise ValueError("rank_by must be either 'gating' or 'contribution'.")
        self.top_k = top_k
        self.rank_by = rank_by
        self.clusters_per_server = clusters_per_server
        self.kmeans_iterations = kmeans_iterations
        self.config = config or FormulationConfig()
        self.servers = normalize_servers(servers)
        self.devices = normalize_devices(devices)
        self.experts = normalize_experts(experts)
        self.tasks = normalize_tasks(tasks)
        self.evaluator = FormulationEvaluator(self.servers, self.devices, self.experts, self.tasks, self.config)
        self.placement_rng = random.Random(0 if placement_random_seed is None else placement_random_seed)
        self.scheduler_violations: List[str] = []

    def run(self) -> ChainedPipelineResult:
        assignments, selected_prob, selected_experts, activated, used_memory = self._assign_topk_experts()
        candidate_features = self._candidate_server_features(assignments)
        candidate_groups = self._build_distance_groups(candidate_features)
        if self.activate_all_candidate_groups:
            active_groups, subtask_features, required_prob, rec_loss, perf_loss = self._use_all_candidate_groups(
                assignments,
                selected_prob,
                selected_experts,
                candidate_groups,
                activated,
                used_memory,
            )
        else:
            active_groups, subtask_features, required_prob, rec_loss, perf_loss = self._repair_loss_with_groups(
                assignments,
                selected_prob,
                selected_experts,
                candidate_groups,
                activated,
                used_memory,
            )
        data_req = self._subtask_data_requirements(assignments, subtask_features)
        server_features = self._server_required_features(data_req)
        feature_to_groups = self._feature_to_groups(active_groups)
        backhaul, plan = self._derive_backhaul(data_req, feature_to_groups)
        groups, budgets = self._derive_group_bandwidths(active_groups, assignments, data_req, backhaul, feature_to_groups)
        eval_config = replace(self.config, derive_bandwidth=False)
        evaluator = FormulationEvaluator(self.servers, self.devices, self.experts, self.tasks, eval_config)
        evaluation = evaluator.evaluate(assignments, groups, backhaul, subtask_features=subtask_features)
        violations = [*self.scheduler_violations, *evaluation.violations]
        return ChainedPipelineResult(
            expert_placement=self._activated_expert_map(assignments),
            subtask_assignment=assignments,
            subtask_data_requirements=data_req,
            server_required_features={sid: sorted(v) for sid, v in server_features.items()},
            rsma_groups=self._group_map(groups),
            group_required_features={self._group_label(g): sorted(g.required_features) for g in groups},
            group_time_budgets={self._group_key_label(k): v for k, v in budgets.items()},
            rsma_group_bandwidths=self._group_bandwidth_map(groups),
            backhaul_plan=plan,
            selected_probability=selected_prob,
            required_probability=required_prob,
            reconstruction_loss=rec_loss,
            performance_loss=perf_loss,
            total_cost=evaluation.objective.total_cost,
            activation_cost=evaluation.objective.activation_cost,
            bandwidth_cost=evaluation.objective.bandwidth_cost,
            forwarding_cost=evaluation.objective.forwarding_cost,
            inference_cost=evaluation.objective.inference_cost,
            backhaul=backhaul,
            violations=violations,
            evaluation=evaluation,
        )
    def _use_all_candidate_groups(
        self,
        assignments,
        selected_probability,
        selected_experts_by_key,
        candidate_groups,
        activated,
        used_memory,
    ):
        active_groups = list(candidate_groups)
        active_feature_coverage: Set[str] = set()
        for group in active_groups:
            active_feature_coverage.update(group.required_features)
        subtask_features: Dict[AssignmentKey, Set[str]] = {}
        required_probability: Dict[str, float] = {}
        reconstruction_loss: Dict[str, float] = {}
        performance_loss: Dict[str, float] = {}
        reported_unavailable_features: Set[Tuple[AssignmentKey, ServerId, str]] = set()

        for task in self.tasks.values():
            for subtask in task.subtasks:
                key = (task.id, subtask.id)
                label = self._assignment_label(key)
                selected_features = set(subtask.required_features) & active_feature_coverage
                probability = selected_probability.get(label, 0.0)
                threshold = self.evaluator.conformal_loss_threshold(subtask)
                loss = self.evaluator.performance_loss_from_probability(subtask, probability, selected_features)

                while self.allow_expert_loss_repair and loss > threshold + 1e-12:
                    selected_experts = selected_experts_by_key.setdefault(key, set())
                    target_servers = self.evaluator.participating_servers(assignments, key)
                    assigned_server = target_servers[0] if len(target_servers) == 1 else None
                    candidate = self._best_repair_candidate(
                        subtask,
                        selected_experts,
                        activated,
                        used_memory,
                        assigned_server=assigned_server,
                    )
                    if candidate is None:
                        break
                    server_id, expert_id = candidate
                    pairs = assignments.setdefault(key, [])
                    if not self._append_unique_assignment(pairs, server_id, expert_id):
                        break
                    selected_experts.add(expert_id)
                    self._activate(server_id, expert_id, activated, used_memory)
                    probability += self.evaluator.expert_contribution(subtask, expert_id)
                    selected_probability[label] = probability
                    loss = self.evaluator.performance_loss_from_probability(subtask, probability, selected_features)

                if loss > threshold + 1e-12:
                    self.scheduler_violations.append(
                        f"Loss repair: {key} loss {loss:.6g} > threshold {threshold:.6g}"
                    )
                subtask_features[key] = selected_features
                required_probability[label] = self.evaluator.required_selection_probability(subtask, selected_features)
                reconstruction_loss[label] = self.evaluator.reconstruction_loss_for_features(subtask, selected_features)
                performance_loss[label] = loss

        return active_groups, subtask_features, required_probability, reconstruction_loss, performance_loss
    @staticmethod
    def _append_unique_assignment(
        pairs: List[Tuple[ServerId, ExpertId]],
        server_id: ServerId,
        expert_id: ExpertId,
    ) -> bool:
        if any(existing_expert == expert_id for _, existing_expert in pairs):
            return False
        pairs.append((server_id, expert_id))
        return True
    
    #這部份負責expert selection，因此對這個部份來修改
    def _assign_topk_experts(self):
        assignments: Dict[AssignmentKey, List[Tuple[ServerId, ExpertId]]] = {}
        selected_probability: Dict[str, float] = {}
        selected_experts_by_key: Dict[AssignmentKey, Set[ExpertId]] = {}
        activated = {sid: set(server.active_experts) for sid, server in self.servers.items()}
        used_memory = {
            sid: sum(self.experts[eid].memory for eid in eids if eid in self.experts)
            for sid, eids in activated.items()
        }
        for task in self.tasks.values():
            for subtask in task.subtasks:
                key = (task.id, subtask.id)
                label = self._assignment_label(key)
                server_id, expert_ids = self._best_server_for_subtask(subtask, activated, used_memory)
                selected: List[Tuple[ServerId, ExpertId]] = []
                selected_experts: Set[ExpertId] = set()
                if server_id is not None:
                    for expert_id in expert_ids:
                        selected.append((server_id, expert_id))
                        selected_experts.add(expert_id)
                        self._activate(server_id, expert_id, activated, used_memory)

                '''
                if not selected:
                    self.scheduler_violations.append(
                        f"TopK placement: no feasible expert on any single server for {key}"
                    )
                
                上面是AI改的（將265~268行內容改成256~259行內容），他覺得「選 Top-K 個 expert 只是演算法策略，不是 System Model constraint」，所以就改成不強制選 K 個 expert，只要確保能至少選到一個 expert 就好。
                但是我不同意他的說法，Top-K 這個比較方法本來就是要選 K 個 expert 才對，所以我就改回來了。
                '''

                if len(selected) < min(self.top_k, len(self.experts)):
                    self.scheduler_violations.append(
                        f"TopK placement: only {len(selected)} feasible experts for {key}; target {self.top_k}"
                    )
                assignments[key] = selected
                selected_experts_by_key[key] = selected_experts
                selected_probability[label] = self.evaluator.selection_probability(subtask, selected_experts)
        return assignments, selected_probability, selected_experts_by_key, activated, used_memory
        
    #此為額外新增的function，用在「_assign_topk_experts」function內
    def _best_server_for_subtask(
        self,
        subtask,
        activated,
        used_memory,
    ):
        candidates = []
        '''
        threshold = None
        try:
            threshold = self.evaluator.conformal_loss_threshold(subtask)
        except Exception:
            threshold = None
        '''

        for server_id, server in self.servers.items():

            feasible_experts = []
            extra_memory = 0.0

            for expert_id in self._ranked_experts(subtask):

                # 這台 server 必須有存這個 expert
                if expert_id not in server.stored_experts:
                    continue

                memory_add = 0.0

                if expert_id not in activated.get(server_id, set()):
                    memory_add = self.experts[expert_id].memory

                # 檢查這個 node 的多個 experts 加起來是否超過 GPU memory
                if (
                    used_memory.get(server_id, 0.0)
                    + extra_memory
                    + memory_add
                    > server.gpu_memory
                ):
                    continue

                feasible_experts.append(expert_id)
                extra_memory += memory_add

                if len(feasible_experts) >= self.top_k:
                    break

            if not feasible_experts:
                continue

            score = sum(
                self._expert_score(subtask, expert_id)
                for expert_id in feasible_experts
            )
            '''
            pairs = tuple((server_id, expert_id) for expert_id in feasible_experts)
            probability = self.evaluator.selection_probability(subtask, set(feasible_experts))
            rec_loss = self._estimated_local_reconstruction_for_pairs(subtask, pairs)
            loss = self.evaluator.performance_loss_from_probability_and_reconstruction(probability, rec_loss)
            threshold_penalty = 0 if threshold is not None and loss <= threshold + 1e-12 else 1

            candidates.append(
                (
                    threshold_penalty,
                    loss,
                    -len(feasible_experts),
                    -score,
                    server_id,
                    feasible_experts,
                )
            )
            '''

            candidates.append(
                (
                    len(feasible_experts),
                    score,
                    server_id,
                    feasible_experts,
                )
            )

        if not candidates:
            return None, []

        '''
        # Prefer a server whose local feature/expert combination is already
        # closest to satisfying the PDF performance-loss constraint.
        candidates.sort(
            key=lambda x: (x[0], x[1], x[2], x[3], str(x[4]))
        )

        _, _, _, _, server_id, expert_ids = candidates[0]

        以上是AI原本改的，包含這個 function（_best_server_for_subtask）內的註解都是AI改（新加上）的。
        他說不改會對於 Top-K 這個比較方法不利（差不多意思就是這樣）
        但是我不同意他的說法，Top-K 這個比較方法本來就只會根據 expert 因素來選 expert ，不會牽扯到 feature 相關的因素，而且正是這樣才會拿這個比較方法來比，以突顯我們的方法的優勢，所以我就改回來了。
        '''

        # 優先：
        # 1. 能提供較多 experts 的 server
        # 2. expert suitability / gating score 較高
        candidates.sort(
            key=lambda x: (-x[0], -x[1], str(x[2]))
        )

        _, _, server_id, expert_ids = candidates[0]
        
        return server_id, expert_ids

    #未使用
    def _estimated_local_reconstruction_for_pairs(
        self,
        subtask: SubtaskSpec,
        selected_pairs: Sequence[Tuple[ServerId, ExpertId]],
    ) -> float:
        required = set(subtask.required_features)
        if not required:
            return 0.0
        weighted_loss = 0.0
        total_experts = 0
        for server_id, _ in selected_pairs:
            local_features = self._server_local_features(
                self._server_transmittable_devices(server_id)
            ) & required
            weighted_loss += self.evaluator.reconstruction_loss_for_features(subtask, local_features)
            total_experts += 1
        if total_experts <= 0:
            return subtask.reconstruction_loss
        return weighted_loss / total_experts

    def _repair_loss_with_groups(
        self,
        assignments,
        selected_probability,
        selected_experts_by_key,
        candidate_groups,
        activated,
        used_memory,
    ):
        active_groups: Dict[GroupKey, GroupSpec] = {}
        active_backhaul: Set[BackhaulEdge] = set()
        active_group_budgets: Dict[GroupKey, float] = {}
        subtask_features: Dict[Tuple[str, str, str], Set[str]] = {}
        required_probability: Dict[str, float] = {}
        reconstruction_loss: Dict[str, float] = {}
        performance_loss: Dict[str, float] = {}
        reported_unavailable_features: Set[Tuple[AssignmentKey, ServerId, str]] = set()

        for task in self.tasks.values():
            for subtask in task.subtasks:
                key = (task.id, subtask.id)
                label = self._assignment_label(key)
                target_servers = self.evaluator.participating_servers(assignments, key)
                probability = selected_probability.get(label, 0.0)
                threshold = self.evaluator.conformal_loss_threshold(subtask)

                self._activate_local_groups_for_subtask(
                    subtask,
                    key,
                    target_servers,
                    candidate_groups,
                    active_groups,
                    active_group_budgets,
                    subtask_features,
                    assignments,
                    probability,
                    threshold,
                )
                avg_rec = self._average_reconstruction_for_servers(subtask, key, target_servers, subtask_features, assignments)
                loss = self.evaluator.performance_loss_from_probability_and_reconstruction(probability, avg_rec)

                while loss > threshold + 1e-12:
                    self._record_unavailable_missing_features(
                        subtask,
                        key,
                        target_servers,
                        subtask_features,
                        candidate_groups,
                        reported_unavailable_features,
                    )
                    repair = self._best_backhaul_group_loss_repair(
                        subtask,
                        key,
                        subtask_features,
                        probability,
                        threshold,
                        target_servers,
                        candidate_groups,
                        active_groups,
                        active_backhaul,
                        active_group_budgets,
                        assignments,
                    )
                    if repair is None:
                        break
                    group, target_server = repair
                    group_key = (group.server_id, group.id)
                    active_groups[group_key] = group
                    group_budget = self._subtask_bandwidth_budget(key, assignments)
                    active_group_budgets[group_key] = min(active_group_budgets.get(group_key, group_budget), group_budget)
                    if group.server_id != target_server:
                        active_backhaul.add((group.server_id, group.id, target_server))
                    self._server_feature_set(subtask_features, key, target_server).update(
                        group.required_features & subtask.required_features
                    )
                    avg_rec = self._average_reconstruction_for_servers(subtask, key, target_servers, subtask_features, assignments)
                    loss = self.evaluator.performance_loss_from_probability_and_reconstruction(probability, avg_rec)

                while self.allow_expert_loss_repair and loss > threshold + 1e-12:
                    selected_experts = selected_experts_by_key.setdefault(key, set())
                    assigned_server = target_servers[0] if len(target_servers) == 1 else None
                    candidate = self._best_repair_candidate(
                        subtask,
                        selected_experts,
                        activated,
                        used_memory,
                        assigned_server=assigned_server,
                    )
                    if candidate is None:
                        break
                    server_id, expert_id = candidate
                    pairs = assignments.setdefault(key, [])
                    if not self._append_unique_assignment(pairs, server_id, expert_id):
                        break
                    selected_experts.add(expert_id)
                    self._activate(server_id, expert_id, activated, used_memory)
                    probability += self.evaluator.expert_contribution(subtask, expert_id)
                    selected_probability[label] = probability
                    target_servers = self.evaluator.participating_servers(assignments, key)
                    self._activate_local_groups_for_subtask(
                        subtask,
                        key,
                        target_servers,
                        candidate_groups,
                        active_groups,
                        active_group_budgets,
                        subtask_features,
                        assignments,
                        probability,
                        threshold,
                    )
                    avg_rec = self._average_reconstruction_for_servers(subtask, key, target_servers, subtask_features, assignments)
                    loss = self.evaluator.performance_loss_from_probability_and_reconstruction(probability, avg_rec)

                if loss > threshold + 1e-12:
                    self.scheduler_violations.append(
                        f"Loss repair: {key} loss {loss:.6g} > threshold {threshold:.6g}"
                    )
                required_probability[label] = self._required_probability_with_reconstruction(subtask, avg_rec)
                reconstruction_loss[label] = avg_rec
                performance_loss[label] = loss

        return list(active_groups.values()), subtask_features, required_probability, reconstruction_loss, performance_loss

    def _server_feature_key(self, key: AssignmentKey, server_id: ServerId) -> Tuple[str, str, str]:
        return (key[0], key[1], server_id)

    def _server_feature_set(self, subtask_features, key: AssignmentKey, server_id: ServerId) -> Set[str]:
        return subtask_features.setdefault(self._server_feature_key(key, server_id), set())

    def _device_can_transmit_to_server(self, device_id: DeviceId, server_id: ServerId) -> bool:
        device = self.devices[device_id]
        return server_id in device.channel_gain or device.home_server == server_id

    def _server_transmittable_devices(self, server_id: ServerId) -> List[DeviceId]:
        return [
            device_id
            for device_id in self.devices
            if self._device_can_transmit_to_server(device_id, server_id)
        ]

    def _group_available_with_active_devices(
        self,
        group: GroupSpec,
        active_groups: Mapping[GroupKey, GroupSpec],
    ) -> bool:
        group_key = (group.server_id, group.id)
        if group_key in active_groups:
            return True
        active_devices: Set[DeviceId] = set()
        for active_group in active_groups.values():
            active_devices.update(active_group.devices)
        return not (set(group.devices) & active_devices)

    def _activate_local_groups_for_subtask(
        self,
        subtask: SubtaskSpec,
        key: AssignmentKey,
        target_servers: Sequence[ServerId],
        candidate_groups: Sequence[GroupSpec],
        active_groups: Dict[GroupKey, GroupSpec],
        active_group_budgets: Dict[GroupKey, float],
        subtask_features: Dict[Tuple[str, str, str], Set[str]],
        assignments: Mapping[AssignmentKey, Sequence[Tuple[ServerId, ExpertId]]],
        probability: float,
        threshold: float,
    ) -> None:
        group_budget = self._subtask_bandwidth_budget(key, assignments)
        needed = set(subtask.required_features)
        while True:
            current_rec = self._average_reconstruction_for_servers(subtask, key, target_servers, subtask_features, assignments)
            current_loss = self.evaluator.performance_loss_from_probability_and_reconstruction(probability, current_rec)
            if current_loss <= threshold + 1e-12:
                break
            repair = self._best_local_group_loss_repair(
                subtask,
                key,
                target_servers,
                candidate_groups,
                active_groups,
                active_group_budgets,
                subtask_features,
                assignments,
                probability,
                threshold,
            )
            if repair is None:
                break
            group, target_server = repair
            group_key = (group.server_id, group.id)
            active_groups[group_key] = group
            active_group_budgets[group_key] = min(active_group_budgets.get(group_key, group_budget), group_budget)
            self._server_feature_set(subtask_features, key, target_server).update(
                group.required_features & needed
            )

    def _best_local_group_loss_repair(
        self,
        subtask: SubtaskSpec,
        key: AssignmentKey,
        target_servers: Sequence[ServerId],
        candidate_groups: Sequence[GroupSpec],
        active_groups: Mapping[GroupKey, GroupSpec],
        active_group_budgets: Mapping[GroupKey, float],
        subtask_features: Mapping[Tuple[str, str, str], Set[str]],
        assignments: Mapping[AssignmentKey, Sequence[Tuple[ServerId, ExpertId]]],
        probability: float,
        threshold: float,
    ) -> Optional[Tuple[GroupSpec, ServerId]]:
        needed = set(subtask.required_features)
        satisfying = []
        improving = []
        current_rec = self._average_reconstruction_for_servers(subtask, key, target_servers, subtask_features, assignments)
        current_loss = self.evaluator.performance_loss_from_probability_and_reconstruction(probability, current_rec)
        for target_server in target_servers:
            selected = set(subtask_features.get(self._server_feature_key(key, target_server), set()))
            missing = needed - selected
            if not missing:
                continue
            usable = [
                group
                for group in candidate_groups
                if group.server_id == target_server and group.required_features & missing
                and self._group_available_with_active_devices(group, active_groups)
            ]
            for group in usable:
                trial_features = dict((k, set(v)) for k, v in subtask_features.items())
                trial_features.setdefault(self._server_feature_key(key, target_server), set()).update(
                    group.required_features & needed
                )
                new_rec = self._average_reconstruction_for_servers(subtask, key, target_servers, trial_features, assignments)
                new_loss = self.evaluator.performance_loss_from_probability_and_reconstruction(probability, new_rec)
                if new_loss >= current_loss - 1e-12:
                    continue
                cost = self._group_incremental_cost(
                    group,
                    key,
                    [target_server],
                    active_groups,
                    set(),
                    active_group_budgets,
                    assignments,
                )
                coverage = len(group.required_features & missing)
                item = (cost, -coverage, group.server_id, group.id, target_server, group)
                if new_loss <= threshold + 1e-12:
                    satisfying.append(item)
                else:
                    improvement = current_loss - new_loss
                    improving.append((-(improvement / max(cost, 1e-12)), cost, group.server_id, group.id, target_server, group))
        if satisfying:
            item = min(satisfying)
            return item[-1], item[-2]
        if improving:
            item = min(improving)
            return item[-1], item[-2]
        return None

    def _average_reconstruction_for_servers(
        self,
        subtask: SubtaskSpec,
        key: AssignmentKey,
        target_servers: Sequence[ServerId],
        subtask_features: Mapping[Tuple[str, str, str], Set[str]],
        assignments: Mapping[AssignmentKey, Sequence[Tuple[ServerId, ExpertId]]],
    ) -> float:
        if not target_servers:
            return subtask.reconstruction_loss

        weighted_loss = 0.0
        total_experts = 0
        for server_id in target_servers:
            expert_count = sum(1 for pair_server, _ in assignments.get(key, []) if pair_server == server_id)
            if expert_count <= 0:
                continue
            selected = set(subtask_features.get(self._server_feature_key(key, server_id), set()))
            server_loss = self.evaluator.reconstruction_loss_for_features(subtask, selected)
            weighted_loss += expert_count * server_loss
            total_experts += expert_count
        if total_experts <= 0:
            return subtask.reconstruction_loss
        return weighted_loss / total_experts

    def _record_unavailable_missing_features(
        self,
        subtask: SubtaskSpec,
        key: AssignmentKey,
        target_servers: Sequence[ServerId],
        subtask_features: Mapping[Tuple[str, str, str], Set[str]],
        candidate_groups: Sequence[GroupSpec],
        reported: Set[Tuple[AssignmentKey, ServerId, str]],
    ) -> None:
        available_features = set()
        for group in candidate_groups:
            available_features.update(group.required_features)
        needed = set(subtask.required_features)
        for target_server in target_servers:
            selected = set(subtask_features.get(self._server_feature_key(key, target_server), set()))
            for feature in sorted(needed - selected):
                report_key = (key, target_server, feature)
                if feature in available_features or report_key in reported:
                    continue
                reported.add(report_key)
                self.scheduler_violations.append(
                    f"Data requirement: feature {feature} needed by {key} on {target_server} "
                    "is unavailable from all IoT groups"
                )

    def _required_probability_with_reconstruction(self, subtask: SubtaskSpec, reconstruction_loss: float) -> float:
        threshold = self.evaluator.conformal_loss_threshold(subtask)
        if threshold == float("inf"):
            return 0.0
        denominator = threshold - (self.config.lambda_reconstruction * reconstruction_loss)
        if denominator <= 0.0:
            return float("inf")
        return 1.0 / denominator

    def _best_backhaul_group_loss_repair(
        self,
        subtask: SubtaskSpec,
        key: AssignmentKey,
        subtask_features: Mapping[Tuple[str, str, str], Set[str]],
        probability: float,
        threshold: float,
        target_servers: Sequence[ServerId],
        candidate_groups: Sequence[GroupSpec],
        active_groups: Mapping[GroupKey, GroupSpec],
        active_backhaul: Set[BackhaulEdge],
        active_group_budgets: Mapping[GroupKey, float],
        assignments: Mapping[AssignmentKey, Sequence[Tuple[ServerId, ExpertId]]],
    ) -> Optional[Tuple[GroupSpec, ServerId]]:
        needed = set(subtask.required_features)
        satisfying = []
        improving = []
        current_rec = self._average_reconstruction_for_servers(subtask, key, target_servers, subtask_features, assignments)
        current_loss = self.evaluator.performance_loss_from_probability_and_reconstruction(probability, current_rec)
        for target_server in target_servers:
            selected = set(subtask_features.get(self._server_feature_key(key, target_server), set()))
            missing = needed - selected
            if not missing:
                continue
            usable = [
                group
                for group in candidate_groups
                if group.server_id != target_server and group.required_features & missing
                and self._group_available_with_active_devices(group, active_groups)
            ]
            for group in usable:
                trial_features = dict((k, set(v)) for k, v in subtask_features.items())
                trial_features.setdefault(self._server_feature_key(key, target_server), set()).update(
                    group.required_features & needed
                )
                new_rec = self._average_reconstruction_for_servers(subtask, key, target_servers, trial_features, assignments)
                new_loss = self.evaluator.performance_loss_from_probability_and_reconstruction(probability, new_rec)
                if new_loss >= current_loss - 1e-12:
                    continue
                cost = self._group_incremental_cost(
                    group,
                    key,
                    [target_server],
                    active_groups,
                    active_backhaul,
                    active_group_budgets,
                    assignments,
                )
                coverage = len(group.required_features & missing)
                item = (cost, -coverage, group.server_id, group.id, target_server, group)
                if new_loss <= threshold + 1e-12:
                    satisfying.append(item)
                else:
                    improvement = current_loss - new_loss
                    improving.append((-(improvement / max(cost, 1e-12)), cost, group.server_id, group.id, target_server, group))
        if satisfying:
            item = min(satisfying)
            return item[-1], item[-2]
        if improving:
            item = min(improving)
            return item[-1], item[-2]
        return None

    def _group_incremental_cost(
        self,
        group: GroupSpec,
        key: AssignmentKey,
        target_servers: Sequence[ServerId],
        active_groups: Mapping[GroupKey, GroupSpec],
        active_backhaul: Set[BackhaulEdge],
        active_group_budgets: Mapping[GroupKey, float],
        assignments: Mapping[AssignmentKey, Sequence[Tuple[ServerId, ExpertId]]],
    ) -> float:
        cost = 0.0
        group_key = (group.server_id, group.id)
        try:
            new_budget = self._subtask_bandwidth_budget(
                key,
                assignments,
                group,
                target_servers,
            )
        except InfeasibleBandwidthBudget:
            return float("inf")
        if group_key in active_groups:
            old_budget = active_group_budgets.get(group_key, self.evaluator.bandwidth_time_budget())
            resolved_budget = min(old_budget, new_budget)
            if resolved_budget < old_budget - 1e-12:
                old_bw = self.evaluator.derive_group_bandwidth_for_budget(group, old_budget)
                new_bw = self.evaluator.derive_group_bandwidth_for_budget(group, resolved_budget)
                cost += self.config.c_bw * max(0.0, new_bw - old_bw)
        else:
            bandwidth = self.evaluator.derive_group_bandwidth_for_budget(group, new_budget)
            cost += self.config.c_bw * bandwidth
        for target_server in target_servers:
            edge = (group.server_id, group.id, target_server)
            if group.server_id != target_server and edge not in active_backhaul:
                cost += self.config.c_fwd * self.evaluator.wired_weight(group.server_id, target_server)
        return cost

    def _subtask_bandwidth_budget(
        self,
        key: AssignmentKey,
        assignments: Mapping[AssignmentKey, Sequence[Tuple[ServerId, ExpertId]]],
        group: GroupSpec | None = None,
        target_servers: Sequence[ServerId] = (),
    ) -> float:
        task = self.tasks.get(key[0])
        if task is None or task.deadline == float("inf"):
            return self.evaluator.bandwidth_time_budget()
        latest_start = self.evaluator.latest_subtask_start_times(assignments)
        remaining = latest_start.get(key, task.deadline)
        if group is not None:
            for target_server in target_servers:
                if group.server_id == target_server:
                    continue
                remaining = min(
                    remaining,
                    latest_start.get(key, task.deadline)
                    - self.evaluator.group_feature_volume(group)
                    / self.evaluator.wired_rate(group.server_id, target_server),
                )
        if remaining <= 0.0:
            raise InfeasibleBandwidthBudget(
                f"RSMA group {group.server_id}:{group.id} cannot serve "
                f"{key[0]}:{key[1]}: remaining uplink time is "
                f"{remaining:.6g}s"
            )
        return remaining
    def _candidate_server_features(self, assignments):
        required = {server_id: set() for server_id in self.servers}
        for task in self.tasks.values():
            for subtask in task.subtasks:
                key = (task.id, subtask.id)
                for server_id in self.evaluator.participating_servers(assignments, key):
                    required.setdefault(server_id, set()).update(subtask.required_features)
        return required
    def _ranked_experts(self, subtask: SubtaskSpec) -> List[ExpertId]:
        ranked = sorted(self.experts, key=lambda eid: self._expert_score(subtask, eid), reverse=True)
        return [eid for eid in ranked if self._expert_score(subtask, eid) > 0.0]

    def _top_k_experts(self, subtask: SubtaskSpec) -> List[ExpertId]:
        return self._ranked_experts(subtask)[: min(self.top_k, len(self.experts))]

    def _expert_score(self, subtask: SubtaskSpec, expert_id: ExpertId) -> float:
        expert = self.experts[expert_id]
        index = expert.index
        if index < 0 or index >= len(subtask.gating_weights):
            return 0.0
        gating = subtask.gating_weights[index]
        if self.rank_by == "gating":
            return gating
        if index >= len(subtask.expert_confidence):
            return 0.0
        return gating * subtask.expert_confidence[index]

    def _feasible_servers_for_expert(self, expert_id, activated, used_memory):
        feasible = []
        expert = self.experts[expert_id]
        for server_id, server in self.servers.items():
            if expert_id not in server.stored_experts:
                continue
            memory_after = used_memory.get(server_id, 0.0)
            if expert_id not in activated.get(server_id, set()):
                memory_after += expert.memory
            if memory_after <= server.gpu_memory:
                feasible.append(server_id)
        return sorted(feasible)

    def _random_server_for_expert(self, expert_id, activated, used_memory):
        feasible = self._feasible_servers_for_expert(expert_id, activated, used_memory)
        if not feasible:
            return None
        return self.placement_rng.choice(feasible)

    #不使用
    def _best_server_for_expert(self, expert_id, activated, used_memory):
        return self._random_server_for_expert(expert_id, activated, used_memory)

    def _best_repair_candidate(
        self,
        subtask,
        selected_experts,
        activated,
        used_memory,
        assigned_server: Optional[ServerId] = None,
    ):
        candidates = []
        for expert_id in sorted(self.experts):
            if expert_id in selected_experts:
                continue
            contribution = self.evaluator.expert_contribution(subtask, expert_id)
            if contribution <= 0.0:
                continue
            if assigned_server is None:
                server_id = self._random_server_for_expert(expert_id, activated, used_memory)
            elif assigned_server in self._feasible_servers_for_expert(expert_id, activated, used_memory):
                server_id = assigned_server
            else:
                server_id = None
            if server_id is None:
                continue
            expert = self.experts[expert_id]
            candidates.append((-contribution, expert.latency, expert_id, server_id))
        if not candidates:
            return None
        _, _, expert_id, server_id = min(candidates)
        return server_id, expert_id

    def _activate(self, server_id, expert_id, activated, used_memory):
        if expert_id in activated.setdefault(server_id, set()):
            return
        activated[server_id].add(expert_id)
        used_memory[server_id] = used_memory.get(server_id, 0.0) + self.experts[expert_id].memory

    def _subtask_data_requirements(self, assignments, subtask_features):
        requirements: Dict[str, Dict[ServerId, List[str]]] = {}
        for task in self.tasks.values():
            for subtask in task.subtasks:
                key = (task.id, subtask.id)
                server_map: Dict[ServerId, List[str]] = {}
                for server_id, _ in assignments.get(key, []):
                    server_key = (key[0], key[1], server_id)
                    if server_key in subtask_features:
                        features = sorted(subtask_features.get(server_key, set()))
                    else:
                        features = sorted(subtask_features.get(key, set()))
                    server_map[server_id] = features
                requirements[self._assignment_label(key)] = server_map
        return requirements
    def _server_required_features(self, subtask_data_requirements):
        required = {sid: set() for sid in self.servers}
        for server_map in subtask_data_requirements.values():
            for server_id, features in server_map.items():
                required.setdefault(server_id, set()).update(features)
        return required

    def _build_distance_groups(self, server_required_features):
        groups: List[GroupSpec] = []
        globally_required = set()
        for features in server_required_features.values():
            globally_required.update(features)
        for server_id in self.servers:
            local_devices = self._server_transmittable_devices(server_id)
            local_required = globally_required & self._server_local_features(local_devices)
            if not local_required:
                continue
            candidates = [did for did in local_devices if self.devices[did].features & local_required]
            if not candidates:
                self._record_local_missing_features(server_id, local_required, groups)
                continue

            candidate_groups: List[GroupSpec] = []
            clusters = self._kmeans_device_clusters(server_id, candidates, self._cluster_count(len(candidates)))
            for cluster in clusters:
                for chunk in self._chunk_by_distance(server_id, cluster):
                    required_features = self._group_local_features(chunk, local_required)
                    if required_features:
                        candidate_groups.append(
                            self._make_group(
                                server_id,
                                f"g{len(groups) + len(candidate_groups)}",
                                tuple(chunk),
                                required_features,
                            )
                        )
            groups.extend(candidate_groups)
            self._record_local_missing_features(server_id, local_required, groups)
        return groups

    def _server_local_features(self, local_devices):
        features = set()
        for device_id in local_devices:
            features.update(self.devices[device_id].features)
        return features
    def _group_local_features(self, device_ids, local_required):
        features = set()
        for device_id in device_ids:
            features.update(self.devices[device_id].features & local_required)
        return features

    def _record_local_missing_features(self, server_id, local_required, groups):
        available = set()
        for group in groups:
            if group.server_id == server_id:
                available.update(group.required_features)
        for feature in sorted(local_required - available):
            if not any(feature in device.features for device in self.devices.values()):
                self.scheduler_violations.append(
                    f"Data requirement: feature {feature} required by {server_id} is unavailable from all IoT groups"
                )

    def _cluster_count(self, num_devices):
        if self.clusters_per_server is not None:
            return min(self.clusters_per_server, num_devices)
        return max(1, math.ceil(num_devices / max(1, self.config.max_group_size)))

    def _kmeans_device_clusters(self, server_id, device_ids, cluster_count):
        ordered = sorted(device_ids, key=self._id_sort_key)
        centers = self._initial_centers(server_id, ordered, cluster_count)
        clusters = [[] for _ in range(cluster_count)]
        for _ in range(self.kmeans_iterations):
            next_clusters = [[] for _ in range(cluster_count)]
            for device_id in ordered:
                point = self._device_position(device_id, server_id)
                best_index = min(range(cluster_count), key=lambda index: self._squared_distance(point, centers[index]))
                next_clusters[best_index].append(device_id)
            for index, cluster in enumerate(next_clusters):
                if cluster:
                    centers[index] = self._centroid(server_id, cluster)
            if next_clusters == clusters:
                break
            clusters = next_clusters
        return [cluster for cluster in clusters if cluster]

    def _initial_centers(self, server_id, ordered, cluster_count):
        if cluster_count == 1:
            return [self._device_position(ordered[0], server_id)]
        centers = []
        for index in range(cluster_count):
            position = round(index * (len(ordered) - 1) / max(cluster_count - 1, 1))
            centers.append(self._device_position(ordered[position], server_id))
        return centers

    def _chunk_by_distance(self, server_id, cluster):
        centroid = self._centroid(server_id, cluster)
        ordered = sorted(
            cluster,
            key=lambda did: (self._squared_distance(self._device_position(did, server_id), centroid), self._id_sort_key(did)),
        )
        size = max(1, self.config.max_group_size)
        return [ordered[index : index + size] for index in range(0, len(ordered), size)]

    def _device_position(self, device_id, server_id):
        device = self.devices[device_id]
        if server_id in device.distance:
            distance = float(device.distance[server_id])
            theta = float(device.phase_angle.get(server_id, 0.0))
            return distance * math.cos(theta), distance * math.sin(theta)
        return device.position

    def _centroid(self, server_id, device_ids):
        points = [self._device_position(did, server_id) for did in device_ids]
        return (sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points))

    def _make_group(self, server_id, group_id, members, required_features):
        common_power = {did: self.devices[did].max_power * self.config.common_power_ratio for did in members}
        private_power = {did: self.devices[did].max_power * (1.0 - self.config.common_power_ratio) for did in members}
        return GroupSpec(
            server_id=server_id,
            id=group_id,
            devices=members,
            bandwidth=0.0,
            required_features=set(required_features),
            common_power=common_power,
            private_power=private_power,
        )

    def _derive_backhaul(self, subtask_data_requirements, feature_to_groups):
        backhaul: Set[BackhaulEdge] = set()
        source_by_edge: Dict[BackhaulEdge, GroupSpec] = {}
        for server_map in subtask_data_requirements.values():
            for target_server, features in server_map.items():
                for feature in features:
                    candidates = feature_to_groups.get(feature, [])
                    if any(group.server_id == target_server for group in candidates):
                        continue
                    source = self._best_source_group(target_server, candidates)
                    if source is None:
                        self.scheduler_violations.append(
                            f"Backhaul routing: feature {feature} is unavailable for server {target_server}"
                        )
                        continue
                    edge = (source.server_id, source.id, target_server)
                    backhaul.add(edge)
                    source_by_edge[edge] = source
        plan = [
            ChainedBackhaulDecision(
                source_server=s,
                group_id=g,
                target_server=t,
                features=sorted(
                    self.evaluator.group_transmitted_features(
                        source_by_edge[(s, g, t)]
                    )
                ),
            )
            for s, g, t in sorted(backhaul)
        ]
        return backhaul, plan

    def _best_source_group(self, target_server, candidates):
        remote = [group for group in candidates if group.server_id != target_server]
        if not remote:
            return None
        return min(
            remote,
            key=lambda group: (
                self.evaluator.group_feature_volume(group) / self.evaluator.wired_rate(group.server_id, target_server),
                group.server_id,
                group.id,
            ),
        )

    def _derive_group_bandwidths(
        self,
        groups,
        assignments,
        data_req,
        backhaul,
        feature_to_groups,
        feature_source_map=None,
    ):
        dependencies = self._group_dependencies(
            groups,
            assignments,
            data_req,
            backhaul,
            feature_to_groups,
            feature_source_map=feature_source_map,
        )
        latest_start = self.evaluator.latest_subtask_start_times(assignments)
        budgets: Dict[GroupKey, float] = {}
        resolved: List[GroupSpec] = []
        for group in groups:
            group_key = (group.server_id, group.id)
            budget = self._strict_group_budget(group, dependencies, latest_start)
            budgets[group_key] = budget
            bandwidth = self.evaluator.derive_group_bandwidth_for_budget(group, budget)
            resolved.append(replace(group, bandwidth=bandwidth))
        return resolved, budgets
    def _group_dependencies(
        self,
        groups,
        assignments,
        data_req,
        backhaul,
        feature_to_groups,
        feature_source_map=None,
    ):
        dependencies: Dict[GroupKey, Set[Tuple[AssignmentKey, ServerId]]] = {(g.server_id, g.id): set() for g in groups}
        group_by_key = {(group.server_id, group.id): group for group in groups}
        source_map = feature_source_map or {}
        for key in assignments:
            label = self._assignment_label(key)
            for target_server, features in data_req.get(label, {}).items():
                for feature in features:
                    demand_key = (key[0], key[1], target_server, feature)
                    mapped_source = source_map.get(demand_key)
                    if mapped_source is None:
                        group = self._serving_group_for_feature(
                            feature,
                            target_server,
                            backhaul,
                            feature_to_groups,
                        )
                    else:
                        source_key = tuple(mapped_source)
                        group = group_by_key.get(source_key)
                        if group is None:
                            raise ValueError(
                                f"Mapped feature source {source_key} does not exist "
                                f"for demand {demand_key}"
                            )
                        if feature not in self.evaluator.group_transmitted_features(group):
                            raise ValueError(
                                f"Mapped feature source {source_key} does not transmit "
                                f"{feature} for demand {demand_key}"
                            )
                        edge = (group.server_id, group.id, target_server)
                        if group.server_id != target_server and edge not in backhaul:
                            raise ValueError(
                                f"Mapped feature source {source_key} cannot reach "
                                f"{target_server} for demand {demand_key}"
                            )
                    if group is not None:
                        dependencies.setdefault((group.server_id, group.id), set()).add((key, target_server))
        return dependencies

    def _serving_group_for_feature(self, feature, target_server, backhaul, feature_to_groups):
        candidates = list(feature_to_groups.get(feature, []))
        local = [group for group in candidates if group.server_id == target_server]
        if local:
            return sorted(local, key=lambda group: group.id)[0]
        for source_server, group_id, dst_server in sorted(backhaul):
            if dst_server != target_server:
                continue
            for group in candidates:
                if group.server_id == source_server and group.id == group_id:
                    return group
        return None

    def _downstream_compute_times(self, assignments):
        downstream: Dict[AssignmentKey, float] = {}
        for task in self.tasks.values():
            children = {sub.id: [] for sub in task.subtasks}
            for sub in task.subtasks:
                for pred_id in sub.predecessors:
                    if pred_id in children:
                        children[pred_id].append(sub)

            def visit(subtask):
                key = (task.id, subtask.id)
                if key in downstream:
                    return downstream[key]
                compute = self._subtask_compute_time(key, assignments)
                child_tail = max((visit(child) for child in children[subtask.id]), default=0.0)
                downstream[key] = compute + child_tail
                return downstream[key]

            for subtask in task.subtasks:
                visit(subtask)
        return downstream

    def _subtask_compute_time(self, key, assignments):
        return max((self.experts[eid].latency for _, eid in assignments.get(key, [])), default=0.0)

    def _strict_group_budget(self, group, dependencies, latest_start):
        group_key = (group.server_id, group.id)
        dependent_items = dependencies.get(group_key, set())
        if not dependent_items:
            return self.evaluator.bandwidth_time_budget()
        budgets = []
        for (task_id, subtask_id), target_server in dependent_items:
            task = self.tasks[task_id]
            if task.deadline == float("inf"):
                budgets.append(self.evaluator.bandwidth_time_budget())
            else:
                backhaul_time = 0.0
                if group.server_id != target_server:
                    backhaul_time = self.evaluator.group_feature_volume(group) / self.evaluator.wired_rate(group.server_id, target_server)
                remaining = (
                    latest_start.get((task_id, subtask_id), task.deadline)
                    - backhaul_time
                )
                if remaining <= 0.0:
                    raise InfeasibleBandwidthBudget(
                        f"RSMA group {group.server_id}:{group.id} cannot serve "
                        f"{task_id}:{subtask_id} on {target_server}: remaining "
                        f"uplink time is {remaining:.6g}s"
                    )
                budgets.append(remaining)
        return min(budgets)

    def _derive_bandwidth_for_budget(self, group, budget):
        if not group.devices or not group.required_features:
            return 0.0
        return self.evaluator.derive_group_bandwidth_for_budget(group, budget)

    def _feature_to_groups(self, groups):
        feature_to_groups: Dict[str, List[GroupSpec]] = {}
        for group in groups:
            for feature in self.evaluator.group_payload_features(group):
                feature_to_groups.setdefault(feature, []).append(group)
        return feature_to_groups

    def _activated_expert_map(self, assignments):
        placement = {sid: set() for sid in self.servers}
        for pairs in assignments.values():
            for server_id, expert_id in pairs:
                placement.setdefault(server_id, set()).add(expert_id)
        return {sid: sorted(experts) for sid, experts in placement.items()}

    def _group_map(self, groups):
        grouped = {sid: [] for sid in self.servers}
        for group in groups:
            grouped.setdefault(group.server_id, []).append(list(group.devices))
        return grouped

    def _group_bandwidth_map(self, groups):
        bandwidths = {sid: [] for sid in self.servers}
        for group in groups:
            bandwidths.setdefault(group.server_id, []).append(group.bandwidth)
        return bandwidths

    @staticmethod
    def _assignment_label(key):
        return f"{key[0]}:{key[1]}"

    @staticmethod
    def _group_label(group):
        return f"{group.server_id}:{group.id}"

    @staticmethod
    def _group_key_label(group_key):
        return f"{group_key[0]}:{group_key[1]}"

    @staticmethod
    def _id_sort_key(value):
        prefix, _, suffix = str(value).rpartition("_")
        try:
            return prefix, int(suffix), str(value)
        except ValueError:
            return str(value), 0, str(value)

    @staticmethod
    def _squared_distance(a, b):
        return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

def _safe_usage(cost: float, unit_cost: float) -> float | int:
    if unit_cost == 0:
        return 0
    usage = cost / unit_cost
    if abs(usage - round(usage)) < 1e-9:
        return int(round(usage))
    return usage


def _cost_breakdown(result: ChainedPipelineResult, cost_units: Mapping[str, float]) -> Dict[str, Any]:
    c_act = float(cost_units.get("activation", 0.0))
    c_bw = float(cost_units.get("bandwidth", 0.0))
    c_fwd = float(cost_units.get("forwarding", 0.0))
    c_inf = float(cost_units.get("inference", 1.0))
    return {
        "activation": {
            "usage": _safe_usage(result.activation_cost, c_act),
            "usage_unit": "activated expert memory",
            "unit_cost": c_act,
            "cost": int(result.activation_cost + 0.5),
        },
        "bandwidth": {
            "usage": _safe_usage(result.bandwidth_cost, c_bw),
            "usage_unit": "Hz",
            "unit_cost": c_bw,
            "cost": int(result.bandwidth_cost + 0.5),
        },
        "forwarding": {
            "usage": _safe_usage(result.forwarding_cost, c_fwd),
            "usage_unit": "forwarding events",
            "unit_cost": c_fwd,
            "cost": int(result.forwarding_cost + 0.5),
        },
        "inference": {
            "usage": _safe_usage(result.inference_cost, c_inf),
            "usage_unit": "expert inference cost units",
            "unit_cost": c_inf,
            "cost": int(result.inference_cost + 0.5),
        },
    }


def result_to_jsonable(
    result: ChainedPipelineResult,
    network_context: Mapping[str, Any] | None = None,
    cost_units: Mapping[str, float] | None = None,
) -> Dict[str, Any]:
    evaluation = result.evaluation
    cost_units = cost_units or {}
    payload: Dict[str, Any] = {
        "method": "hybrid_topk_location_aware",
        "objective": {
            "total_cost": int(result.total_cost + 0.5),
            "activation_cost": int(result.activation_cost + 0.5),
            "bandwidth_cost": int(result.bandwidth_cost + 0.5),
            "forwarding_cost": int(result.forwarding_cost + 0.5),
            "inference_cost": int(result.inference_cost + 0.5),
        },
        "cost_breakdown": _cost_breakdown(result, cost_units),
        "expert_placement": result.expert_placement,
        "subtask_assignment": {f"{tid}:{sid}": pairs for (tid, sid), pairs in result.subtask_assignment.items()},
        "subtask_data_requirements": result.subtask_data_requirements,
        "server_required_features": result.server_required_features,
        "selected_probability": result.selected_probability,
        "required_probability": result.required_probability,
        "reconstruction_loss": result.reconstruction_loss,
        "performance_loss": result.performance_loss,
        "rsma_groups": result.rsma_groups,
        "group_required_features": result.group_required_features,
        "group_time_budgets": result.group_time_budgets,
        "rsma_group_bandwidths": result.rsma_group_bandwidths,
        "feature_source_map": {
            f"{task_id}:{subtask_id}:{target_server}:{feature}": {
                "source_server": source_server,
                "group_id": group_id,
            }
            for (
                task_id,
                subtask_id,
                target_server,
                feature,
            ), (
                source_server,
                group_id,
            ) in sorted(result.feature_source_map.items())
        },
        "backhaul": [list(item) for item in sorted(result.backhaul)],
        "backhaul_plan": [decision.__dict__ for decision in result.backhaul_plan],
        "violations": result.violations,
        "task_finish_time": evaluation.timing.task_finish_time if evaluation else {},
    }
    if network_context is not None:
        payload["network_model"] = dict(network_context)
    return payload


def run_chained_pipeline(
    graphs: Sequence[Any],
    output_path: Path,
    num_experts: int,
    num_iot_features: int,
    expert_memory_range: tuple[float, float] | None = None,
    expert_memory_sizes_mb: Sequence[float] | None = None,
    expert_inference_times_ms: Sequence[float] | None = None,
    inference_unit_cost_per_mb: float = 0.06,
    reasoning_data_sizes_bytes: Sequence[int] = (512,),
    feature_bits_range: tuple[float, float] | None = None,
    top_k: int = 2,
    num_servers: int = 9,
    num_iot_devices: int = 100,
    features_per_device_range: tuple[int, int] = (5, 12),
    server_feature_overlap_ratio: float = 0.25,
    global_random_feature_fraction: float = 0.15,
    experts_per_server: int = 4,
    server_gpu_memory: float = 8192.0,
    server_gpu_memory_range: tuple[float, float] | None = None,
    wired_rate_range: tuple[float, float] | None = None,
    wired_extra_link_probability: float = 0.05,
    wired_edge_weight_range: tuple[float, float] = (1.0, 3.0),
    c_bw: float = 1e-3,
    c_act: float = 1.0,
    c_fwd: float = 1.0,
    default_feature_bits: float = 12000.0,
    wavelength: float = 0.125,
    noise_power: float = NOISE_PSD_W_PER_HZ,
    common_power_ratio: float = 0.6,
    max_device_power: float = TX_PSD_W_PER_HZ,
    area_size: float = 1000.0,
    cell_radius: float = 300.0,
    num_antennas: int = 4,
    loss_threshold: float | None = None,
    lambda_reconstruction: float = 0.1,
    calibration_alpha: float = 0.1,
    reconstruction_sigma: float = 1.0,
    random_seed: int = 42,
    max_group_size: int = 5,
    min_rate: float = 1.0,
    gssgd_beamforming_gain_threshold: float = 0.0,
    rank_by: str = "gating",
    clusters_per_server: Optional[int] = None,
    kmeans_iterations: int = 20,
) -> ChainedPipelineResult:
    inputs = build_scheduler_inputs(
        graphs=graphs,
        num_experts=num_experts,
        num_iot_features=num_iot_features,
        expert_memory_range=expert_memory_range,
        expert_memory_sizes_mb=expert_memory_sizes_mb,
        expert_inference_times_ms=expert_inference_times_ms,
        inference_unit_cost_per_mb=inference_unit_cost_per_mb,
        reasoning_data_sizes_bytes=reasoning_data_sizes_bytes,
        feature_bits_range=feature_bits_range,
        num_servers=num_servers,
        num_iot_devices=num_iot_devices,
        features_per_device_range=features_per_device_range,
        server_feature_overlap_ratio=server_feature_overlap_ratio,
        global_random_feature_fraction=global_random_feature_fraction,
        experts_per_server=experts_per_server,
        server_gpu_memory=server_gpu_memory,
        server_gpu_memory_range=server_gpu_memory_range,
        wired_rate_range=wired_rate_range,
        wired_extra_link_probability=wired_extra_link_probability,
        wired_edge_weight_range=wired_edge_weight_range,
        default_feature_bits=default_feature_bits,
        max_device_power=max_device_power,
        area_size=area_size,
        cell_radius=cell_radius,
        num_antennas=num_antennas,
        loss_threshold=loss_threshold,
        random_seed=random_seed,
    )
    tasks = inputs.tasks
    feature_bits_by_name = inputs.feature_bits_by_name
    experts = inputs.experts
    servers = inputs.servers
    devices = inputs.devices
    topology = inputs.topology
    pipeline = ChainedComparisonPipeline(
        servers=servers,
        devices=devices,
        experts=experts,
        tasks=tasks,
        top_k=top_k,
        rank_by=rank_by,
        clusters_per_server=clusters_per_server,
        kmeans_iterations=kmeans_iterations,
        placement_random_seed=random_seed,
        config=FormulationConfig(
            max_group_size=max_group_size,
            min_rate=min_rate,
            gssgd_beamforming_gain_threshold=gssgd_beamforming_gain_threshold,
            c_bw=c_bw,
            c_act=c_act,
            c_fwd=c_fwd,
            c_inf=inference_unit_cost_per_mb,
            derive_bandwidth=True,
            noise_power=noise_power,
            common_power_ratio=common_power_ratio,
            default_power=max_device_power,
            default_feature_bits=default_feature_bits,
            feature_bits_by_name=feature_bits_by_name,
            default_loss_threshold=loss_threshold if loss_threshold is not None else 3.0,
            lambda_reconstruction=lambda_reconstruction,
            calibration_alpha=calibration_alpha,
            reconstruction_sigma=reconstruction_sigma,
            wavelength=wavelength,
        ),
    )
    result = pipeline.run()
    network_context = {
        "experts": experts,
        "servers": servers,
        "devices": devices,
        "topology": topology,
        "unit_cost": {
            "activation": c_act,
            "bandwidth": c_bw,
            "forwarding": c_fwd,
            "inference": inference_unit_cost_per_mb,
        },
        "hybrid_algorithm_parameters": {
            "top_k": top_k,
            "rank_by": rank_by,
            "clusters_per_server": clusters_per_server,
            "kmeans_iterations": kmeans_iterations,
            "local_grouping_only": True,
            "bandwidth_budget": "strictest dependent subtask remaining time",
        },
        "rsma_parameters": {
            "num_servers": num_servers,
            "num_iot_devices": num_iot_devices,
            "features_per_device_range": features_per_device_range,
            "expert_memory_range": expert_memory_range,
            "expert_memory_sizes_mb": list(expert_memory_sizes_mb) if expert_memory_sizes_mb is not None else None,
            "reasoning_data_sizes_bytes": list(reasoning_data_sizes_bytes),
            "experts_per_server": experts_per_server,
            "server_gpu_memory_range": server_gpu_memory_range,
            "wired_rate_range": wired_rate_range,
            "wired_extra_link_probability": wired_extra_link_probability,
            "wired_edge_weight_range": wired_edge_weight_range,
            "bandwidth_mode": "pdf_equivalent_bandwidth_demand_by_group_slack",
            "default_feature_bits": default_feature_bits,
            "feature_bits_range": feature_bits_range,
            "feature_bits_by_name": feature_bits_by_name,
            "wavelength": wavelength,
            "noise_power": noise_power,
            "noise_power_unit": "W/Hz",
            "common_power_ratio": common_power_ratio,
            "max_device_power": max_device_power,
            "max_device_power_unit": "W/Hz",
            "rate_model": "Shannon (linear SINR)",
            "spectral_efficiency_unit": "bit/s/Hz",
            "area_size": area_size,
            "cell_radius": cell_radius,
            "num_antennas": num_antennas,
            "max_group_size": max_group_size,
            "min_rate": min_rate,
            "location_aware_group_selection_mode": "loss_repair_until_bound",
            "loss_threshold": loss_threshold,
            "lambda_reconstruction": lambda_reconstruction,
            "calibration_alpha": calibration_alpha,
            "reconstruction_sigma": reconstruction_sigma,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            result_to_jsonable(
                result,
                network_context,
                cost_units={
                    "activation": c_act,
                    "bandwidth": c_bw,
                    "forwarding": c_fwd,
                    "inference": inference_unit_cost_per_mb,
                },
            ),
            file,
            ensure_ascii=False,
            indent=2,
        )
    return result



