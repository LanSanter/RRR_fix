"""WDMoE expert selection adapted to this project.

This module implements a DAG-aware version of WDMoE Algorithm 1 for the
current RSMA-MoE environment. The original paper routes tokens to wireless
experts; here each DAG node is treated as one subtask and each selected expert
must be placed on a feasible edge server that stores it.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Set

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from compared_method.hybrid_topk_location_aware_backhaul import (  # noqa: E402
    ChainedComparisonPipeline,
    ChainedPipelineResult,
    result_to_jsonable as hybrid_result_to_jsonable,
)
from compared_method.hybrid_topk_gssgd_backhaul import HybridTopKGSSGDBackhaulPipeline  # noqa: E402
from physical_layer import NOISE_PSD_W_PER_HZ, TX_PSD_W_PER_HZ

from rsma_integration import build_scheduler_inputs  # noqa: E402
from utils.formulation import AssignmentKey, ExpertId, FormulationConfig, ServerId  # noqa: E402


class WDMoEExpertSelectionMixin:
    """DAG-aware WDMoE expert selection adapted to per-subtask loss bounds.

    For each task graph, the ordinary Top-K result is used as the WLR baseline.
    The WDMoE threshold theta starts from alpha and increases by Delta-theta;
    each theta trial reruns all subtasks from the same pre-task state. A subtask
    whose cosine(G, latency) is below theta drops the lowest-gating selected
    expert once, matching the paper-style Top-K to K-1 pruning behavior.
    """

    allow_expert_loss_repair: bool = False
    wdmoe_initial_threshold: float = 0.8
    wdmoe_threshold_step: float = 0.05
    wdmoe_max_threshold: float = 1.0
    wdmoe_wlr_target_ratio: float = 1.05

    def _assign_topk_experts(self):
        assignments: Dict[AssignmentKey, list[tuple[ServerId, ExpertId]]] = {}
        selected_probability: Dict[str, float] = {}
        selected_experts_by_key: Dict[AssignmentKey, Set[ExpertId]] = {}
        activated = {sid: set(server.active_experts) for sid, server in self.servers.items()}
        used_memory = {
            sid: sum(self.experts[eid].memory for eid in eids if eid in self.experts)
            for sid, eids in activated.items()
        }
        self.wdmoe_task_metrics: Dict[str, Dict[str, float]] = {}

        for task in self.tasks.values():
            task_assignments, task_selected, activated, used_memory, metrics = self._wdmoe_assign_task_graph(
                task=task,
                committed_assignments=assignments,
                activated=activated,
                used_memory=used_memory,
            )
            assignments.update(task_assignments)
            selected_experts_by_key.update(task_selected)
            self.wdmoe_task_metrics[task.id] = metrics

            for key, selected_experts in task_selected.items():
                subtask = self._task_subtask_by_id(task, key[1])
                if subtask is None:
                    continue
                selected_probability[self._assignment_label(key)] = self.evaluator.selection_probability(
                    subtask,
                    selected_experts,
                )

        return assignments, selected_probability, selected_experts_by_key, activated, used_memory

    def _wdmoe_assign_task_graph(self, task, committed_assignments, activated, used_memory):
        base_activated = {sid: set(eids) for sid, eids in activated.items()}
        base_memory = dict(used_memory)

        _, _, _, _, baseline_node_metrics = self._wdmoe_run_task_trial(
            task=task,
            committed_assignments=committed_assignments,
            activated={sid: set(eids) for sid, eids in base_activated.items()},
            used_memory=dict(base_memory),
            theta=None,
            allow_pruning=False,
        )
        baseline_wlr = self._wdmoe_task_wlr(baseline_node_metrics, "baseline_wlr")

        max_theta = min(float(self.wdmoe_max_threshold), 1.0)
        theta = float(self.wdmoe_initial_threshold)
        step = max(float(self.wdmoe_threshold_step), 1e-12)
        best_trial = None
        theta_trials = []
        met_gamma = False

        while theta <= max_theta + 1e-12:
            trial_assignments, trial_selected, trial_activated, trial_memory, node_metrics = self._wdmoe_run_task_trial(
                task=task,
                committed_assignments=committed_assignments,
                activated={sid: set(eids) for sid, eids in base_activated.items()},
                used_memory=dict(base_memory),
                theta=theta,
                allow_pruning=True,
            )
            selected_wlr = self._wdmoe_task_wlr(node_metrics, "selected_wlr")
            ratio = selected_wlr / max(baseline_wlr, 1e-12)
            pruned_nodes = sum(1 for item in node_metrics.values() if item.get("pruned", False))
            trial_summary = {
                "theta": theta,
                "selected_wlr": selected_wlr,
                "wlr_ratio": ratio,
                "pruned_nodes": pruned_nodes,
            }
            theta_trials.append(trial_summary)
            best_trial = (trial_assignments, trial_selected, trial_activated, trial_memory, node_metrics, theta, ratio)
            if ratio > self.wdmoe_wlr_target_ratio:
                met_gamma = True
                break
            if theta >= max_theta - 1e-12:
                break
            theta = min(theta + step, max_theta)

        if best_trial is None:
            best_trial = self._wdmoe_run_task_trial(
                task=task,
                committed_assignments=committed_assignments,
                activated={sid: set(eids) for sid, eids in base_activated.items()},
                used_memory=dict(base_memory),
                theta=max_theta,
                allow_pruning=True,
            ) + (max_theta, 1.0)

        task_assignments, selected_by_key, committed_activated, committed_memory, node_metrics, final_theta, ratio = best_trial
        selected_wlr = self._wdmoe_task_wlr(node_metrics, "selected_wlr")
        pruned_nodes = sum(1 for item in node_metrics.values() if item.get("pruned", False))
        metrics = {
            "wlr_scope": "task_graph",
            "baseline_wlr": baseline_wlr,
            "selected_wlr": selected_wlr,
            "wlr_ratio": ratio,
            "initial_threshold": self.wdmoe_initial_threshold,
            "final_threshold": final_theta,
            "threshold_step": self.wdmoe_threshold_step,
            "max_threshold": max_theta,
            "met_gamma": met_gamma,
            "pruned_nodes": pruned_nodes,
            "theta_trials": theta_trials,
            "nodes": node_metrics,
        }
        return task_assignments, selected_by_key, committed_activated, committed_memory, metrics

    def _wdmoe_run_task_trial(
        self,
        task,
        committed_assignments,
        activated,
        used_memory,
        theta: Optional[float],
        allow_pruning: bool,
    ):
        task_assignments: Dict[AssignmentKey, list[tuple[ServerId, ExpertId]]] = {}
        selected_by_key: Dict[AssignmentKey, Set[ExpertId]] = {}
        node_metrics: Dict[str, Dict[str, float]] = {}

        for subtask in self._topological_subtasks(task):
            key = (task.id, subtask.id)
            pairs, feasible_experts, metrics = self._wdmoe_assign_subtask(
                task=task,
                subtask=subtask,
                committed_assignments=committed_assignments,
                task_assignments=task_assignments,
                activated=activated,
                used_memory=used_memory,
                theta=theta,
                allow_pruning=allow_pruning,
            )
            task_assignments[key] = pairs
            selected_by_key[key] = feasible_experts
            node_metrics[subtask.id] = metrics

        return task_assignments, selected_by_key, activated, used_memory, node_metrics

    def _wdmoe_task_wlr(self, node_metrics: Mapping[str, Mapping[str, float]], key: str) -> float:
        values = [float(item.get(key, 0.0)) for item in node_metrics.values()]
        if not values:
            return 0.0
        return sum(values) / len(values)

    #這部份負責expert selection，因此對這個部份來修改
    def _wdmoe_assign_subtask(
        self,
        task,
        subtask,
        committed_assignments,
        task_assignments,
        activated,
        used_memory,
        theta: Optional[float],
        allow_pruning: bool,
    ):
        best_candidate = None

        for server_id in sorted(self.servers):
            # 1. Only consider experts stored on this server
            baseline_experts = self._wdmoe_candidate_experts_on_server(
                subtask=subtask,
                server_id=server_id,
                activated=activated,
                used_memory=used_memory,
            )

            '''
            # This server cannot execute the subtask without any expert.
            if not baseline_experts:
                continue
            
            以上3行是AI原本改的，他覺得「選 Top-K 個 expert 只是演算法策略，不是 System Model constraint」，所以就改成不強制選 K 個 expert，只要確保能至少選到一個 expert 就好。
            但是我不同意他的說法，所以我就改回來了。
            '''

            # This server cannot provide enough experts
            if len(baseline_experts) < self.top_k:
                continue

            # 2. Calculate latency assuming the entire node
            #    is executed on this server
            latency_vector = self._wdmoe_latency_vector_on_server(
                task=task,
                subtask=subtask,
                server_id=server_id,
                expert_ids=baseline_experts,
                committed_assignments=committed_assignments,
                task_assignments=task_assignments,
            )
            baseline_wlr = self._wdmoe_subtask_wlr(subtask, baseline_experts, latency_vector)
            similarity = self._wdmoe_weight_latency_similarity(subtask, latency_vector)
            candidate_experts = set(baseline_experts)

            # 3. WDMoE pruning
            attempted_pruning = False
            if allow_pruning and theta is not None and len(candidate_experts) > 1 and similarity <= theta:
                dropped = min(
                    candidate_experts,
                    key=lambda expert_id: (self._expert_score(subtask, expert_id), expert_id),
                )
                candidate_experts.remove(dropped)
                attempted_pruning = True

            # 4. Check performance-loss constraint
            if attempted_pruning:
                probability = self.evaluator.selection_probability(subtask, candidate_experts)
                full_features = set(subtask.required_features)
                loss = self.evaluator.performance_loss_from_probability(
                    subtask,
                    probability,
                    full_features,
                )
                loss_bound = self.evaluator.conformal_loss_threshold(subtask)
                if loss > loss_bound + 1e-12:
                    candidate_experts = set(baseline_experts)
                    attempted_pruning = False

            # 5. Evaluate this server
            selected_wlr = self._wdmoe_subtask_wlr(
                subtask,
                candidate_experts,
                latency_vector,
            )
            candidate = {
                "server_id": server_id,
                "experts": candidate_experts,
                "baseline_experts": baseline_experts,
                "baseline_wlr": baseline_wlr,
                "selected_wlr": selected_wlr,
                "similarity": similarity,
                "attempted_pruning": attempted_pruning,
            }
            # WDMoE: select candidate with highest WLR
            if best_candidate is None or candidate["selected_wlr"] > best_candidate["selected_wlr"]:
                best_candidate = candidate

        if best_candidate is None:
            raise RuntimeError(
                f"No feasible server for subtask "
                f"{task.id}-{subtask.id}"
            )

        # 6. Commit ONLY ONE server
        server_id = best_candidate["server_id"]
        selected_experts = best_candidate["experts"]

        pairs: list[tuple[ServerId, ExpertId]] = []
        for expert_id in sorted(selected_experts):
            pairs.append((server_id, expert_id))
            self._activate(server_id, expert_id, activated, used_memory)

        metrics = {
            "baseline_wlr": best_candidate["baseline_wlr"],
            "selected_wlr": best_candidate["selected_wlr"],
            "wlr_ratio": (best_candidate["selected_wlr"] / max(best_candidate["baseline_wlr"], 1e-12)),
            "similarity": best_candidate["similarity"],
            "theta": theta if theta is not None else 0.0,
            "attempted_pruning": best_candidate["attempted_pruning"],
            "pruned": (len(selected_experts) < len(best_candidate["baseline_experts"])),
            "baseline_expert_count": len(best_candidate["baseline_experts"]),
            "selected_expert_count": len(selected_experts),
            "selected_server": server_id,
        }

        return pairs, selected_experts, metrics

    #此為額外新增的function，用在「_wdmoe_assign_subtask」function內
    def _wdmoe_candidate_experts_on_server(
        self,
        subtask,
        server_id,
        activated,
        used_memory,
    ):
        server = self.servers[server_id]

        ranked_experts = sorted(
            [
                expert_id
                for expert_id in server.stored_experts
                if expert_id in self.experts
            ],
            key=lambda eid: (
                -self._expert_score(subtask, eid),
                eid,
            ),
        )

        selected = []
        memory = used_memory.get(server_id, 0.0)

        for expert_id in ranked_experts:
            extra_memory = 0.0

            if expert_id not in activated.get(server_id, set()):
                extra_memory = self.experts[expert_id].memory

            if memory + extra_memory > server.gpu_memory:
                continue

            selected.append(expert_id)
            memory += extra_memory

            if len(selected) >= self.top_k:
                break

        return set(selected)

    #此為額外新增的function，用在「_wdmoe_assign_subtask」function內
    #與「_wdmoe_latency_vector」function差別在於「server_id」變數是直接做為input，而不是在這個function內產生
    def _wdmoe_latency_vector_on_server(
        self,
        task,
        subtask,
        server_id,
        expert_ids,
        committed_assignments,
        task_assignments,
    ):
        merged = dict(committed_assignments)
        merged.update(task_assignments)

        forwarding_time = self._wdmoe_predecessor_forwarding_time(
            task,
            subtask,
            server_id,
            merged,
            {},
        )

        latencies = {}

        for expert_id in expert_ids:
            latencies[expert_id] = (
                forwarding_time
                + self.experts[expert_id].latency
            )

        return latencies

    def _wdmoe_subtask_wlr(self, subtask, selected_experts: Set[ExpertId], latency_vector: Mapping[ExpertId, float]) -> float:
        values = [
            self._expert_weight(subtask, expert_id) / max(latency_vector.get(expert_id, 1e6), 1e-12)
            for expert_id in selected_experts
        ]
        if not values:
            return 0.0
        return sum(values) / len(values)
        
    #不使用
    def _wdmoe_latency_vector(
        self,
        task,
        subtask,
        committed_assignments,
        task_assignments,
        activated,
        used_memory,
    ) -> Dict[ExpertId, float]:
        latencies: Dict[ExpertId, float] = {}
        merged = dict(committed_assignments)
        merged.update(task_assignments)
        for expert_id in self.experts:
            feasible_servers = self._wdmoe_feasible_servers_for_expert(expert_id, activated, used_memory)
            if not feasible_servers:
                latencies[expert_id] = 1e6
                continue
            server_id = feasible_servers[0]
            latencies[expert_id] = (
                self._wdmoe_predecessor_forwarding_time(task, subtask, server_id, merged, {})
                + self.experts[expert_id].latency
            )
        return latencies

    #此處也進行修改
    def _wdmoe_weight_latency_similarity(self, subtask, latency_vector):
        weights = []
        latencies = []
        for expert_id in latency_vector:
            weights.append(self._expert_weight(subtask, expert_id))
            #latencies.append(latency_vector.get(expert_id, 1e6))
            latencies.append(latency_vector[expert_id])
        weight_norm = math.sqrt(sum(value * value for value in weights))
        latency_norm = math.sqrt(sum(value * value for value in latencies))
        if weight_norm <= 0.0 or latency_norm <= 0.0:
            return 1.0
        return sum(w * t for w, t in zip(weights, latencies)) / (weight_norm * latency_norm)

    def _wdmoe_feasible_servers_for_expert(self, expert_id, activated, used_memory) -> list[ServerId]:
        expert = self.experts[expert_id]
        feasible: list[ServerId] = []
        for server_id in sorted(self.servers):
            server = self.servers[server_id]
            if expert_id not in server.stored_experts:
                continue
            memory_after = used_memory.get(server_id, 0.0)
            if expert_id not in activated.get(server_id, set()):
                memory_after += expert.memory
            if memory_after <= server.gpu_memory:
                feasible.append(server_id)
        return feasible

    #不使用
    def _best_server_for_expert_with_predecessors(
        self,
        expert_id,
        task,
        subtask,
        committed_assignments,
        task_assignments,
        activated,
        used_memory,
    ):
        feasible_servers = self._wdmoe_feasible_servers_for_expert(expert_id, activated, used_memory)
        if not feasible_servers:
            return None
        return self.placement_rng.choice(feasible_servers)

    #不使用
    def _best_wdmoe_loss_repair_candidate(
        self,
        task,
        subtask,
        selected_experts,
        committed_assignments,
        task_assignments,
        activated,
        used_memory,
    ):
        candidates = []
        for expert_id in sorted(self.experts):
            if expert_id in selected_experts:
                continue
            contribution = self.evaluator.expert_contribution(subtask, expert_id)
            if contribution <= 0.0:
                continue
            feasible_servers = self._wdmoe_feasible_servers_for_expert(expert_id, activated, used_memory)
            if not feasible_servers:
                continue
            expert = self.experts[expert_id]
            candidates.append((-contribution, expert.latency, expert_id, feasible_servers))
        if not candidates:
            return None
        _, _, expert_id, feasible_servers = min(candidates, key=lambda item: (item[0], item[1], item[2]))
        return self.placement_rng.choice(feasible_servers), expert_id

    def _wdmoe_predecessor_forwarding_time(self, task, subtask, target_server, assignments, finish_cache) -> float:
        ready = 0.0
        subtask_map = {item.id: item for item in task.subtasks}
        for pred_id in subtask.predecessors:
            pred = subtask_map.get(pred_id)
            if pred is None:
                continue
            pred_key = (task.id, pred_id)
            for src_server in self.evaluator.participating_servers(assignments, pred_key):
                candidate = finish_cache.get((pred_key, src_server), 0.0)
                if src_server != target_server:
                    pred_output_bits = self.evaluator.subtask_output_volume(pred_key, src_server, assignments)
                    candidate += pred_output_bits / self.evaluator.wired_rate(src_server, target_server)
                ready = max(ready, candidate)
        return ready

    def _expert_weight(self, subtask, expert_id: ExpertId) -> float:
        expert = self.experts[expert_id]
        if expert.index < 0 or expert.index >= len(subtask.gating_weights):
            return 0.0
        return float(subtask.gating_weights[expert.index])

    def _fallback_top_experts(self, subtask, count: int) -> list[ExpertId]:
        ranked = sorted(self.experts, key=lambda eid: self._expert_weight(subtask, eid), reverse=True)
        return ranked[:count]

    def _topological_subtasks(self, task):
        remaining = {subtask.id: subtask for subtask in task.subtasks}
        ordered = []
        while remaining:
            ready_ids = [
                subtask_id
                for subtask_id, subtask in remaining.items()
                if all(pred_id not in remaining for pred_id in subtask.predecessors)
            ]
            if not ready_ids:
                ordered.extend(remaining.values())
                break
            for subtask_id in sorted(ready_ids, key=self._id_sort_key):
                ordered.append(remaining.pop(subtask_id))
        return ordered

    @staticmethod
    def _task_subtask_by_id(task, subtask_id):
        return next((subtask for subtask in task.subtasks if subtask.id == subtask_id), None)


class WDMoELocationAwarePipeline(WDMoEExpertSelectionMixin, ChainedComparisonPipeline):
    """WDMoE expert selection + location_aware/K-means IoT grouping."""


class WDMoEGSSGDPipeline(WDMoEExpertSelectionMixin, HybridTopKGSSGDBackhaulPipeline):
    """WDMoE expert selection + GSSGD DBG IoT grouping."""


def _run_wdmoe_pipeline(
    pipeline_cls,
    method_name: str,
    grouping_name: str,
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
    wdmoe_initial_threshold: float = 0.8,
    wdmoe_threshold_step: float = 0.05,
    wdmoe_max_threshold: float = 1.0,
    wdmoe_wlr_target_ratio: float = 1.05,
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
    pipeline = pipeline_cls(
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
    pipeline.wdmoe_initial_threshold = wdmoe_initial_threshold
    pipeline.wdmoe_threshold_step = wdmoe_threshold_step
    pipeline.wdmoe_max_threshold = wdmoe_max_threshold
    pipeline.wdmoe_wlr_target_ratio = wdmoe_wlr_target_ratio
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
        "wdmoe_parameters": {
            "algorithm": "WDMoE-Based DAG-Aware Expert Selection",
            "initial_top_k": top_k,
            "initial_threshold_alpha": pipeline.wdmoe_initial_threshold,
            "threshold_step_delta": pipeline.wdmoe_threshold_step,
            "max_threshold": pipeline.wdmoe_max_threshold,
            "wlr_target_ratio_gamma": pipeline.wdmoe_wlr_target_ratio,
            "wlr_scope": "task graph average over DAG nodes",
            "latency_vector": "predecessor forwarding time + expert inference time",
            "cosine_similarity": "full gating weight vector versus full expert latency vector",
            "dropped_expert_rule": "paper-style theta search: each theta trial reruns the task graph; nodes with similarity <= theta drop the lowest-gating selected expert once",
            "activation_y_rule": "Y(s,p)=1 if any selected X(i,j,s,p)=1",
            "task_metrics": getattr(pipeline, "wdmoe_task_metrics", {}),
            "paper_preserving_note": "WDMoE uses graph-level WLR ratio to choose theta, then the selected experts are passed to the same IoT grouping and backhaul stages",
            "grouping": grouping_name,
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
            "gssgd_beamforming_gain_threshold": gssgd_beamforming_gain_threshold,
            "loss_threshold": loss_threshold,
            "lambda_reconstruction": lambda_reconstruction,
            "calibration_alpha": calibration_alpha,
            "reconstruction_sigma": reconstruction_sigma,
        },
    }
    payload = hybrid_result_to_jsonable(
        result,
        network_context,
        cost_units={
            "activation": c_act,
            "bandwidth": c_bw,
            "forwarding": c_fwd,
            "inference": inference_unit_cost_per_mb,
        },
    )
    payload["method"] = method_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    return result


def run_wdmoe_location_aware(
    graphs: Sequence[Any],
    output_path: Path,
    **kwargs: Any,
) -> ChainedPipelineResult:
    return _run_wdmoe_pipeline(
        WDMoELocationAwarePipeline,
        "wdmoe_location_aware",
        "location_aware/K-means IoT grouping",
        graphs,
        output_path,
        **kwargs,
    )


def run_wdmoe_gssgd(
    graphs: Sequence[Any],
    output_path: Path,
    **kwargs: Any,
) -> ChainedPipelineResult:
    return _run_wdmoe_pipeline(
        WDMoEGSSGDPipeline,
        "wdmoe_gssgd",
        "GSSGD DBG IoT grouping",
        graphs,
        output_path,
        **kwargs,
    )





