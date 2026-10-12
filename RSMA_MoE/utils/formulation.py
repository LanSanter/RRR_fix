from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import ceil, cos, isfinite, pi
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


from physical_layer import NOISE_PSD_W_PER_HZ, TX_PSD_W_PER_HZ, spectral_efficiency


ServerId = str
ExpertId = str
DeviceId = str
TaskId = str
SubtaskId = str
GroupId = str
AssignmentKey = Tuple[TaskId, SubtaskId]
FeatureDemandKey = Tuple[TaskId, SubtaskId, ServerId, str]
FeatureSource = Tuple[ServerId, GroupId]


class InfeasibleBandwidthBudget(ValueError):
    """Raised when an RSMA demand cannot meet its uplink time/rate budget."""


@dataclass(frozen=True)
class ExpertSpec:
    id: ExpertId
    index: int
    memory: float = 0.0
    latency: float = 1.0
    inference_cost: float = 0.0


@dataclass(frozen=True)
class ServerSpec:
    id: ServerId
    gpu_memory: float
    stored_experts: Set[ExpertId] = field(default_factory=set)
    covered_devices: Set[DeviceId] = field(default_factory=set)
    wired_rates: Dict[ServerId, float] = field(default_factory=dict)
    wired_weights: Dict[ServerId, float] = field(default_factory=dict)
    active_experts: Set[ExpertId] = field(default_factory=set)


@dataclass(frozen=True)
class DeviceSpec:
    id: DeviceId
    home_server: Optional[ServerId] = None
    features: Set[str] = field(default_factory=set)
    position: Tuple[float, float] = (0.0, 0.0)
    channel_gain: Dict[ServerId, float] = field(default_factory=dict)
    channel_vector: Dict[ServerId, Tuple[complex, ...]] = field(default_factory=dict)
    spatial_correlation: Dict[ServerId, Dict[DeviceId, float]] = field(default_factory=dict)
    phase_angle: Dict[ServerId, float] = field(default_factory=dict)
    distance: Dict[ServerId, float] = field(default_factory=dict)
    max_power: float = TX_PSD_W_PER_HZ  # W/Hz


@dataclass(frozen=True)
class SubtaskSpec:
    id: SubtaskId
    unique_id: Optional[str] = None
    required_features: Set[str] = field(default_factory=set)
    predecessors: Set[SubtaskId] = field(default_factory=set)
    output_tokens: int = 256
    feature_volume_bits: float = 0.0
    gating_weights: Tuple[float, ...] = field(default_factory=tuple)
    expert_confidence: Tuple[float, ...] = field(default_factory=tuple)
    reconstruction_loss: float = 0.0
    reconstruction_errors_by_feature: Dict[str, float] = field(default_factory=dict)
    calibration_losses: Tuple[float, ...] = field(default_factory=tuple)
    loss_threshold: Optional[float] = None
    max_loss: Optional[float] = None


@dataclass(frozen=True)
class TaskSpec:
    id: TaskId
    subtasks: List[SubtaskSpec]
    deadline: float = float("inf")


@dataclass(frozen=True)
class GroupSpec:
    server_id: ServerId
    id: GroupId
    devices: Tuple[DeviceId, ...]
    bandwidth: float
    required_features: Set[str] = field(default_factory=set)
    # Per-stream transmit PSD overrides, in W/Hz.
    common_power: Dict[DeviceId, float] = field(default_factory=dict)
    private_power: Dict[DeviceId, float] = field(default_factory=dict)


@dataclass
class FormulationConfig:
    c_bw: float = 1e-3
    c_act: float = 1.0
    c_fwd: float = 1.0
    c_inf: Optional[float] = None
    derive_bandwidth: bool = True
    default_channel_gain: float = 1.0
    default_power: float = TX_PSD_W_PER_HZ  # W/Hz
    common_power_ratio: float = 0.6
    noise_power: float = NOISE_PSD_W_PER_HZ  # W/Hz; legacy field name
    wavelength: float = 0.125
    max_group_size: int = 4
    gssgd_beamforming_gain_threshold: float = 0.0
    min_rate: float = 1.0
    output_token_bits: float = 16.0
    default_feature_bits: float = 12_000.0
    feature_bits_by_name: Dict[str, float] = field(default_factory=dict)
    default_loss_threshold: float = 3.0
    lambda_reconstruction: float = 0.1
    calibration_alpha: float = 0.1
    reconstruction_sigma: float = 1.0
    min_selection_probability: float = 1e-12
    deadline_tolerance: float = 1e-3
    enforce_single_server_assignment: bool = True

    def __post_init__(self) -> None:
        if not isfinite(self.noise_power) or self.noise_power <= 0.0:
            raise ValueError("noise_power must be a positive, finite PSD in W/Hz.")
        if not isfinite(self.common_power_ratio) or not 0.0 <= self.common_power_ratio <= 1.0:
            raise ValueError("common_power_ratio must be in [0, 1].")


@dataclass
class ObjectiveBreakdown:
    activation_cost: float
    forwarding_cost: float
    bandwidth_cost: float
    inference_cost: float = 0.0

    @property
    def total_cost(self) -> float:
        return (
            self.activation_cost
            + self.forwarding_cost
            + self.bandwidth_cost
            + self.inference_cost
        )


@dataclass
class TimingBreakdown:
    uplink_time: Dict[Tuple[ServerId, GroupId], float]
    feature_ready_time: Dict[Tuple[AssignmentKey, ServerId], float]
    predecessor_ready_time: Dict[Tuple[AssignmentKey, ServerId], float]
    subtask_start_time: Dict[Tuple[AssignmentKey, ServerId], float]
    subtask_finish_time: Dict[Tuple[AssignmentKey, ServerId], float]
    subtask_finish_global: Dict[AssignmentKey, float]
    task_finish_time: Dict[TaskId, float]


@dataclass
class EvaluationResult:
    objective: ObjectiveBreakdown
    timing: TimingBreakdown
    violations: List[str]
    common_rates: Dict[Tuple[ServerId, GroupId], float]
    private_rates: Dict[Tuple[ServerId, GroupId, DeviceId], float]


def obj_id(value: Any) -> str:
    return str(getattr(value, "id", value.get("id") if isinstance(value, Mapping) else value))


def as_set(value: Any) -> Set[str]:
    if value is None:
        return set()
    if isinstance(value, Mapping):
        return {str(v) for v in value.keys()}
    if isinstance(value, set):
        return {str(v) for v in value}
    if isinstance(value, (list, tuple)):
        return {str(v) for v in value}
    return {str(value)}


def _first(mapping: Mapping[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


def _scalar_gain(value: Any) -> float:
    if isinstance(value, complex):
        return abs(value)
    try:
        import numpy as np

        arr = np.asarray(value)
        if arr.size:
            return float(np.linalg.norm(arr))
    except Exception:
        pass
    return abs(float(value))


def _complex_value(value: Any) -> complex:
    if isinstance(value, complex):
        return value
    if isinstance(value, Mapping):
        return complex(float(value.get("real", 0.0)), float(value.get("imag", 0.0)))
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return complex(float(value[0]), float(value[1]))
    return complex(float(value), 0.0)


def _complex_vector(value: Any) -> Tuple[complex, ...]:
    if value is None:
        return tuple()
    if isinstance(value, (list, tuple)):
        return tuple(_complex_value(item) for item in value)
    return (_complex_value(value),)


def normalize_experts(experts: Mapping[str, Any]) -> Dict[ExpertId, ExpertSpec]:
    normalized: Dict[ExpertId, ExpertSpec] = {}
    for fallback_index, (expert_id, raw) in enumerate(experts.items()):
        data = raw if isinstance(raw, Mapping) else vars(raw)
        parsed_index = _first(data, ("index", "expert_index"), None)
        if parsed_index is None:
            try:
                parsed_index = int(str(expert_id).rsplit("_", 1)[1])
            except (IndexError, ValueError):
                parsed_index = fallback_index
        normalized[str(expert_id)] = ExpertSpec(
            id=str(expert_id),
            index=int(parsed_index),
            memory=float(_first(data, ("memory", "mem", "memory_size"), 0.0)),
            latency=float(_first(data, ("latency", "inference_latency", "tau"), 1.0)),
            inference_cost=float(_first(data, ("inference_cost", "c_inf", "execution_cost"), 0.0)),
        )
    return normalized


def normalize_servers(servers: Iterable[Any]) -> Dict[ServerId, ServerSpec]:
    normalized: Dict[ServerId, ServerSpec] = {}
    for raw in servers:
        data = raw if isinstance(raw, Mapping) else vars(raw)
        server_id = str(_first(data, ("server_id", "id"), len(normalized)))
        normalized[server_id] = ServerSpec(
            id=server_id,
            gpu_memory=float(_first(data, ("gpu_memory", "gpu_memory_capacity", "gpu_mem"), 0.0)),
            stored_experts=as_set(_first(data, ("stored_experts", "existing_experts", "experts"), set())),
            covered_devices=as_set(_first(data, ("device_ids", "covered_devices", "coverage"), set())),
            wired_rates={str(k): float(v) for k, v in dict(data.get("wired_rates", {})).items()},
            wired_weights={str(k): float(v) for k, v in dict(data.get("wired_weights", {})).items()},
            active_experts=as_set(_first(data, ("active_experts", "activated_expert"), set())),
        )
    return normalized


def normalize_devices(devices: Iterable[Any]) -> Dict[DeviceId, DeviceSpec]:
    normalized: Dict[DeviceId, DeviceSpec] = {}
    for raw in devices:
        data = raw if isinstance(raw, Mapping) else vars(raw)
        device_id = str(_first(data, ("device_id", "id"), len(normalized)))
        connected = list(_first(data, ("connected_edge_server_ids", "covered_servers"), []) or [])
        home_server = _first(data, ("home_server", "server_id"), None)
        if home_server is None and connected:
            home_server = connected[0]
        gains = _first(data, ("channel_gain", "channel_gains"), {}) or {}
        vectors = _first(data, ("channel_vector", "channel_vectors"), {}) or {}
        raw_correlation = _first(data, ("spatial_correlation", "correlation"), {}) or {}
        raw_position = _first(data, ("position", "location", "xy"), (0.0, 0.0))
        if isinstance(raw_position, (list, tuple)) and len(raw_position) >= 2:
            position = (float(raw_position[0]), float(raw_position[1]))
        else:
            position = (0.0, 0.0)
        normalized[device_id] = DeviceSpec(
            id=device_id,
            home_server=str(home_server) if home_server is not None else None,
            features=as_set(_first(data, ("features", "feature_set"), set())),
            position=position,
            channel_gain={str(k): _scalar_gain(v) for k, v in dict(gains).items()},
            channel_vector={str(k): _complex_vector(v) for k, v in dict(vectors).items()},
            spatial_correlation={
                str(server_id): {str(other_id): float(value) for other_id, value in dict(values).items()}
                for server_id, values in dict(raw_correlation).items()
                if isinstance(values, Mapping)
            },
            phase_angle={str(k): float(v) for k, v in dict(data.get("phase_angle", {})).items()},
            distance={str(k): float(v) for k, v in dict(data.get("distance", data.get("distances", {}))).items()} if isinstance(data.get("distance", data.get("distances", {})), Mapping) else {},
            max_power=float(_first(data, ("max_power", "power_budget"), TX_PSD_W_PER_HZ)),
        )
    return normalized


def normalize_tasks(tasks: Iterable[Any]) -> Dict[TaskId, TaskSpec]:
    normalized: Dict[TaskId, TaskSpec] = {}
    for task_index, raw in enumerate(tasks):
        data = raw if isinstance(raw, Mapping) else vars(raw)
        task_id = str(_first(data, ("task_id", "id"), task_index))
        raw_subtasks = _first(data, ("subtasks", "nodes"), [])
        subtasks: List[SubtaskSpec] = []
        for sub_index, sub_raw in enumerate(raw_subtasks):
            sub = sub_raw if isinstance(sub_raw, Mapping) else vars(sub_raw)
            subtask_id = str(_first(sub, ("subtask_id", "id", "node_id"), sub_index))
            required_features = as_set(_first(sub, ("required_features", "features"), set()))
            raw_errors = _first(sub, ("reconstruction_errors_by_feature", "reconstruction_errors"), {})
            if isinstance(raw_errors, Mapping):
                reconstruction_errors = {
                    str(feature): float(error)
                    for feature, error in raw_errors.items()
                    if str(feature) in required_features
                }
            elif isinstance(raw_errors, (list, tuple)):
                reconstruction_errors = {
                    feature: float(error)
                    for feature, error in zip(sorted(required_features), raw_errors)
                }
            else:
                reconstruction_errors = {}
            subtasks.append(
                SubtaskSpec(
                    id=subtask_id,
                    unique_id=str(_first(sub, ("unique_id", "unique_subtask_id"), subtask_id)),
                    required_features=required_features,
                    predecessors=as_set(_first(sub, ("predecessors", "parents"), set())),
                    output_tokens=int(_first(sub, ("output_tokens", "max_output_tokens"), 256)),
                    feature_volume_bits=float(_first(sub, ("feature_volume_bits", "feature_bits"), 0.0)),
                    gating_weights=tuple(float(value) for value in _first(sub, ("gating_weights", "gating"), [])),
                    expert_confidence=tuple(float(value) for value in _first(sub, ("expert_confidence", "pi"), [])),
                    reconstruction_loss=float(_first(sub, ("reconstruction_loss", "L_rec"), 0.0)),
                    reconstruction_errors_by_feature=reconstruction_errors,
                    calibration_losses=tuple(float(value) for value in _first(sub, ("calibration_losses", "D_cal"), [])),
                    loss_threshold=_first(sub, ("loss_threshold", "q", "q_hat"), None),
                    max_loss=_first(sub, ("max_loss", "loss"), None),
                )
            )
        normalized[task_id] = TaskSpec(
            id=task_id,
            subtasks=subtasks,
            deadline=float(_first(data, ("deadline", "D_max"), float("inf"))),
        )
    return normalized


class FormulationEvaluator:
    def __init__(
        self,
        servers: Mapping[ServerId, ServerSpec],
        devices: Mapping[DeviceId, DeviceSpec],
        experts: Mapping[ExpertId, ExpertSpec],
        tasks: Mapping[TaskId, TaskSpec],
        config: Optional[FormulationConfig] = None,
    ):
        self.servers = dict(servers)
        self.devices = dict(devices)
        self.experts = dict(experts)
        self.tasks = dict(tasks)
        self.config = config or FormulationConfig()

    def evaluate(
        self,
        assignments: Mapping[AssignmentKey, Sequence[Tuple[ServerId, ExpertId]]],
        groups: Sequence[GroupSpec],
        backhaul: Optional[Set[Tuple[ServerId, GroupId, ServerId]]] = None,
        subtask_features: Optional[Mapping[Any, Set[str]]] = None,
        feature_source_map: Optional[Mapping[FeatureDemandKey, FeatureSource]] = None,
    ) -> EvaluationResult:
        backhaul = backhaul or set()
        groups = self.resolve_group_bandwidths(
            groups,
            assignments,
            backhaul,
            subtask_features=subtask_features,
            feature_source_map=feature_source_map,
        )
        common_rates, private_rates = self.compute_rates(groups)
        objective = self.compute_objective(assignments, groups, backhaul)
        timing = self.compute_timing(
            assignments,
            groups,
            backhaul,
            common_rates,
            private_rates,
            subtask_features,
            feature_source_map=feature_source_map,
        )
        violations = self.check_constraints(assignments, groups, timing, common_rates, private_rates, subtask_features)
        return EvaluationResult(objective, timing, violations, common_rates, private_rates)

    def compute_objective(
        self,
        assignments: Mapping[AssignmentKey, Sequence[Tuple[ServerId, ExpertId]]],
        groups: Sequence[GroupSpec],
        backhaul: Set[Tuple[ServerId, GroupId, ServerId]],
    ) -> ObjectiveBreakdown:
        activated = {(server_id, expert_id) for pairs in assignments.values() for server_id, expert_id in pairs}
        activation_cost = self.config.c_act * sum(
            self.experts[expert_id].memory
            for _, expert_id in activated
            if expert_id in self.experts
        )
        inference_cost = sum(
            (
                self.config.c_inf * self.experts[expert_id].memory
                if self.config.c_inf is not None
                else self.experts[expert_id].inference_cost
            )
            for pairs in assignments.values()
            for _, expert_id in pairs
            if expert_id in self.experts
        )
        bandwidth_cost = self.config.c_bw * sum(group.bandwidth for group in groups)
        forwarding_cost = 0.0

        for task in self.tasks.values():
            subtask_map = {sub.id: sub for sub in task.subtasks}
            for child in task.subtasks:
                child_key = (task.id, child.id)
                for pred_id in child.predecessors:
                    pred_key = (task.id, pred_id)
                    pred = subtask_map.get(pred_id)
                    if pred is None:
                        continue
                    for src_server in self.participating_servers(assignments, pred_key):
                        for dst_server in self.participating_servers(assignments, child_key):
                            if src_server != dst_server:
                                forwarding_cost += self.config.c_fwd * self.wired_weight(src_server, dst_server)

        for src_server, group_id, dst_server in backhaul:
            group = next((g for g in groups if g.server_id == src_server and g.id == group_id), None)
            if group and src_server != dst_server:
                forwarding_cost += self.config.c_fwd * self.wired_weight(src_server, dst_server)

        return ObjectiveBreakdown(activation_cost, forwarding_cost, bandwidth_cost, inference_cost)

    def resolve_group_bandwidths(
        self,
        groups: Sequence[GroupSpec],
        assignments: Optional[Mapping[AssignmentKey, Sequence[Tuple[ServerId, ExpertId]]]] = None,
        backhaul: Optional[Set[Tuple[ServerId, GroupId, ServerId]]] = None,
        subtask_features: Optional[Mapping[Any, Set[str]]] = None,
        feature_source_map: Optional[Mapping[FeatureDemandKey, FeatureSource]] = None,
    ) -> List[GroupSpec]:
        """Return groups with bandwidth derived from their own dependent subtasks."""
        if not self.config.derive_bandwidth:
            return list(groups)
        budgets = self.group_bandwidth_budgets(
            groups,
            assignments or {},
            backhaul or set(),
            subtask_features=subtask_features,
            feature_source_map=feature_source_map,
        )
        return [
            replace(
                group,
                bandwidth=self.derive_group_bandwidth_for_budget(
                    group,
                    budgets.get((group.server_id, group.id), self.bandwidth_time_budget()),
                ),
            )
            for group in groups
        ]

    def bandwidth_time_budget(self) -> float:
        finite_deadlines = [
            task.deadline
            for task in self.tasks.values()
            if task.deadline != float("inf")
        ]
        if not finite_deadlines:
            return 1.0
        return max(min(finite_deadlines), 1e-12)

    def group_bandwidth_budgets(
        self,
        groups: Sequence[GroupSpec],
        assignments: Mapping[AssignmentKey, Sequence[Tuple[ServerId, ExpertId]]],
        backhaul: Set[Tuple[ServerId, GroupId, ServerId]],
        subtask_features: Optional[Mapping[Any, Set[str]]] = None,
        feature_source_map: Optional[Mapping[FeatureDemandKey, FeatureSource]] = None,
    ) -> Dict[Tuple[ServerId, GroupId], float]:
        dependencies = self.group_dependencies(
            groups,
            assignments,
            backhaul,
            subtask_features=subtask_features,
            feature_source_map=feature_source_map,
        )
        latest_start = self.latest_subtask_start_times(assignments)
        fallback = self.bandwidth_time_budget()
        budgets: Dict[Tuple[ServerId, GroupId], float] = {}
        for group in groups:
            group_key = (group.server_id, group.id)
            dependent_items = dependencies.get(group_key, set())
            if not dependent_items:
                budgets[group_key] = fallback
                continue
            candidates: List[float] = []
            for (task_id, subtask_id), target_server in dependent_items:
                task = self.tasks.get(task_id)
                if task is None or task.deadline == float("inf"):
                    candidates.append(fallback)
                    continue
                backhaul_time = 0.0
                if group.server_id != target_server:
                    backhaul_time = self.group_feature_volume(group) / self.wired_rate(
                        group.server_id,
                        target_server,
                    )
                remaining = latest_start.get((task_id, subtask_id), task.deadline) - backhaul_time
                if remaining <= 0.0:
                    raise InfeasibleBandwidthBudget(
                        f"RSMA group {group.server_id}:{group.id} cannot serve "
                        f"{task_id}:{subtask_id} on {target_server}: remaining "
                        f"uplink time is {remaining:.6g}s"
                    )
                candidates.append(remaining)
            budgets[group_key] = min(candidates)
        return budgets

    def group_dependencies(
        self,
        groups: Sequence[GroupSpec],
        assignments: Mapping[AssignmentKey, Sequence[Tuple[ServerId, ExpertId]]],
        backhaul: Set[Tuple[ServerId, GroupId, ServerId]],
        subtask_features: Optional[Mapping[Any, Set[str]]] = None,
        feature_source_map: Optional[Mapping[FeatureDemandKey, FeatureSource]] = None,
    ) -> Dict[Tuple[ServerId, GroupId], Set[Tuple[AssignmentKey, ServerId]]]:
        group_by_key = {(group.server_id, group.id): group for group in groups}
        feature_to_groups: Dict[str, List[GroupSpec]] = {}
        for group in groups:
            for feature in self.group_payload_features(group):
                feature_to_groups.setdefault(feature, []).append(group)

        dependencies: Dict[
            Tuple[ServerId, GroupId],
            Set[Tuple[AssignmentKey, ServerId]],
        ] = {
            key: set() for key in group_by_key
        }
        for task in self.tasks.values():
            for subtask in task.subtasks:
                key = (task.id, subtask.id)
                for target_server in self.participating_servers(assignments, key):
                    features = (
                        set(subtask.required_features)
                        if subtask_features is None
                        else self.features_for_server(subtask_features, key, target_server)
                    )
                    for feature in features:
                        if feature_source_map is None:
                            group = self.serving_group_for_feature(
                                feature,
                                target_server,
                                feature_to_groups,
                                backhaul,
                            )
                        else:
                            source = feature_source_map.get(
                                (task.id, subtask.id, target_server, feature)
                            )
                            group = None if source is None else group_by_key.get(tuple(source))
                            if group is not None:
                                source_edge = (group.server_id, group.id, target_server)
                                if (
                                    feature not in self.group_payload_features(group)
                                    or (
                                        group.server_id != target_server
                                        and source_edge not in backhaul
                                    )
                                ):
                                    group = None
                        if group is not None:
                            dependencies.setdefault((group.server_id, group.id), set()).add(
                                (key, target_server)
                            )
        return dependencies

    def serving_group_for_feature(
        self,
        feature: str,
        target_server: ServerId,
        feature_to_groups: Mapping[str, Sequence[GroupSpec]],
        backhaul: Set[Tuple[ServerId, GroupId, ServerId]],
    ) -> Optional[GroupSpec]:
        candidates = list(feature_to_groups.get(feature, []))
        local = [group for group in candidates if group.server_id == target_server]
        if local:
            return sorted(local, key=lambda group: group.id)[0]
        for src_server, group_id, dst_server in sorted(backhaul):
            if dst_server != target_server:
                continue
            group = next(
                (candidate for candidate in candidates if candidate.server_id == src_server and candidate.id == group_id),
                None,
            )
            if group is not None:
                return group
        return None

    def downstream_compute_times(
        self,
        assignments: Mapping[AssignmentKey, Sequence[Tuple[ServerId, ExpertId]]],
    ) -> Dict[AssignmentKey, float]:
        downstream: Dict[AssignmentKey, float] = {}
        for task in self.tasks.values():
            children = {sub.id: [] for sub in task.subtasks}
            subtask_by_id = {sub.id: sub for sub in task.subtasks}
            for subtask in task.subtasks:
                for pred_id in subtask.predecessors:
                    if pred_id in children:
                        children[pred_id].append(subtask)

            def visit(subtask: SubtaskSpec) -> float:
                key = (task.id, subtask.id)
                if key in downstream:
                    return downstream[key]
                compute = max(
                    (
                        self.server_computation_time(key, server_id, assignments)
                        for server_id in self.participating_servers(assignments, key)
                    ),
                    default=0.0,
                )
                child_tail = max((visit(child) for child in children.get(subtask.id, [])), default=0.0)
                downstream[key] = compute + child_tail
                return downstream[key]

            for subtask in subtask_by_id.values():
                visit(subtask)
        return downstream

    def dependency_forwarding_time(
        self,
        task_id: TaskId,
        predecessor_id: SubtaskId,
        successor_id: SubtaskId,
        assignments: Mapping[AssignmentKey, Sequence[Tuple[ServerId, ExpertId]]],
    ) -> float:
        """Worst predecessor-output delay for one DAG edge under C8 placement."""
        predecessor_key = (task_id, predecessor_id)
        successor_key = (task_id, successor_id)
        predecessor_servers = self.participating_servers(assignments, predecessor_key)
        successor_servers = self.participating_servers(assignments, successor_key)
        delay = 0.0
        for source_server in predecessor_servers:
            output_bits = self.subtask_output_volume(
                predecessor_key,
                source_server,
                assignments,
            )
            for target_server in successor_servers:
                if source_server == target_server:
                    continue
                delay = max(
                    delay,
                    output_bits / self.wired_rate(source_server, target_server),
                )
        return delay

    def latest_subtask_start_times(
        self,
        assignments: Mapping[AssignmentKey, Sequence[Tuple[ServerId, ExpertId]]],
    ) -> Dict[AssignmentKey, float]:
        """Reverse-DAG latest starts used by the CFRG bandwidth budget.

        A sink must start no later than its deadline minus its computation
        time.  Every other node must finish, and forward its output when the
        next node is remote, before the successor's latest start.
        """
        latest: Dict[AssignmentKey, float] = {}
        visiting: Set[AssignmentKey] = set()

        for task in self.tasks.values():
            subtask_by_id = {subtask.id: subtask for subtask in task.subtasks}
            children: Dict[SubtaskId, List[SubtaskId]] = {
                subtask.id: [] for subtask in task.subtasks
            }
            for subtask in task.subtasks:
                for predecessor_id in subtask.predecessors:
                    if predecessor_id in children:
                        children[predecessor_id].append(subtask.id)

            def visit(subtask_id: SubtaskId) -> float:
                key = (task.id, subtask_id)
                if key in latest:
                    return latest[key]
                if key in visiting:
                    raise ValueError(f"Task {task.id} is not a DAG")
                visiting.add(key)
                compute_time = max(
                    (
                        self.server_computation_time(key, server_id, assignments)
                        for server_id in self.participating_servers(assignments, key)
                    ),
                    default=0.0,
                )
                successors = children.get(subtask_id, [])
                if not successors:
                    value = task.deadline - compute_time
                else:
                    value = min(
                        visit(successor_id)
                        - compute_time
                        - self.dependency_forwarding_time(
                            task.id,
                            subtask_id,
                            successor_id,
                            assignments,
                        )
                        for successor_id in successors
                    )
                visiting.remove(key)
                latest[key] = value
                return value

            for subtask_id in subtask_by_id:
                visit(subtask_id)
        return latest

    def group_spectral_efficiencies(
        self,
        group: GroupSpec,
    ) -> Tuple[float, Dict[DeviceId, float]]:
        if not group.devices:
            return 0.0, {}

        # Signal, interference, and noise are all PSDs (W/Hz).
        common_signal = sum(
            self.effective_channel_gain(device_id, group.server_id, group) ** 2
            * group.common_power.get(device_id, self.default_common_power(device_id))
            for device_id in group.devices
        )
        private_interference = sum(
            self.effective_channel_gain(device_id, group.server_id, group) ** 2
            * group.private_power.get(device_id, self.default_private_power(device_id))
            for device_id in group.devices
        )
        db_gain = self.distributed_beamforming_gain(group)
        common_sinr = common_signal / (private_interference + self.config.noise_power)
        common_efficiency = spectral_efficiency(max(db_gain, 0.0) * max(common_sinr, 0.0))

        private_efficiencies: Dict[DeviceId, float] = {}
        for device_id in group.devices:
            signal = self.effective_channel_gain(device_id, group.server_id, group) ** 2
            signal *= group.private_power.get(device_id, self.default_private_power(device_id))
            interference = 0.0
            for other_id in group.devices:
                if other_id == device_id:
                    continue
                interference += (
                    self.effective_channel_gain(other_id, group.server_id, group) ** 2
                    * group.private_power.get(other_id, self.default_private_power(other_id))
                )
            private_sinr = signal / (interference + self.config.noise_power)
            private_efficiencies[device_id] = spectral_efficiency(max(private_sinr, 0.0))
        return common_efficiency, private_efficiencies

    def derive_group_bandwidth(self, group: GroupSpec) -> float:
        return self.derive_group_bandwidth_for_budget(group, self.bandwidth_time_budget())

    def derive_group_bandwidth_for_budget(self, group: GroupSpec, budget: float) -> float:
        if not group.devices:
            return 0.0
        budget = float(budget)
        if budget != budget or budget <= 0.0:
            raise InfeasibleBandwidthBudget(
                f"RSMA group {group.server_id}:{group.id} has non-positive "
                f"uplink time budget {budget:.6g}s"
            )
        common_efficiency, private_efficiencies = self.group_spectral_efficiencies(group)

        def required(demand: float, efficiency: float) -> float:
            if demand <= 0.0:
                return 0.0
            if efficiency <= 0.0:
                raise InfeasibleBandwidthBudget(
                    f"RSMA group {group.server_id}:{group.id} has positive "
                    "demand on a zero-efficiency stream"
                )
            return demand / efficiency

        common_volume = self.group_common_volume(group)
        common_required = required(common_volume, common_efficiency) / budget

        private_required = 0.0
        min_rate_required = 0.0
        for device_id in group.devices:
            private_volume = self.group_private_volume(group, device_id)
            private_efficiency = max(private_efficiencies.get(device_id, 0.0), 0.0)
            private_required = max(
                private_required,
                required(private_volume, private_efficiency) / budget,
            )
            aggregate_efficiency = common_efficiency + private_efficiency
            min_rate_required = max(
                min_rate_required,
                required(self.config.min_rate, aggregate_efficiency),
            )

        return max(common_required, private_required, min_rate_required)

    def compute_rates(
        self, groups: Sequence[GroupSpec]
    ) -> Tuple[Dict[Tuple[ServerId, GroupId], float], Dict[Tuple[ServerId, GroupId, DeviceId], float]]:
        common_rates: Dict[Tuple[ServerId, GroupId], float] = {}
        private_rates: Dict[Tuple[ServerId, GroupId, DeviceId], float] = {}
        for group in groups:
            common_efficiency, private_efficiencies = self.group_spectral_efficiencies(group)
            common_rates[(group.server_id, group.id)] = group.bandwidth * common_efficiency
            for device_id, efficiency in private_efficiencies.items():
                private_rates[(group.server_id, group.id, device_id)] = group.bandwidth * efficiency
        return common_rates, private_rates

    def compute_timing(
        self,
        assignments: Mapping[AssignmentKey, Sequence[Tuple[ServerId, ExpertId]]],
        groups: Sequence[GroupSpec],
        backhaul: Set[Tuple[ServerId, GroupId, ServerId]],
        common_rates: Mapping[Tuple[ServerId, GroupId], float],
        private_rates: Mapping[Tuple[ServerId, GroupId, DeviceId], float],
        subtask_features: Optional[Mapping[Any, Set[str]]] = None,
        feature_source_map: Optional[Mapping[FeatureDemandKey, FeatureSource]] = None,
    ) -> TimingBreakdown:
        uplink: Dict[Tuple[ServerId, GroupId], float] = {}
        for group in groups:
            common_rate = max(common_rates.get((group.server_id, group.id), 0.0), 1e-12)
            common_time = self.group_common_volume(group) / common_rate
            private_time = 0.0
            for device_id in group.devices:
                rate = max(private_rates.get((group.server_id, group.id, device_id), 0.0), 1e-12)
                private_volume = self.group_private_volume(group, device_id)
                private_time = max(private_time, private_volume / rate)
            uplink[(group.server_id, group.id)] = max(common_time, private_time)

        feature_ready: Dict[Tuple[AssignmentKey, ServerId], float] = {}
        pred_ready: Dict[Tuple[AssignmentKey, ServerId], float] = {}
        start: Dict[Tuple[AssignmentKey, ServerId], float] = {}
        finish: Dict[Tuple[AssignmentKey, ServerId], float] = {}
        finish_global: Dict[AssignmentKey, float] = {}
        task_finish: Dict[TaskId, float] = {}

        group_by_device: Dict[DeviceId, GroupSpec] = {}
        group_by_key: Dict[FeatureSource, GroupSpec] = {}
        for group in groups:
            group_by_key[(group.server_id, group.id)] = group
            for device_id in group.devices:
                group_by_device[device_id] = group

        for task in self.tasks.values():
            remaining = {sub.id: sub for sub in task.subtasks}
            while remaining:
                ready = [
                    sub for sub in remaining.values()
                    if all((task.id, pred) in finish_global or pred not in remaining for pred in sub.predecessors)
                ]
                if not ready:
                    ready = [remaining[next(iter(remaining))]]
                for sub in ready:
                    key = (task.id, sub.id)
                    for server_id in self.participating_servers(assignments, key):
                        server_features = None if subtask_features is None else self.features_for_server(subtask_features, key, server_id)
                        feature_ready[(key, server_id)] = self.feature_ready_time(
                            server_id,
                            sub,
                            group_by_device,
                            uplink,
                            backhaul,
                            server_features,
                            assignment_key=key,
                            group_by_key=group_by_key,
                            feature_source_map=feature_source_map,
                        )
                        pred_ready[(key, server_id)] = self.predecessor_ready_time(task, sub, server_id, assignments, finish)
                        start[(key, server_id)] = max(feature_ready[(key, server_id)], pred_ready[(key, server_id)])
                        finish[(key, server_id)] = start[(key, server_id)] + self.server_computation_time(key, server_id, assignments)
                    if assignments.get(key):
                        finish_global[key] = max(finish[(key, s)] for s in self.participating_servers(assignments, key))
                    del remaining[sub.id]
            task_keys = [k for k in finish_global if k[0] == task.id]
            task_finish[task.id] = max((finish_global[k] for k in task_keys), default=0.0)

        return TimingBreakdown(uplink, feature_ready, pred_ready, start, finish, finish_global, task_finish)

    def check_constraints(
        self,
        assignments: Mapping[AssignmentKey, Sequence[Tuple[ServerId, ExpertId]]],
        groups: Sequence[GroupSpec],
        timing: TimingBreakdown,
        common_rates: Mapping[Tuple[ServerId, GroupId], float],
        private_rates: Mapping[Tuple[ServerId, GroupId, DeviceId], float],
        subtask_features: Optional[Mapping[AssignmentKey, Set[str]]] = None,
    ) -> List[str]:
        violations: List[str] = []

        deadline_tolerance = max(self.config.deadline_tolerance, 0.0)
        for task_id, finish_time in timing.task_finish_time.items():
            deadline = self.tasks[task_id].deadline
            if finish_time > deadline + deadline_tolerance:
                violations.append(f"C1 deadline: task {task_id} finish {finish_time:.6g} > {deadline:.6g}")

        activated_by_server: Dict[ServerId, Set[ExpertId]] = {server_id: set() for server_id in self.servers}
        for pairs in assignments.values():
            for server_id, expert_id in pairs:
                activated_by_server.setdefault(server_id, set()).add(expert_id)
        used_memory: Dict[ServerId, float] = {
            server_id: sum(self.experts[expert_id].memory for expert_id in expert_ids)
            for server_id, expert_ids in activated_by_server.items()
        }
        for server_id, memory in used_memory.items():
            capacity = self.servers[server_id].gpu_memory
            if memory > capacity:
                violations.append(f"C2 GPU memory: server {server_id} uses {memory:.6g} > {capacity:.6g}")

        for task in self.tasks.values():
            for sub in task.subtasks:
                threshold = self.conformal_loss_threshold(sub)
                key = (task.id, sub.id)
                loss = sub.max_loss if sub.max_loss is not None else self.estimated_loss_for_assignment(key, assignments, subtask_features)
                if loss > threshold:
                    violations.append(f"C3 performance loss: {(task.id, sub.id)} loss {loss:.6g} > {threshold:.6g}")

        if self.config.enforce_single_server_assignment:
            for task in self.tasks.values():
                for subtask in task.subtasks:
                    key = (task.id, subtask.id)
                    pairs = list(assignments.get(key, []))
                    servers = {server_id for server_id, _ in pairs}
                    expert_ids = [expert_id for _, expert_id in pairs]
                    if len(servers) != 1:
                        violations.append(
                            f"C8 single-server assignment: {key} uses {len(servers)} servers"
                        )
                    if not pairs:
                        violations.append(
                            f"C8 single-server assignment: {key} has no selected expert"
                        )
                    if len(set(expert_ids)) != len(expert_ids):
                        violations.append(
                            f"C8 single-server assignment: {key} selects a duplicate expert"
                        )
                    for server_id, expert_id in pairs:
                        server = self.servers.get(server_id)
                        if server is None:
                            violations.append(
                                f"C9 expert activation: {key} selects {expert_id} on unknown server {server_id}"
                            )
                            continue
                        if expert_id not in server.stored_experts:
                            violations.append(
                                f"C9 expert activation: {key} selects expert {expert_id} "
                                f"not deployed on server {server_id}"
                            )

        for group in groups:
            if len(group.devices) > self.config.max_group_size:
                violations.append(f"C4 RSMA grouping: group {(group.server_id, group.id)} size {len(group.devices)} > {self.config.max_group_size}")

        seen_devices: Set[DeviceId] = set()
        for group in groups:
            for device_id in group.devices:
                if device_id in seen_devices:
                    violations.append(f"C5 IoT grouping: device {device_id} appears in multiple groups")
                seen_devices.add(device_id)


        tolerance = 1e-9
        for key in assignments:
            for server_id in self.participating_servers(assignments, key):
                start_time = timing.subtask_start_time.get((key, server_id), 0.0)
                feature_ready = timing.feature_ready_time.get((key, server_id), 0.0)
                predecessor_ready = timing.predecessor_ready_time.get((key, server_id), 0.0)
                if feature_ready == float("inf"):
                    violations.append(
                        f"C6 feature availability: {key} cannot receive all required features on server {server_id}"
                    )
                if predecessor_ready == float("inf"):
                    violations.append(
                        f"C6 predecessor availability: {key} cannot receive all predecessor outputs on server {server_id}"
                    )
                if start_time + tolerance < feature_ready:
                    violations.append(
                        f"C6 execution order: {key} on server {server_id} "
                        f"starts {start_time:.6g} before features ready {feature_ready:.6g}"
                    )
                if start_time + tolerance < predecessor_ready:
                    violations.append(
                        f"C6 execution order: {key} on server {server_id} "
                        f"starts {start_time:.6g} before predecessors ready {predecessor_ready:.6g}"
                    )

        violations.extend(self.check_minimum_transmission_rate(groups, common_rates, private_rates))

        return violations

    def check_minimum_transmission_rate(
        self,
        groups: Sequence[GroupSpec],
        common_rates: Mapping[Tuple[ServerId, GroupId], float],
        private_rates: Mapping[Tuple[ServerId, GroupId, DeviceId], float],
    ) -> List[str]:
        """C7: minimum aggregate RSMA transmission rate constraint.

        B(n,g)=1 exactly when n is a member of group g. Non-members have
        B(n,g)=0 and therefore do not need an aggregate-rate check.
        """
        violations: List[str] = []
        r_min = max(float(self.config.min_rate), 0.0)
        if r_min <= 0.0:
            return violations

        for group in groups:
            group_key = (group.server_id, group.id)
            common_rate = common_rates.get(group_key, 0.0)
            for device_id in group.devices:
                private_rate = private_rates.get((group.server_id, group.id, device_id), 0.0)
                aggregate_rate = common_rate + private_rate
                # Ignore only floating-point round-off at the C7 boundary.
                # Example: 399999.99999999994 should satisfy r_min=400000.0.
                if aggregate_rate < r_min - 1e-6:
                    violations.append(
                        f"C7 min aggregate rate: device {device_id} in group {group_key} "
                        f"common={common_rate:.6g} + private={private_rate:.6g} "
                        f"= {aggregate_rate:.6g} < {r_min:.6g}"
                    )
        return violations

    def wired_rate(self, src: ServerId, dst: ServerId) -> float:
        if src == dst:
            return float("inf")
        return self.servers.get(src, ServerSpec(src, 0)).wired_rates.get(dst, 1e9)

    def wired_weight(self, src: ServerId, dst: ServerId) -> float:
        if src == dst:
            return 0.0
        return self.servers.get(src, ServerSpec(src, 0)).wired_weights.get(dst, 1.0)

    def channel_gain(self, device_id: DeviceId, server_id: ServerId) -> float:
        vector = self.devices[device_id].channel_vector.get(server_id, tuple())
        if vector:
            return sum(abs(value) ** 2 for value in vector) ** 0.5
        return self.devices[device_id].channel_gain.get(server_id, self.config.default_channel_gain)

    def spatial_correlation(self, device_id: DeviceId, other_id: DeviceId, server_id: ServerId) -> float:
        if device_id == other_id:
            return 1.0
        stored = self.devices[device_id].spatial_correlation.get(server_id, {}).get(other_id)
        if stored is not None:
            return max(0.0, min(float(stored), 1.0))
        first = self.devices[device_id].channel_vector.get(server_id, tuple())
        second = self.devices[other_id].channel_vector.get(server_id, tuple())
        if not first or not second or len(first) != len(second):
            return 1.0
        numerator = abs(sum(a.conjugate() * b for a, b in zip(first, second))) ** 2
        denominator = sum(abs(a) ** 2 for a in first) * sum(abs(b) ** 2 for b in second)
        if denominator <= 0.0:
            return 0.0
        return max(0.0, min(float(numerator / denominator), 1.0))

    def average_group_correlation(self, device_id: DeviceId, server_id: ServerId, group: GroupSpec) -> float:
        others = [other_id for other_id in group.devices if other_id != device_id]
        if not others:
            return 0.0
        return sum(self.spatial_correlation(device_id, other_id, server_id) for other_id in others) / len(others)

    def common_message_ratio(self, group: GroupSpec) -> float:
        transmitted = self.group_transmitted_features(group)
        if not transmitted:
            return 0.0
        return len(self.group_common_features(group)) / len(transmitted)

    def db_phase_term(self, device_id: DeviceId, server_id: ServerId) -> complex:
        device = self.devices[device_id]
        theta = device.phase_angle.get(server_id, 0.0)
        distance = device.distance.get(server_id, 0.0)
        phase = 2.0 * pi * distance * cos(theta) / max(self.config.wavelength, 1e-12)
        return complex_phase(phase)

    def distributed_beamforming_gain(self, group: GroupSpec) -> float:
        """Distributed beamforming gain for the RSMA common stream."""
        if not group.devices:
            return 0.0

        weighted_terms = []
        magnitudes = []
        for device_id in group.devices:
            gain = self.channel_gain(device_id, group.server_id)
            power = group.common_power.get(device_id, self.default_common_power(device_id))
            weight = (max(power, 0.0) ** 0.5) * gain
            weighted_terms.append(weight * self.db_phase_term(device_id, group.server_id))
            magnitudes.append(weight)

        denominator = sum(magnitudes)
        if denominator <= 0.0:
            return 0.0
        return abs(sum(weighted_terms)) / denominator
    def effective_channel_gain(self, device_id: DeviceId, server_id: ServerId, group: GroupSpec) -> float:
        del group
        return self.channel_gain(device_id, server_id)

    def db_gain(self, device_id: DeviceId, server_id: ServerId) -> float:
        device = self.devices[device_id]
        theta = device.phase_angle.get(server_id, 0.0)
        distance = device.distance.get(server_id, 0.0)
        phase = 2.0 * pi * distance * cos(theta) / max(self.config.wavelength, 1e-12)
        return abs(self.channel_gain(device_id, server_id) * complex_phase(phase))

    def default_common_power(self, device_id: DeviceId) -> float:
        return self.devices[device_id].max_power * self.config.common_power_ratio

    def default_private_power(self, device_id: DeviceId) -> float:
        return self.devices[device_id].max_power * (1.0 - self.config.common_power_ratio)

    def feature_bits(self, feature: str) -> float:
        return max(float(self.config.feature_bits_by_name.get(feature, self.config.default_feature_bits)), 0.0)

    def feature_set_volume(self, features: Iterable[str]) -> float:
        return sum(self.feature_bits(feature) for feature in set(features))

    def device_feature_volume(self, device_id: DeviceId) -> float:
        return self.feature_set_volume(self.devices[device_id].features)

    def group_transmitted_features(self, group: GroupSpec) -> Set[str]:
        """All features uploaded by the IoT devices selected for this group."""
        transmitted: Set[str] = set()
        for device_id in group.devices:
            transmitted.update(self.devices[device_id].features)
        return transmitted

    def group_payload_features(self, group: GroupSpec) -> Set[str]:
        """Complete feature union uploaded by the selected IoT devices."""
        return self.group_transmitted_features(group)

    def group_common_features(self, group: GroupSpec) -> Set[str]:
        """Intersection of the complete feature sets of all selected IoTs."""
        if not group.devices:
            return set()

        common = set(self.devices[group.devices[0]].features)
        for device_id in group.devices[1:]:
            common &= set(self.devices[device_id].features)
        return common

    def group_private_features(self, group: GroupSpec, device_id: DeviceId) -> Set[str]:
        """Complete device feature set excluding the group-common features."""
        common = self.group_common_features(group)
        return set(self.devices[device_id].features) - common

    def group_feature_volume(self, group: GroupSpec) -> float:
        """Complete IoT-group volume forwarded over each active backhaul."""
        return self.feature_set_volume(self.group_transmitted_features(group))

    def group_common_volume(self, group: GroupSpec) -> float:
        """D_cmn(g) = number of common features times D_feat."""
        common_features = self.group_common_features(group)
        d_feat = max(float(self.config.default_feature_bits), 0.0)
        return len(common_features) * d_feat

    def group_private_volume(self, group: GroupSpec, device_id: DeviceId) -> float:
        """D_priv_n(g) = number of private features times D_feat."""
        private_features = self.group_private_features(group, device_id)
        d_feat = max(float(self.config.default_feature_bits), 0.0)
        return len(private_features) * d_feat

    def feature_ready_time(
        self,
        target_server: ServerId,
        subtask: SubtaskSpec,
        group_by_device: Mapping[DeviceId, GroupSpec],
        uplink: Mapping[Tuple[ServerId, GroupId], float],
        backhaul: Set[Tuple[ServerId, GroupId, ServerId]],
        required_features: Optional[Set[str]] = None,
        *,
        assignment_key: Optional[AssignmentKey] = None,
        group_by_key: Optional[Mapping[FeatureSource, GroupSpec]] = None,
        feature_source_map: Optional[Mapping[FeatureDemandKey, FeatureSource]] = None,
    ) -> float:
        features = set(subtask.required_features if required_features is None else required_features)
        if not features:
            return 0.0
        ready = 0.0
        for feature in features:
            if feature_source_map is not None:
                if assignment_key is None or group_by_key is None:
                    return float("inf")
                source = feature_source_map.get(
                    (assignment_key[0], assignment_key[1], target_server, feature)
                )
                if source is None:
                    return float("inf")
                source = tuple(source)
                group = group_by_key.get(source)
                if group is None or feature not in self.group_payload_features(group):
                    return float("inf")
                arrival = uplink.get(source, float("inf"))
                if group.server_id != target_server:
                    edge = (group.server_id, group.id, target_server)
                    if edge not in backhaul:
                        return float("inf")
                    arrival += self.group_feature_volume(group) / self.wired_rate(
                        group.server_id,
                        target_server,
                    )
                ready = max(ready, arrival)
                continue

            best = float("inf")
            for device_id, device in self.devices.items():
                if feature not in device.features or device_id not in group_by_device:
                    continue
                group = group_by_device[device_id]
                if feature not in self.group_payload_features(group):
                    continue
                arrival = uplink.get((group.server_id, group.id), 0.0)
                if group.server_id != target_server:
                    if (group.server_id, group.id, target_server) not in backhaul:
                        continue
                    arrival += self.group_feature_volume(group) / self.wired_rate(group.server_id, target_server)
                best = min(best, arrival)
            ready = max(ready, best)
        return ready

    def predecessor_ready_time(
        self,
        task: TaskSpec,
        subtask: SubtaskSpec,
        target_server: ServerId,
        assignments: Mapping[AssignmentKey, Sequence[Tuple[ServerId, ExpertId]]],
        finish: Mapping[Tuple[AssignmentKey, ServerId], float],
    ) -> float:
        ready = 0.0
        subtask_map = {sub.id: sub for sub in task.subtasks}
        for pred_id in subtask.predecessors:
            pred = subtask_map.get(pred_id)
            pred_key = (task.id, pred_id)
            for src_server in self.participating_servers(assignments, pred_key):
                pred_output_bits = self.subtask_output_volume(pred_key, src_server, assignments)
                candidate = finish.get((pred_key, src_server), 0.0)
                if src_server != target_server:
                    candidate += pred_output_bits / self.wired_rate(src_server, target_server)
                ready = max(ready, candidate)
        return ready

    def computation_time(self, server_id: ServerId, expert_id: ExpertId) -> float:
        del server_id
        return self.experts[expert_id].latency

    def participating_servers(
        self,
        assignments: Mapping[AssignmentKey, Sequence[Tuple[ServerId, ExpertId]]],
        key: AssignmentKey,
    ) -> List[ServerId]:
        return sorted({server_id for server_id, _ in assignments.get(key, [])})

    def selected_experts_on_server(
        self,
        assignments: Mapping[AssignmentKey, Sequence[Tuple[ServerId, ExpertId]]],
        key: AssignmentKey,
        server_id: ServerId,
    ) -> List[ExpertId]:
        return [expert_id for pair_server, expert_id in assignments.get(key, []) if pair_server == server_id]

    def server_computation_time(
        self,
        key: AssignmentKey,
        server_id: ServerId,
        assignments: Mapping[AssignmentKey, Sequence[Tuple[ServerId, ExpertId]]],
    ) -> float:
        return max(
            (self.experts[expert_id].latency for expert_id in self.selected_experts_on_server(assignments, key, server_id)),
            default=0.0,
        )

    def subtask_output_volume(
        self,
        key: AssignmentKey,
        server_id: ServerId,
        assignments: Mapping[AssignmentKey, Sequence[Tuple[ServerId, ExpertId]]],
    ) -> float:
        subtask = self.subtask_by_key(key)
        output_tokens = subtask.output_tokens if subtask is not None else 256
        del server_id, assignments
        return output_tokens * self.config.output_token_bits

    def subtask_by_key(self, key: AssignmentKey) -> Optional[SubtaskSpec]:
        task = self.tasks.get(key[0])
        if task is None:
            return None
        return next((sub for sub in task.subtasks if sub.id == key[1]), None)

    def expert_contribution(self, subtask: SubtaskSpec, expert_id: ExpertId) -> float:
        expert = self.experts[expert_id]
        index = expert.index
        if index < 0:
            return 0.0
        if index >= len(subtask.gating_weights):
            return 0.0
        if index >= len(subtask.expert_confidence):
            return 0.0
        return subtask.gating_weights[index] * subtask.expert_confidence[index]

    def selection_probability(
        self,
        subtask: SubtaskSpec,
        expert_ids: Iterable[ExpertId],
    ) -> float:
        return sum(
            self.expert_contribution(subtask, expert_id)
            for expert_id in expert_ids
            if expert_id in self.experts
        )

    def features_for_server(
        self,
        subtask_features: Mapping[Any, Set[str]],
        key: AssignmentKey,
        server_id: ServerId,
    ) -> Set[str]:
        server_key = (key[0], key[1], server_id)
        if server_key in subtask_features:
            return set(subtask_features.get(server_key, set()))
        return set(subtask_features.get(key, set()))

    def average_reconstruction_loss_for_assignment(
        self,
        subtask: SubtaskSpec,
        key: AssignmentKey,
        assignments: Mapping[AssignmentKey, Sequence[Tuple[ServerId, ExpertId]]],
        subtask_features: Optional[Mapping[Any, Set[str]]] = None,
    ) -> float:
        target_servers = self.participating_servers(assignments, key)
        if not target_servers:
            return subtask.reconstruction_loss

        weighted_loss = 0.0
        total_experts = 0
        for server_id in target_servers:
            expert_count = len(self.selected_experts_on_server(assignments, key, server_id))
            if expert_count <= 0:
                continue
            selected = None if subtask_features is None else self.features_for_server(subtask_features, key, server_id)
            server_loss = self.reconstruction_loss_for_features(subtask, selected)
            weighted_loss += expert_count * server_loss
            total_experts += expert_count
        if total_experts <= 0:
            return subtask.reconstruction_loss
        return weighted_loss / total_experts

    def performance_loss_from_probability_and_reconstruction(
        self,
        probability: float,
        reconstruction_loss: float,
    ) -> float:
        if probability <= self.config.min_selection_probability:
            return float("inf")
        return (1.0 / probability) + (self.config.lambda_reconstruction * reconstruction_loss)

    def estimated_loss_for_assignment(
        self,
        key: AssignmentKey,
        assignments: Mapping[AssignmentKey, Sequence[Tuple[ServerId, ExpertId]]],
        subtask_features: Optional[Mapping[Any, Set[str]]] = None,
    ) -> float:
        pairs = assignments.get(key, [])
        if not pairs:
            return float("inf")
        subtask = self.subtask_by_key(key)
        if subtask is None:
            return float("inf")
        probability = self.selection_probability(
            subtask,
            (expert_id for _, expert_id in pairs),
        )
        reconstruction_loss = self.average_reconstruction_loss_for_assignment(
            subtask,
            key,
            assignments,
            subtask_features,
        )
        return self.performance_loss_from_probability_and_reconstruction(probability, reconstruction_loss)

    def reconstruction_loss_for_features(
        self,
        subtask: SubtaskSpec,
        selected_features: Optional[Set[str]],
    ) -> float:
        required_features = set(subtask.required_features)
        if not required_features:
            return 0.0
        selected = set() if selected_features is None else set(selected_features)
        missing = required_features - selected
        if not missing:
            return 0.0
        sigma_sq = max(self.config.reconstruction_sigma ** 2, 1e-12)
        if subtask.reconstruction_errors_by_feature:
            squared_error = sum(
                subtask.reconstruction_errors_by_feature.get(feature, 1.0) ** 2
                for feature in missing
            )
            return squared_error / (2.0 * sigma_sq)
        return len(missing) / (2.0 * sigma_sq)

    def performance_loss_from_probability(
        self,
        subtask: SubtaskSpec,
        probability: float,
        selected_features: Optional[Set[str]] = None,
    ) -> float:
        if probability <= self.config.min_selection_probability:
            return float("inf")
        reconstruction_loss = self.reconstruction_loss_for_features(subtask, selected_features)
        return (1.0 / probability) + (
            self.config.lambda_reconstruction * reconstruction_loss
        )

    def conformal_loss_threshold(self, subtask: SubtaskSpec) -> float:
        if subtask.loss_threshold is not None:
            return float(subtask.loss_threshold)
        if not subtask.calibration_losses:
            return self.config.default_loss_threshold

        losses = sorted(subtask.calibration_losses)
        required_count = ceil((len(losses) + 1) * (1.0 - self.config.calibration_alpha))
        if required_count <= 0:
            return losses[0]
        if required_count > len(losses):
            return float("inf")
        return losses[required_count - 1]

    def required_selection_probability(self, subtask: SubtaskSpec, selected_features: Optional[Set[str]] = None) -> float:
        threshold = self.conformal_loss_threshold(subtask)
        if threshold == float("inf"):
            return 0.0
        reconstruction_loss = self.reconstruction_loss_for_features(subtask, selected_features)
        denominator = threshold - (self.config.lambda_reconstruction * reconstruction_loss)
        if denominator <= 0.0:
            return float("inf")
        return 1.0 / denominator

    def estimated_loss(
        self,
        key: AssignmentKey,
        assignments: Mapping[AssignmentKey, Sequence[Tuple[ServerId, ExpertId]]],
        selected_features: Optional[Set[str]] = None,
    ) -> float:
        pairs = assignments.get(key, [])
        if not pairs:
            return float("inf")
        subtask = self.subtask_by_key(key)
        if subtask is None:
            return float("inf")
        probability = self.selection_probability(
            subtask,
            (expert_id for _, expert_id in pairs),
        )
        return self.performance_loss_from_probability(subtask, probability, selected_features)


def complex_phase(phase: float) -> complex:
    from cmath import exp

    return exp(1j * phase)
