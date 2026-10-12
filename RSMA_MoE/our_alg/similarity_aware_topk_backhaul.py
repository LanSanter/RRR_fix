"""JRGEP pipeline: SCOPE placement, CFRG grouping, and Phase-III refinement."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from compared_method.hybrid_topk_location_aware_backhaul import (  # noqa: E402
    ChainedComparisonPipeline,
    ChainedPipelineResult,
    result_to_jsonable as hybrid_result_to_jsonable,
)
from our_alg.phase_1 import Phase1ServerExpertSelectionMixin  # noqa: E402
from our_alg.phase_2 import Phase2GroupingBackhaulMixin  # noqa: E402
from our_alg.phase_3 import Phase3RefinementPruningMixin  # noqa: E402
from physical_layer import NOISE_PSD_W_PER_HZ, TX_PSD_W_PER_HZ

from rsma_integration import build_scheduler_inputs  # noqa: E402
from utils.formulation import FormulationConfig, FormulationEvaluator, SubtaskSpec, TaskSpec  # noqa: E402

SimilarityKey = Tuple[str, str]
SimilarityRelation = Sequence[Tuple[SimilarityKey, SimilarityKey]]


class SimilarityAwareTopKBackhaulPipeline(
    Phase1ServerExpertSelectionMixin,
    Phase2GroupingBackhaulMixin,
    Phase3RefinementPruningMixin,
    ChainedComparisonPipeline,
):
    """JRGEP pipeline built from three explicit phases."""

    allow_expert_loss_repair = True
    activate_all_candidate_groups = False

    def __init__(
        self,
        servers: Iterable[Any],
        devices: Iterable[Any],
        experts: Mapping[str, Any],
        tasks: Iterable[Any],
        top_k: int = 2,
        rank_by: str = "gating",
        r_sim: Optional[SimilarityRelation] = None,
        similarity_weight: float = 0.05,
        enable_offline_group_pruning: bool = False,
        server_spread_penalty_multiplier: float = 1.0,
        predecessor_penalty_multiplier: float = 1.0,
        config: Optional[FormulationConfig] = None,
        placement_random_seed: Optional[int] = None,
    ):
        server_items = list(servers.values()) if isinstance(servers, Mapping) else list(servers)
        self.physical_wired_links: tuple[dict[str, Any], ...] = tuple()
        for raw_server in server_items:
            server_data = raw_server if isinstance(raw_server, Mapping) else vars(raw_server)
            links = server_data.get("physical_wired_links", ())
            if links:
                self.physical_wired_links = tuple(dict(link) for link in links)
                break
        super().__init__(
            servers=server_items,
            devices=devices,
            experts=experts,
            tasks=tasks,
            top_k=top_k,
            rank_by=rank_by,
            config=config,
            placement_random_seed=placement_random_seed,
        )
        self.r_sim = tuple(r_sim or ())
        self.similarity_weight = max(0.0, min(1.0, float(similarity_weight)))
        self.enable_offline_group_pruning = bool(enable_offline_group_pruning)
        self.server_spread_penalty_multiplier = float(server_spread_penalty_multiplier)
        self.predecessor_penalty_multiplier = float(predecessor_penalty_multiplier)
        self.phase1_report: dict[str, Any] = {
            "mode": "SCOPE",
            "similarity_pairs": len(self.r_sim),
            "similarity_weight": self.similarity_weight,
            "similarity_smoothing_applied": False,
            "server_spread_penalty_multiplier": self.server_spread_penalty_multiplier,
            "predecessor_penalty_multiplier": self.predecessor_penalty_multiplier,
        }
        self.phase2_report: dict[str, Any] = {"mode": "CFRG"}
        self.phase3_report: dict[str, Any] = {
            "mode": "FAR",
            "max_iterations": 2,
            "max_joint_assignments": 32,
            "max_private_candidates": 48,
            "physical_link_count": len(self.physical_wired_links),
        }
        self.post_pruning_report: dict[str, Any] = {
            "enabled": self.enable_offline_group_pruning,
            "mode": "offline_final_group_pruning",
        }

    def run(self) -> ChainedPipelineResult:
        assignments, selected_probability, selected_experts, activated, used_memory = self._assign_topk_experts()
        initial_result = self._run_phase2_for_assignments(
            assignments,
            selected_probability,
            selected_experts,
            activated,
            used_memory,
        )
        refined_result = self._phase3_refine(assignments, initial_result)
        if self.enable_offline_group_pruning:
            final_result = self._run_offline_group_pruning(refined_result)
        else:
            self.post_pruning_report.update(
                {
                    "enabled": False,
                    "removed_groups": 0,
                    "cost_before": round(refined_result.total_cost, 6),
                    "cost_after": round(refined_result.total_cost, 6),
                }
            )
            setattr(refined_result, "post_pruning_report", self.post_pruning_report)
            final_result = refined_result

        setattr(final_result, "phase1_report", self.phase1_report)
        setattr(final_result, "phase2_report", self.phase2_report)
        setattr(final_result, "phase3_report", self.phase3_report)
        setattr(final_result, "post_pruning_report", self.post_pruning_report)
        return final_result

    def _apply_similarity_smoothing(self) -> None:
        subtasks = {
            (task.id, subtask.id): subtask
            for task in self.tasks.values()
            for subtask in task.subtasks
        }
        neighbor_map: dict[SimilarityKey, list[SubtaskSpec]] = {}
        for left, right in self.r_sim:
            left_key = (str(left[0]), str(left[1]))
            right_key = (str(right[0]), str(right[1]))
            if left_key in subtasks and right_key in subtasks:
                neighbor_map.setdefault(left_key, []).append(subtasks[right_key])
                neighbor_map.setdefault(right_key, []).append(subtasks[left_key])
        if not neighbor_map:
            return

        smoothed_tasks: dict[str, TaskSpec] = {}
        for task in self.tasks.values():
            smoothed_subtasks = []
            for subtask in task.subtasks:
                key = (task.id, subtask.id)
                neighbors = neighbor_map.get(key, [])
                if not neighbors or not subtask.gating_weights:
                    smoothed_subtasks.append(subtask)
                    continue
                vector_len = len(subtask.gating_weights)
                mean_vector = []
                for index in range(vector_len):
                    values = [neighbor.gating_weights[index] for neighbor in neighbors if index < len(neighbor.gating_weights)]
                    mean_vector.append(sum(values) / len(values) if values else 0.0)
                mixed = [
                    (1.0 - self.similarity_weight) * subtask.gating_weights[index]
                    + self.similarity_weight * mean_vector[index]
                    for index in range(vector_len)
                ]
                total = sum(mixed)
                if total > 0.0:
                    mixed = [value / total for value in mixed]
                smoothed_subtasks.append(replace(subtask, gating_weights=tuple(mixed)))
            smoothed_tasks[task.id] = replace(task, subtasks=smoothed_subtasks)
        self.tasks = smoothed_tasks
        self.evaluator = FormulationEvaluator(self.servers, self.devices, self.experts, self.tasks, self.config)


def build_similarity_relations_from_gating(
    tasks: Iterable[Mapping[str, Any]],
    max_pairs: int = 4,
) -> list[tuple[SimilarityKey, SimilarityKey]]:
    """Build cross-task r_sim pairs by gating-vector cosine similarity."""
    nodes: list[tuple[SimilarityKey, tuple[float, ...]]] = []
    for task in tasks:
        task_id = str(task.get("id"))
        for subtask in task.get("subtasks", []):
            gating = tuple(float(value) for value in subtask.get("gating_weights", ()))
            if gating:
                nodes.append(((task_id, str(subtask.get("id"))), gating))

    candidates: list[tuple[float, SimilarityKey, SimilarityKey]] = []
    for left_index, (left_key, left_gating) in enumerate(nodes):
        for right_key, right_gating in nodes[left_index + 1 :]:
            if left_key[0] == right_key[0]:
                continue
            candidates.append((_cosine_similarity(left_gating, right_gating), left_key, right_key))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [(left, right) for _, left, right in candidates[: max(0, max_pairs)]]


def _cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    length = min(len(left), len(right))
    if length == 0:
        return 0.0
    dot = sum(left[index] * right[index] for index in range(length))
    left_norm = math.sqrt(sum(value * value for value in left[:length]))
    right_norm = math.sqrt(sum(value * value for value in right[:length]))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def run_similarity_aware_topk_backhaul(
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
    r_sim: Optional[SimilarityRelation] = None,
    similarity_weight: float = 0.05,
    enable_offline_group_pruning: bool = False,
    server_spread_penalty_multiplier: float = 1.0,
    predecessor_penalty_multiplier: float = 1.0,
) -> ChainedPipelineResult:
    del clusters_per_server, kmeans_iterations, gssgd_beamforming_gain_threshold
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
    pipeline = SimilarityAwareTopKBackhaulPipeline(
        servers=inputs.servers,
        devices=inputs.devices,
        experts=inputs.experts,
        tasks=inputs.tasks,
        top_k=top_k,
        rank_by=rank_by,
        r_sim=r_sim,
        similarity_weight=similarity_weight,
        enable_offline_group_pruning=enable_offline_group_pruning,
        server_spread_penalty_multiplier=server_spread_penalty_multiplier,
        predecessor_penalty_multiplier=predecessor_penalty_multiplier,
        placement_random_seed=random_seed,
        config=FormulationConfig(
            max_group_size=max_group_size,
            min_rate=min_rate,
            c_bw=c_bw,
            c_act=c_act,
            c_fwd=c_fwd,
            c_inf=inference_unit_cost_per_mb,
            derive_bandwidth=True,
            noise_power=noise_power,
            common_power_ratio=common_power_ratio,
            default_power=max_device_power,
            default_feature_bits=default_feature_bits,
            feature_bits_by_name=inputs.feature_bits_by_name,
            default_loss_threshold=loss_threshold if loss_threshold is not None else 3.0,
            lambda_reconstruction=lambda_reconstruction,
            calibration_alpha=calibration_alpha,
            reconstruction_sigma=reconstruction_sigma,
            wavelength=wavelength,
        ),
    )
    result = pipeline.run()
    network_context = {
        "experts": inputs.experts,
        "servers": inputs.servers,
        "devices": inputs.devices,
        "topology": inputs.topology,
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
            "feature_bits_range": feature_bits_range,
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
            "min_rate": min_rate,
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
    payload["method"] = "jrgep"
    payload["algorithm_reports"] = {
        "phase1": getattr(result, "phase1_report", {}),
        "phase2": getattr(result, "phase2_report", {}),
        "phase3": getattr(result, "phase3_report", {}),
        "post_pruning": getattr(result, "post_pruning_report", {}),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    return result
