from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from physical_layer import NOISE_PSD_W_PER_HZ, TX_PSD_W_PER_HZ
from generate_dags import DAGSpec


BASE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ExperimentConfig:
    # Reproducibility / output.
    auto_random_seed: bool = True
    num_runs: int = 100
    num_dags: int = 8
    random_seed: int = 42
    network_random_seed: int = 1
    output_dir: Path = BASE_DIR / "generated_dags"
    results_dir: Path = BASE_DIR / "results"
    image_dpi: int = 180

    # DAG / task graph settings.
    common_dag_spec: DAGSpec = field(
        default_factory=lambda: DAGSpec(
            num_nodes=10,
            num_branches=(2,4),
            branch_length_range=(1, 5),
            depth=8,
            max_width=5,
            edge_probability=0.2,
            allow_skip_edges=False,
            allow_early_branch_end=True,
            single_source=True,
            single_sink=True,
        )
    )
    # A shared-subtask ratio counts node instances whose unique ID is reused
    # by another DAG. Each shared ID appears at most once in any one DAG.
    shared_subtask_ratio: float = 0.4
    shared_subtask_graph_count_range: tuple[int, int] = (2, 4)

    # Expert / task metadata settings.
    num_experts: int = 16
    expert_memory_sizes_mb: tuple[float, ...] = (
        14.16,
        17.30,
        20.45,
        23.59,
        26.74,
        29.88,
        33.03,
        36.18,
        14.35350005,
        31.98747879,
        17.25680641,
        34.72570471,
        21.79071219,
        18.28066402,
        30.92318615,
        34.76502272,
    )
    expert_memory_range: tuple[float, float] = (14.16, 36.18)
    expert_inference_times_ms: tuple[float, ...] = (
        35.0,
        45.0,
        58.0,
        72.0,
        88.0,
        105.0,
        125.0,
        150.0,
        92.85908316,
        62.21828652,
        126.85002314,
        128.20651959,
        89.01387579,
        56.13957668,
        112.22036417,
        140.48559838,
    )
    reasoning_data_sizes_bytes: tuple[int, ...] = (256, 512, 1024)
    task_deadline_seconds_range: tuple[int, int] = (30, 120)
    num_iot_features: int = 100
    iot_features_per_node_range: tuple[int, int] = (2, 5)
    gating_peak_count_range: tuple[int, int] = (1, 1)
    gating_peak_mass_range: tuple[float, float] = (0.75, 0.85)

    # Performance loss settings.
    loss_threshold: Optional[float] = None
    lambda_reconstruction: float = 2.0
    calibration_alpha: float = 0.1
    reconstruction_sigma: float = 1.0
    reconstruction_error_range: tuple[float, float] = (0.8, 1.5)
    num_calibration_samples: int = 20
    calibration_loss_range: tuple[float, float] = (1.5, 3.3)

    # Which methods to run. The default workflow compares four baselines and JRGEP.
    run_hybrid_topk_gssgd_backhaul: bool = True
    run_hybrid_topk_location_aware_backhaul: bool = True
    run_wdmoe_gssgd: bool = True
    run_wdmoe_location_aware: bool = True
    run_our_similarity_aware_topk_backhaul: bool = True

    # Shared Top-K and grouping settings.
    topk_k: int = 3
    topk_rank_by: str = "gating"
    location_aware_grouping_clusters_per_server: Optional[int] = None
    location_aware_grouping_kmeans_iterations: int = 20

    # Our algorithm settings.
    similarity_weight: float = 0.05
    random_similarity_pair_count: Optional[int] = 4
    enable_offline_group_pruning: bool = False
    jrgep_server_spread_penalty_multiplier: float = 1.0
    jrgep_predecessor_penalty_multiplier: float = 1.0
    # WDMoE pruning settings. Paper-style WDMoE searches theta at task/DAG
    # scope: each theta trial reruns all subtasks and stops when graph WLR
    # ratio passes gamma.
    wdmoe_initial_threshold: float = 0.8
    wdmoe_threshold_step: float = 0.05
    wdmoe_max_threshold: float = 1.0
    wdmoe_wlr_target_ratio: float = 1.05

    # RSMA / edge network settings. These follow the original SIoT-RSMA defaults
    # as closely as possible, with multiple edge servers and coverage limits.
    num_edge_servers: int = 15 #ori is 10
    num_iot_devices: int = 100
    area_size: float = 2000.0
    cell_radius: float = 350.0
    num_antennas: int = 1
    iot_features_per_device_range: tuple[int, int] = (4, 8)
    iot_server_feature_overlap_ratio: float = 0.0
    iot_global_random_feature_fraction: float = 0.2
    experts_per_server: int = 4
    server_gpu_memory_range: tuple[float, float] = (150.0, 168.24)
    # Paper: [10, 100] MB/s = [80, 800] Mbit/s.
    wired_rate_range: tuple[float, float] = (80e6, 800e6)
    wired_extra_link_probability: float = 0.05
    wired_edge_weight_range: tuple[float, float] = (1.0, 3.0) #8-10
    wavelength: float = 0.125
    # Paper: 16 KiB per feature.
    default_feature_bits: float = 16.0 * 1024.0 * 8.0
    feature_bits_range: tuple[float, float] = (
        16.0 * 1024.0 * 8.0,
        16.0 * 1024.0 * 8.0,
    )
    # Paper PSDs, converted from dBm/Hz to W/Hz.
    noise_power: float = NOISE_PSD_W_PER_HZ
    common_power_ratio: float = 0.6
    # Kept under the existing field name for compatibility; interpreted as PSD.
    max_device_power: float = TX_PSD_W_PER_HZ
    max_group_size: int = 4
    min_rate: float = 0.4e6

    # Objective unit costs from the simulation setup.
    c_bw: float = 0.000025
    c_act: float = 0.6
    c_fwd: float = 1.0
    c_inf: float = 0.06
