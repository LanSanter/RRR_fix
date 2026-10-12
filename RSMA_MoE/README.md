# RSMA-MoE 實驗程式

這個專案用 DAG 表示 Task。每個 DAG node 是一個 subtask，每個 subtask 需要選擇 MoE experts、取得 IoT features，並在必要時透過 server-to-server backhaul 補資料。最後系統會檢查 performance loss、deadline、GPU memory、RSMA grouping、feature availability、minimum rate 等限制，並計算 total cost。

執行方式：

```bash
python main.py
```

主要參數集中在 `config.py` 的 `ExperimentConfig`。物理層單位、修復對照與測試方式見 [PHYSICAL_LAYER_FIXES.md](PHYSICAL_LAYER_FIXES.md)。

## 程式結構

| 檔案 / 資料夾 | 用途 |
|---|---|
| `main.py` | 程式入口。 |
| `config.py` | 所有實驗輸入參數。 |
| `experiment_runner.py` | 產生 DAG、建立 network、執行所有方法、輸出結果與圖表。 |
| `generate_dags.py` | 產生 branch-based DAG task graph。 |
| `task_metadata.py` | 產生 node prompt、required features、gating weight、expert confidence、loss metadata。 |
| `rsma_integration.py` | 建立 experts、servers、IoT devices、feature pools、wired graph、channel、distance、phase。 |
| `utils/formulation.py` | 成本、RSMA rate、bandwidth、timing、constraints、performance loss。 |
| `compared_method/` | Baseline 方法。 |
| `compared_method/GSSGD/` | GSSGD DBG grouping 的三個 phase。 |
| `our_alg/` | JRGEP，自家演算法，拆成 Phase 1/2/3。 |
| `network_graph_export.py` | 輸出 server wired network graph。 |
| `sensitivity_analysis.py` | 參數掃描與圖表分析。 |
| `generated_dags/` | DAG 與 network graph artifacts。 |
| `results/` | 實驗 JSON 與 figures。 |
| `SIoT_RSMA/` | SIoT-RSMA 參考原始碼，不是正式執行流程。 |

## 正式比較方法

目前正式比較五種方法：

1. `Hybrid TopK+GSSGD`
2. `Hybrid TopK+location_aware`
3. `WDMoE+GSSGD`
4. `WDMoE+location_aware`
5. `JRGEP`

其中：

- `TopK`：每個 subtask 依 gating score 選 Top-K experts。
- `WDMoE`：先用 Top-K 作 baseline，再依 WLR / theta 邏輯嘗試減少 expert。
- `GSSGD`：使用 DBG grouping，保留原本 coverage redundancy / DBG 思路，DCG 不使用。
- `location_aware`：純粹依 IoT 相對 server 的位置做 K-means grouping。
- `JRGEP`：自家 joint selection / grouping / backhaul 方法。

Baseline 不是 random expert。Baseline 的 expert 仍照 Top-K 或 WDMoE 選；只有當同一個 expert 存在多個 feasible server replica 時，才隨機選 server。

## 主要 Input 參數

### DAG / Task

| 參數 | 目前值 | 說明 |
|---|---:|---|
| `num_runs` | `100` | 獨立平均次數。 |
| `num_dags` | `8` | 每次實驗產生的 task graph 數量。 |
| `num_nodes` | `10` | 每個 DAG 的 node/subtask 數量。 |
| `num_branches` | `(2, 4)` | branch 數量範圍。 |
| `branch_length_range` | `(1, 5)` | 每個 branch 長度範圍。 |
| `edge_probability` | `0.2` | 額外 DAG edge 生成機率。 |
| `allow_early_branch_end` | `True` | 允許分支長短不同。 |

### Expert / Node Metadata

| 參數 | 目前值 | 說明 |
|---|---:|---|
| `num_experts` | `8` | 全域 expert 種類數。 |
| `expert_memory_range` | `(128, 256)` | 每個 expert memory size 範圍。 |
| `expert_inference_times_ms` | `(35, 45, 58, 72, 88, 105, 125, 150)` | 各 expert 的推論時間（毫秒）。 |
| `expert_inference_costs` | `(0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2)` | 各 expert 的推論成本。 |
| `experts_per_server` | `8` | 每台 server 最多存放的 expert 數量。 |
| `server_gpu_memory_range` | `(1280, 1792)` | 每台 server GPU memory limit。 |
| `topk_k` | `5` | Top-K baseline 的 K。 |
| `gating_peak_count_range` | `(2, 4)` | 每個 subtask 有幾個高 gating expert。 |
| `gating_peak_mass_range` | `(0.65, 0.85)` | 高 gating experts 分到的總 gating mass。 |

### IoT / Feature / Network

| 參數 | 目前值 | 說明 |
|---|---:|---|
| `num_edge_servers` | `10` | Edge server 數量。 |
| `num_iot_devices` | `100` | IoT device 數量。 |
| `num_iot_features` | `100` | 全域 feature 數量。 |
| `iot_features_per_node_range` | `(2, 5)` | 每個 subtask 需要的 IoT feature 數量。 |
| `iot_features_per_device_range` | `(4, 8)` | 每個 IoT device 擁有的 feature 數量。 |
| `iot_server_feature_overlap_ratio` | `0.20` | 相鄰 server feature pool 重疊比例。 |
| `iot_global_random_feature_fraction` | `0.2` | IoT feature 中來自全域隨機池的比例。 |
| `area_size` | `2000` | server / IoT 的平面範圍。 |
| `cell_radius` | `350` | server 覆蓋半徑。 |
| `wired_rate_range` | `(5e8, 1.5e9)` | server-to-server wired link rate 範圍。 |
| `wired_extra_link_probability` | `0.05` | 額外 wired link 機率。 |
| `feature_bits_range` | `(8000, 80000)` | 每個 feature 的 bit size 範圍。 |

### RSMA / Channel

| 參數 | 目前值 | 說明 |
|---|---:|---|
| `num_antennas` | `1` | antenna metadata；目前使用 scalar amplitude channel。 |
| `wavelength` | `0.125` | DB phase term 使用的波長。 |
| `noise_power` | `3.9810717e-21` W/Hz | −174 dBm/Hz 的 noise PSD。 |
| `common_power_ratio` | `0.6` | IoT power 中 common stream 比例。 |
| `max_device_power` | `1.2589254e-3` W/Hz | 1 dBm/Hz 的 transmit PSD，非總功率。 |
| `max_group_size` | `4` | 每個 IoT group 最多 device 數。 |
| `min_rate` | `0.4e6` bit/s | C7 minimum aggregate common + private rate。 |

### Performance Loss

| 參數 | 目前值 | 說明 |
|---|---:|---|
| `lambda_reconstruction` | `2.0` | reconstruction loss 權重。 |
| `calibration_alpha` | `0.1` | conformal threshold 的 violation risk。 |
| `reconstruction_sigma` | `1.0` | reconstruction loss sigma。 |
| `calibration_loss_range` | `(1.5, 3.3)` | synthetic calibration losses 範圍。 |
| `loss_threshold` | `None` | 若為 `None`，每個 node 用 calibration losses 算 threshold。 |

### Objective Unit Cost

| 參數 | 目前值 | 說明 |
|---|---:|---|
| `c_act` | `2.0` | 每單位 expert model size activation cost。 |
| `c_bw` | `0.0002` | 每 Hz bandwidth cost。 |
| `c_fwd` | `10.0` | 每個 forwarding / backhaul hop cost。 |

## Simulation Figures

執行 `python sensitivity_analysis.py --runs 100` 會依設定產生四張圖：

1. Number of graph nodes vs Total cost
2. Number of K vs Total cost
3. Number of graph nodes vs Count of hops
4. Number of K vs Time of inference

## 主要公式

### Objective

```text
Total cost = Activation cost + Inference cost + Bandwidth cost + Forwarding cost
```

```text
Activation cost = c_act * sum_{(s,p) activated} M_p
```

```text
Inference cost = sum_{subtask i,j} sum_{selected p} c_inf_p
```

```text
Bandwidth cost = c_bw * sum_{g in groups} B_g
```

```text
Forwarding cost = c_fwd * wired_weight(s,s')
```

Forwarding cost 目前包含：

- DAG predecessor output 跨 server forwarding
- IoT group 透過 backhaul 傳到其他 server

### Performance Loss

每個 node 的 performance loss：

```text
L(P_sel, v) = 1 / P(P_sel | v) + lambda_rec * L_rec
```

其中：

```text
P(P_sel | v) = sum_{p in selected experts} G_p(v) * pi(p | v)
```

`G_p(v)` 是 gating weight，`pi(p|v)` 是 expert confidence。

目前 reconstruction loss 使用 required features 對應的 synthetic reconstruction error；缺失 feature 會貢獻 squared error：

```text
L_rec = sum_{missing feature k} error_k^2 / (2 * sigma^2)
```

若同一個 subtask 選到多個 expert，reconstruction loss 以 selected expert 為單位加權平均：

```text
L_rec_avg = sum_s |P_sel(s)| * L_rec(s) / |P_sel|
```

### Conformal Loss Threshold

若 node 沒有固定 `loss_threshold`，使用 calibration losses 算：

```text
q_hat = sorted(calibration_losses)[ceil((|D_cal| + 1)(1-alpha)) - 1]
```

constraint：

```text
L(P_sel, v_j^i) <= q_hat_j^i
```

### RSMA Bandwidth

每個 group 只計一次 bandwidth：

```text
B_g = max(common_volume / (T_g * common_efficiency),
          max_n private_volume_n / (T_g * private_efficiency_n),
          max_n R_min / (common_efficiency + private_efficiency_n))
```

`T_g` 由依賴該 group 的 subtasks 中最緊的 remaining time 決定。為符合 minimum data-rate constraint，實作也會讓 `B_g` 不低於每個 device 的 common + private aggregate rate 達到 `R_min` 所需的 bandwidth。

common stream 使用 distributed beamforming gain：

```text
R_common = B_g * log2(1 + DB_gain(g) * common_signal / (private_interference + noise))
```

private stream 使用 scalar amplitude channel，signal/interference/noise 均為 W/Hz：

```text
R_private,n = B_g * log2(1 + private_signal_n / (sum_{m != n} private_interference_m + noise))
```

### Timing

同 layer / DAG ready nodes 可平行。每個 subtask 在 server `s` 的 start time：

```text
T_start(s,i,j) = max(feature_ready_time, predecessor_ready_time)
```

同 server 的 predecessor output 不需要 forwarding time；不同 server 需要加 wired transmission time。
多個 selected experts 的輸出會先聚合成固定大小 intermediate output，因此 predecessor forwarding volume 不再乘上 expert 數。

### Assignment / Activation

每個 subtask 必須指派到剛好一台 MEC server；該 subtask 的所有 selected experts 都必須部署在同一台 assigned server 上，且至少選一個 expert。Evaluator 會檢查 C8 single-server assignment 與 C9 expert activation constraints。

## Baseline 流程

### Hybrid TopK+location_aware

1. 每個 subtask 依 gating 選 Top-K experts。
2. 若同 expert 存在多個 server replica，隨機選 feasible server。
3. 依 assigned servers 找 required features。
4. 對每個 server 底下有用 feature 的 IoT 做 K-means location-aware grouping。
5. 先用本地 group 修 performance loss。
6. 若本地不足，再用 backhaul group 修 performance loss。
7. 計算 bandwidth、timing、constraints、cost。

### Hybrid TopK+GSSGD

1. Expert selection 同 TopK baseline。
2. IoT grouping 改用 GSSGD DBG phase 1-3。
3. DCG 不使用。
4. 後續 local repair、backhaul repair、cost evaluation 與 shared pipeline 相同。

### WDMoE+location_aware / WDMoE+GSSGD

1. 對每個 DAG 先跑普通 Top-K，得到 WLR baseline。
2. 從 `theta = wdmoe_initial_threshold` 開始。
3. 每個 theta trial 重新跑整個 DAG。
4. 若 node 的 cosine(weight vector, latency vector) 小於等於 theta，嘗試移除 Top-K 中 gating 最低的 expert。
5. 若移除後即使 full features 仍無法滿足 node loss bound，回復該 node 的 Top-K。
6. 若 graph-level WLR ratio 達到 `wdmoe_wlr_target_ratio`，停止。
7. 搭配 location_aware 或 GSSGD grouping。

## JRGEP 流程

### Phase 1: Server-Expert Selection

- 使用 `r_sim` 將跨 graph 相似 nodes 的 gating vector 做 smoothing。
- 對每個 node 選擇 bounded expert-server set。
- 選擇時考慮：
  - expert 對 performance probability 的貢獻
  - GPU memory feasibility
  - activation cost
  - predecessor forwarding cost
  - server local IoT feature availability 的預估資料成本
- 若某組 candidate 已滿足 loss bound，優先選 incremental objective cost 較低者。

### Phase 2: IoT Grouping / Data Repair / Backhaul

- 依 assigned servers 的 required features 建立候選 groups。
- 先用 local groups 補資料，使 performance loss 下降到 threshold 以下。
- 若 local groups 不足，再從其他 server 的 candidate groups 做 backhaul。
- Backhaul cover 使用最小 cost cover 的 DP，而不是單步 greedy。

### Phase 3: Optional Refinement

- 目前預設不啟用 offline group pruning。
- 這是實驗性 refinement，不屬於目前主結果必要流程。

## 輸出

主要結果：

- `results/hybrid_topk_gssgd_backhaul_result.json`
- `results/hybrid_topk_location_aware_result.json`
- `results/wdmoe_gssgd_result.json`
- `results/wdmoe_location_aware_result.json`
- `results/jrgep_result.json`

主要圖表：

- `results/figure/average_total_cost_comparison.png`
- `results/figure/cost_breakdown_stacked_bar.png`
- `results/figure/current_run_total_cost_comparison.png`
- `results/figure/current_run_cost_breakdown_stacked_bar.png`

## 整理狀態

已移除舊的 standalone RSMA scheduler 路徑。現在正式實驗只比較四個 baseline 與 JRGEP。
