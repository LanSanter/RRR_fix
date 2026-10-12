# 物理層修復與單位約定

本次依照提供的 `message.txt` 修復 physical-layer evaluator；正式入口為 `RSMA_MoE/main.py`，`SIoT_RSMA/` 為參考快照，`legacy/utils/SIoT_RSMA/` 為舊實作。正式方法共用同一個 formulation evaluator。

| 問題 | 核對結果與修復 |
|---|---|
| Noise PSD 換算 | 主 `ExperimentConfig` 本來已正確；其他入口與兩份 legacy config 統一為 `10^((-174-30)/10) = 3.9810717e-21` W/Hz。 |
| SINR 硬編碼 noise | 舊 `phase2`、`phase3`、`testing_cost` 全部使用修正後的 `AWGNoise`。 |
| Path loss / shadowing 被平方兩次 | `channel_gain` 明確表示 amplitude；主環境與四份舊 channel 實作改為 `abs(g) / sqrt(L*S)`。 |
| 小尺度衰落平均功率為 2 | 改成 `(X+jY)/sqrt(2)`，其中 X,Y 為獨立 N(0,1)，使 E[abs(g)^2]=1。 |
| Linear SINR 送入 dB MCS thresholds | 移除沒有校準單位的舊 MCS table。`rate_mapping(linear_sinr)` 現在直接回傳 Shannon spectral efficiency。不再使用 dB thresholds。 |
| Shannon 與 MCS 混用 | 主 evaluator 原本已是 Shannon；現在 legacy 也使用 `log1p(SINR)/log(2)`，單位 bit/s/Hz。 |
| rate / bandwidth 單位 | `R = B*eta` 為 bit/s，`B = D_bits/(T_seconds*eta)` 為 Hz。舊實作的 location count 先乘 `config['feature_bits']` 轉成 bits。 |
| Tx power 不透明 | `max_device_power`、`DeviceSpec.max_power`、group power overrides、legacy `SIoT.power` 均為 W/Hz。保留既有欄位名稱；結果 JSON 新增單位註記。 |
| 0.6/0.4 未使用 | 主程式原本已使用 `common_power_ratio`。Legacy 現在使用 `cmn_ratio` / `pvt_ratio` 並檢查總和為 1；common data overlap ratio 只決定 payload。 |
| Bandwidth XOR | `1*10^6` 改成 `1*10**6`，即 1,000,000 Hz。 |
| 零速率 / 空 payload | 無 payload 的 stream 需求為 0；有 payload 但效率為 0 或 time budget 非正時，明確視為不可行，不用 epsilon 假造有限頻寬。 |
| 距離 300 m 上限 | 主環境覆蓋半徑可為 350 m；移除 channel 函數的 300 m 截斷，使用實際 link distance，保留 10 m 參考距離下限。 |
| Wavelength / beamforming | 保留 0.125 m 與提供清單中的 Gamma 定義。 |

## 採用的模型

`PL(d) = 50 + 15*log10(d)` dB，shadowing 維持原本 1 dB 標準差。令 `L*S = 10^((PL+shadow_dB)/10)`，因此 `abs(h)^2 = abs(g)^2/(L*S)`；例如 PL=80 dB、shadow=0、abs(g)^2=1，received power attenuation 為 1e-8，而非 1e-16。

所有 signal、interference、noise 都使用 PSD。若將三者同乘 B Hz，SINR 中的 B 消去，所以反推 bandwidth 不需要在 noise 上另乘一次 B。這是固定 transmit PSD 模型；若改成固定總功率 W，SINR 會隨 B 改變，不能沿用此直接反推公式。

採用清單給定的 1 dBm/Hz，不擅自改成 1 dBm 總功率。此設定的總功率為 `PSD*B`，增加 B 也會增加總功率；本次沒有另加總功率上限。

Legacy 的 `feature_bits` 預設 16 KiB × 8 = 131,072 bits，與主 `ExperimentConfig.default_feature_bits` 對齊，這是明確的模擬假設，可自行修改，並非從 location count 推導出的量。`phase3` 的 `common_message_data_size` 回傳值現在為 bits。Legacy singleton 保留全功率 uplink，multi-device 使用配置的 RSMA 功率切分。

保留現有 concurrent-stream bandwidth 模型與 interference / Gamma 定義，未加入新的 beta 私有頻寬配置或 SIC 順序最佳化。正式主程式的功率分配、排程和 grouping 策略沿用原設計。

## 驗證

在 repo 根目錄執行：

```bash
python -m pip install -r RSMA_MoE/requirements.txt
MPLBACKEND=Agg python -m unittest discover -s tests -v
python -m compileall -q RSMA_MoE legacy/utils/SIoT_RSMA tests
```

測試涵蓋 PSD 數值、入口一致性、path loss/shadowing 單次 power attenuation、unit-power fading 的固定 seed 統計、Shannon 手算、bandwidth 與 deadline、功率切分、zero-rate edge cases、兩份 legacy config 與四份 channel。

另用固定 seed、2 個 5-node DAG、3 servers、12 IoTs 執行五種正式方法，要求所有成本與頻寬有限、至少一個 uplink group、無 constraint violations，並驗證輸出 JSON 的 PSD 與單位。

**修正會改變 channel、SINR、bandwidth 及成本。舊結果必須重新生成，不能把前後數字當成同一模型下的演算法改進。測試驗證數值與執行一致性，不代表實測無線校準或演算法最優性證明。**
