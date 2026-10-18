Scanning base directory: result/release/sharegpt
  C1: found 3 completed rounds
  C4: found 3 completed rounds
  C16: found 3 completed rounds
  C32: found 3 completed rounds
  C64: found 3 completed rounds

### ShareGPT 基准性能对比表

| 并发数 | TPS (基线) | TPS (实测) | TTFT (基线) | TTFT (实测 mean/median/p99) | TPOT (基线) | TPOT (实测 mean/median/p99) | 成功率 |
|---|---|---|---|---|---|---|---|
| 1 | 21.454 | 37.797 | 200.866 ms | 251.86 / 236.85 / 432.35 ms | 45.727 ms | 24.89 / 25.22 / 25.70 ms | 100.0% |
| 4 | 106.207 | 130.989 | 189.951 ms | 253.97 / 238.07 / 456.21 ms | 35.851 ms | 28.97 / 28.00 / 53.22 ms | 100.0% |
| 16 | 307.450 | 333.206 | 189.460 ms | 271.38 / 242.45 / 922.82 ms | 49.725 ms | 45.36 / 45.09 / 95.83 ms | 100.0% |
| 32 | 435.996 | 406.677 | 199.106 ms | 303.34 / 251.57 / 1265.32 ms | 69.259 ms | 76.32 / 75.41 / 161.64 ms | 100.0% |
| 64 | 528.158 | 550.324 | 215.717 ms | 402.04 / 260.53 / 3355.44 ms | 103.927 ms | 121.63 / 111.21 / 358.93 ms | 100.0% |
