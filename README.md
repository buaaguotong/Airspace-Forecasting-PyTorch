#### Undergraduate Final Design 
Airspace complexity forecasting using GNN-based method


**DCRNN on 126**
```python
Horizon 01: MAE 0.2796, RMSE 0.3587, MAPE 20.9172%
Horizon 02: MAE 0.3023, RMSE 0.3914, MAPE 23.1530%
Horizon 03: MAE 0.3361, RMSE 0.4149, MAPE 25.6105%
Horizon 04: MAE 0.3640, RMSE 0.4302, MAPE 27.6814%
Horizon 05: MAE 0.3833, RMSE 0.4426, MAPE 29.2766%
Horizon 06: MAE 0.3971, RMSE 0.4517, MAPE 30.2993%
Horizon 07: MAE 0.4078, RMSE 0.4585, MAPE 31.2024%
Horizon 08: MAE 0.4158, RMSE 0.4629, MAPE 31.9410%
Horizon 09: MAE 0.4229, RMSE 0.4672, MAPE 32.5731%
Horizon 10: MAE 0.4284, RMSE 0.4701, MAPE 33.0874%
Horizon 11: MAE 0.4333, RMSE 0.4727, MAPE 33.5817%
Horizon 12: MAE 0.4381, RMSE 0.4753, MAPE 34.0323%
```

**AGCRN on 126**
```python
Horizon 01, Acc: 0.7725, AccH: 0.8255, AccN: 0.7794, AccL: 0.6490
Horizon 02, Acc: 0.7179, AccH: 0.7705, AccN: 0.7290, AccL: 0.5862
Horizon 03, Acc: 0.6764, AccH: 0.7292, AccN: 0.6940, AccL: 0.5306
Horizon 04, Acc: 0.6392, AccH: 0.7195, AccN: 0.6554, AccL: 0.4496
Horizon 05, Acc: 0.5874, AccH: 0.7398, AccN: 0.5704, AccL: 0.3579
Horizon 06, Acc: 0.5479, AccH: 0.7916, AccN: 0.4717, AccL: 0.3136
Horizon 07, Acc: 0.5208, AccH: 0.8047, AccN: 0.4257, AccL: 0.2721
Horizon 08, Acc: 0.5133, AccH: 0.8051, AccN: 0.4104, AccL: 0.2777
Horizon 09, Acc: 0.5065, AccH: 0.8091, AccN: 0.3999, AccL: 0.2694
Horizon 10, Acc: 0.5009, AccH: 0.8086, AccN: 0.3923, AccL: 0.2683
Horizon 11, Acc: 0.4929, AccH: 0.8125, AccN: 0.3795, AccL: 0.2609
Horizon 12, Acc: 0.4818, AccH: 0.8053, AccN: 0.3715, AccL: 0.2451
```

**Graph Wavenet on 126**
```python
Horizon 01, MAE: 0.1477, RMSE: 0.3739, MAPE: 10.9162%
Horizon 02, MAE: 0.1946, RMSE: 0.4289, MAPE: 14.4367%
Horizon 03, MAE: 0.2373, RMSE: 0.4619, MAPE: 18.3935%
Horizon 04, MAE: 0.2667, RMSE: 0.4912, MAPE: 20.2399%
Horizon 05, MAE: 0.2834, RMSE: 0.5036, MAPE: 21.5618%
Horizon 06, MAE: 0.3400, RMSE: 0.5630, MAPE: 29.2662%
Horizon 07, MAE: 0.3509, RMSE: 0.5764, MAPE: 30.2875%
Horizon 08, MAE: 0.3559, RMSE: 0.5812, MAPE: 30.7022%
Horizon 09, MAE: 0.3621, RMSE: 0.5856, MAPE: 31.2768%
Horizon 10, MAE: 0.3681, RMSE: 0.5913, MAPE: 31.8268%
Horizon 11, MAE: 0.3709, RMSE: 0.5932, MAPE: 32.0680%
Horizon 12, MAE: 0.3736, RMSE: 0.5939, MAPE: 32.3405%
Average mae: 0.3043
```