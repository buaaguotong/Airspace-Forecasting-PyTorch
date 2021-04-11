#### Undergraduate Final Design 
Airspace complexity forecasting using GNN-based method


| Step    |   GCRNN    |   DCRNN    |   Graph-WaveNet   |    AGCRN   |   STGCN   |
|---------|:----------:|:----------:|:----------:|:----------:|:----------:|
| step 1  |   0.7898   |   0.7901   |   0.7392   |  0.7979   |   0.7730   |  
| step 2  |   0.7278   |   0.7277   |   0.6959   |  0.7358   |      |     
| step 3  |   0.6926   |   0.6925   |   0.6560   |  0.7150   |      |     
| step 4  |   0.6577   |   0.6577   |   0.6319   |  0.6839   |      |     
| step 5  |   0.6289   |   0.6284   |   0.6117   |  0.6321   |      |     
| step 6  |   0.6157   |   0.6153   |  0.5850    |  0.6114   |      |     
| step 7  |   0.6010   |   0.6019   |   0.5707   |  0.5440   |      |     
| step 8  |   0.5836   |   0.5823   |   0.5510   |  0.5699   |      |     
| step 9  |   0.5764   |   0.5755   |   0.5432   |  0.5855   |      |     
| step 10 |   0.5659   |   0.5664   |   0.5373   |  0.6321   |      |     
| step 11 |   0.5485   |   0.5517   |  0.5312    |  0.5959   |      |     
| step 12 |   0.5398   |   0.5445   |   0.5242   |  0.6269   |      |     

**DCRNN on 126**
```python
Horizon 00: Acc 0.7810, AccH 0.8385, AccN 0.7698, AccL 0.7010
Horizon 01: Acc 0.7183, AccH 0.7865, AccN 0.7112, AccL 0.6091
Horizon 02: Acc 0.6891, AccH 0.7455, AccN 0.6858, AccL 0.5968
Horizon 03: Acc 0.6476, AccH 0.7190, AccN 0.6336, AccL 0.5523
Horizon 04: Acc 0.6136, AccH 0.6927, AccN 0.5907, AccL 0.5260
Horizon 05: Acc 0.6007, AccH 0.6715, AccN 0.5786, AccL 0.5298
Horizon 06: Acc 0.5819, AccH 0.6482, AccN 0.5528, AccL 0.5322
Horizon 07: Acc 0.5630, AccH 0.6370, AccN 0.5317, AccL 0.5072
Horizon 08: Acc 0.5590, AccH 0.6267, AccN 0.5290, AccL 0.5118
Horizon 09: Acc 0.5490, AccH 0.6175, AccN 0.5183, AccL 0.5025
Horizon 10: Acc 0.5324, AccH 0.6059, AccN 0.5011, AccL 0.4806
Horizon 11: Acc 0.5308, AccH 0.5981, AccN 0.5005, AccL 0.4873
==============
Class count: High 14689, Normal 17517, Low 8114, All 40320
Class count: High 14625, Normal 17734, Low 7961, All 40320
Class count: High 14597, Normal 17424, Low 8299, All 40320
Class count: High 14457, Normal 17635, Low 8228, All 40320
Class count: High 14355, Normal 17624, Low 8341, All 40320
Class count: High 14284, Normal 17084, Low 8952, All 40320
Class count: High 14215, Normal 17217, Low 8888, All 40320
Class count: High 14067, Normal 17393, Low 8860, All 40320
Class count: High 14030, Normal 17044, Low 9246, All 40320
Class count: High 13956, Normal 17026, Low 9338, All 40320
Class count: High 13843, Normal 17315, Low 9162, All 40320
Class count: High 13808, Normal 16921, Low 9591, All 40320
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
=====================
Class count: High 8397, Normal 11668, Low 4253, All 24318
Class count: High 8332, Normal 11665, Low 4321, All 24318
Class count: High 8271, Normal 11652, Low 4395, All 24318
Class count: High 8207, Normal 11643, Low 4468, All 24318
Class count: High 8141, Normal 11636, Low 4541, All 24318
Class count: High 8080, Normal 11608, Low 4630, All 24318
Class count: High 8018, Normal 11584, Low 4716, All 24318
Class count: High 7956, Normal 11562, Low 4800, All 24318
Class count: High 7895, Normal 11534, Low 4889, All 24318
Class count: High 7827, Normal 11515, Low 4976, All 24318
Class count: High 7756, Normal 11502, Low 5060, All 24318
Class count: High 7687, Normal 11482, Low 5149, All 24318
```