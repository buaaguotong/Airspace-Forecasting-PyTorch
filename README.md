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


**DCRNN**
```
Horizon 00: Acc 0.7896, AccH 0.8463, AccN 0.7797, AccL 0.7044
Horizon 01: Acc 0.7342, AccH 0.7961, AccN 0.7085, AccL 0.6720
Horizon 02: Acc 0.6992, AccH 0.7565, AccN 0.6746, AccL 0.6448
Horizon 03: Acc 0.6659, AccH 0.7301, AccN 0.6321, AccL 0.6193
Horizon 04: Acc 0.6369, AccH 0.7056, AccN 0.5853, AccL 0.6209
Horizon 05: Acc 0.6239, AccH 0.6844, AccN 0.5577, AccL 0.6487
Horizon 06: Acc 0.6049, AccH 0.6680, AccN 0.5210, AccL 0.6611
Horizon 07: Acc 0.5850, AccH 0.6574, AccN 0.4897, AccL 0.6486
Horizon 08: Acc 0.5787, AccH 0.6470, AccN 0.4762, AccL 0.6581
Horizon 09: Acc 0.5693, AccH 0.6422, AccN 0.4584, AccL 0.6568
Horizon 10: Acc 0.5497, AccH 0.6337, AccN 0.4333, AccL 0.6371
Horizon 11: Acc 0.5497, AccH 0.6268, AccN 0.4251, AccL 0.6500
===========================
Class count: High 24512, Normal 27677, Low 13091, All 65280
Class count: High 24436, Normal 28116, Low 12728, All 65280
Class count: High 24402, Normal 27764, Low 13114, All 65280
Class count: High 24218, Normal 27779, Low 13283, All 65280
Class count: High 23991, Normal 27722, Low 13567, All 65280
Class count: High 23868, Normal 27173, Low 14239, All 65280
Class count: High 23712, Normal 27333, Low 14235, All 65280
Class count: High 23587, Normal 27426, Low 14267, All 65280
Class count: High 23462, Normal 27067, Low 14751, All 65280
Class count: High 23298, Normal 27079, Low 14903, All 65280
Class count: High 23131, Normal 27589, Low 14560, All 65280
Class count: High 23084, Normal 26734, Low 15462, All 65280
```