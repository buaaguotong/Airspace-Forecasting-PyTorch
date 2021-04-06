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
```python
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

**Graph-Wavenet**
| Cannot model the Low complexity
```python
Horizon 01, Acc: 0.7175, AccH: 0.8364, AccN: 0.8702, AccL: 0.0000
Horizon 02, Acc: 0.6763, AccH: 0.7830, AccN: 0.8293, AccL: 0.0000
Horizon 03, Acc: 0.6382, AccH: 0.7829, AccN: 0.7526, AccL: 0.0000
Horizon 04, Acc: 0.6151, AccH: 0.7326, AccN: 0.7472, AccL: 0.0000
Horizon 05, Acc: 0.5964, AccH: 0.7085, AccN: 0.7305, AccL: 0.0000
Horizon 06, Acc: 0.5813, AccH: 0.6893, AccN: 0.7181, AccL: 0.0000
Horizon 07, Acc: 0.5501, AccH: 0.7705, AccN: 0.5926, AccL: 0.0000
Horizon 08, Acc: 0.5389, AccH: 0.7652, AccN: 0.5776, AccL: 0.0000
Horizon 09, Acc: 0.5303, AccH: 0.7590, AccN: 0.5687, AccL: 0.0000
Horizon 10, Acc: 0.5242, AccH: 0.7555, AccN: 0.5633, AccL: 0.0000
Horizon 11, Acc: 0.5189, AccH: 0.7505, AccN: 0.5601, AccL: 0.0000
Horizon 12, Acc: 0.5117, AccH: 0.7462, AccN: 0.5528, AccL: 0.0000
Average acc: 0.5832
===========================
Class count: High 16145, Normal 20308, Low 6999, All 43452
Class count: High 16051, Normal 20281, Low 7120, All 43452
Class count: High 15947, Normal 20258, Low 7247, All 43452
Class count: High 15849, Normal 20231, Low 7372, All 43452
Class count: High 15751, Normal 20196, Low 7505, All 43452
Class count: High 15661, Normal 20140, Low 7651, All 43452
Class count: High 15554, Normal 20108, Low 7790, All 43452
Class count: High 15447, Normal 20077, Low 7928, All 43452
Class count: High 15349, Normal 20029, Low 8074, All 43452
Class count: High 15244, Normal 19988, Low 8220, All 43452
Class count: High 15148, Normal 19956, Low 8348, All 43452
Class count: High 15049, Normal 19906, Low 8497, All 43452
```

**AGCRN**
| Performance drops sharply on Low complexity categories
```python
Horizon 01, Acc: 0.7758, AccH: 0.8321, AccN: 0.7794, AccL: 0.6483
Horizon 02, Acc: 0.7183, AccH: 0.7791, AccN: 0.7250, AccL: 0.5768
Horizon 03, Acc: 0.6664, AccH: 0.7485, AccN: 0.7153, AccL: 0.3740
Horizon 04, Acc: 0.5753, AccH: 0.7263, AccN: 0.6501, AccL: 0.0887
Horizon 05, Acc: 0.5180, AccH: 0.7147, AccN: 0.5515, AccL: 0.0627
Horizon 06, Acc: 0.4956, AccH: 0.7173, AccN: 0.5090, AccL: 0.0567
Horizon 07, Acc: 0.4866, AccH: 0.7116, AccN: 0.4986, AccL: 0.0567
Horizon 08, Acc: 0.4814, AccH: 0.7122, AccN: 0.4922, AccL: 0.0551
Horizon 09, Acc: 0.4770, AccH: 0.7108, AccN: 0.4887, AccL: 0.0549
Horizon 10, Acc: 0.4725, AccH: 0.7130, AccN: 0.4825, AccL: 0.0545
Horizon 11, Acc: 0.4685, AccH: 0.7121, AccN: 0.4799, AccL: 0.0527
Horizon 12, Acc: 0.4645, AccH: 0.7154, AccN: 0.4737, AccL: 0.0538
=======================
Class count: High 14050, Normal 18581, Low 6741, All 39372
Class count: High 13938, Normal 18575, Low 6859, All 39372
Class count: High 13831, Normal 18559, Low 6982, All 39372
Class count: High 13728, Normal 18534, Low 7110, All 39372
Class count: High 13613, Normal 18517, Low 7242, All 39372
Class count: High 13511, Normal 18471, Low 7390, All 39372
Class count: High 13406, Normal 18433, Low 7533, All 39372
Class count: High 13305, Normal 18393, Low 7674, All 39372
Class count: High 13198, Normal 18354, Low 7820, All 39372
Class count: High 13084, Normal 18324, Low 7964, All 39372
Class count: High 12961, Normal 18311, Low 8100, All 39372
Class count: High 12846, Normal 18274, Low 8252, All 39372
```

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