#### Undergraduate Final Design (toy-project)
Airspace complexity forecasting using GNN-based method

Input data shape:
```python
x shape: (1417, 12, 126, 17), y shape: (1417, 12, 126, 1)
---------------------------------------------------------
train x: (850, 12, 126, 17),  y: (850, 12, 126, 1)
val   x: (284, 12, 126, 17),  y: (284, 12, 126, 1)
test  x: (283, 12, 126, 17),  y: (283, 12, 126, 1)
```
 Using **One-hot encoding** & **CE loss** does not seem to be a good idea, cannot explain
