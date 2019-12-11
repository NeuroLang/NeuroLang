**Program code**

Cytoarchitecture probabilistic maps are defined based on the SMP Anatomy
toolbox
```python
PrimaryVisual(v) :- PMapSTM(l, v), PrimaryVisualLabel(l)
PrimaryAuditory(v) :- PMapSTM(l, v), PrimaryAuditoryLabel(l)
PrimaryMotor(v) :- PMapSTM(l, v), PrimaryMotorLabel(l)
PrimarySomatosensory(v) :- PMapSTM(l, v), PrimarySomatosensoryLabel(l)
```
