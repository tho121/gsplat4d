Based off this work: https://github.com/fudan-zvg/4d-gaussian-splatting

Follow the instructions outlined in that repo to install the environment and get the data

To train run:

```python train.py --config configs/dynerf/coffee_martini.yaml```

To use the interactive viewer on a trained model:

```python viewer.py --config configs/dynerf/coffee_martini.yaml --start_checkpoint gsplat4d/output/N3V/coffee_martini/chkpnt7000.pth --transforms_path gsplat4d/data/N3V/coffee_martini/transforms_train.json```

Controls for the interactive viewer:

W: Forward
S: Backward
A: Left
D: Right

Q: Speed up time
E: Slow down time
R: Reverse time
F: Pause time

Z: Move down
C: Move up

X: Reset rotation
Up Arrow: Tilt up
Down arrow: Tilt down
Left arrow: Tilt left
Right arrow: Tilt right
