import os
os.environ['ETS_TOOLKIT'] = 'qt5'
os.environ['QT_API'] = 'pyqt5'

import numpy as np
import pyvista as pv
import torch
from tqdm import tqdm

from net import Net

# Load model and activity
model = Net()
model.load_state_dict(torch.load('mnist.pth', weights_only=False))
model.eval()
activity = np.load('activity.npz')

# Create plotter
plotter = pv.Plotter(window_size=(1920, 1080), off_screen=False)
plotter.set_background([13 / 255, 21 / 255, 44 / 255])

# Layer units (cast all to float)
in_z, in_x = np.indices((28, 28))
fc1_z, fc1_x = np.indices((32, 32))
fc2_z, fc2_x = np.indices((32, 32))
fc3_z, fc3_x = np.indices((32, 32))

in_x = in_x.ravel().astype(float) - 12
in_z = in_z.ravel().astype(float) - 12
in_y = np.zeros_like(in_x, dtype=float)

fc1_x = fc1_x.ravel().astype(float) - 16
fc1_z = fc1_z.ravel().astype(float) - 16
fc1_y = np.ones_like(fc1_x, dtype=float) + 10

fc2_x = fc2_x.ravel().astype(float) - 16
fc2_z = fc2_z.ravel().astype(float) - 16
fc2_y = np.ones_like(fc2_x, dtype=float) + 30

fc3_x = fc3_x.ravel().astype(float) - 16
fc3_z = fc3_z.ravel().astype(float) - 16
fc3_y = np.ones_like(fc3_x, dtype=float) + 50

out_x = np.arange(10, dtype=float) - 5
out_y = np.ones_like(out_x, dtype=float) + 80
out_z = np.zeros_like(out_x, dtype=float)

# Add randomness
fc1_x += np.random.rand(len(fc1_x)) * 1
fc1_z += np.random.rand(len(fc1_z)) * 1
fc1_y += np.random.rand(len(fc1_y)) * 10

fc2_x += np.random.rand(len(fc2_x)) * 1
fc2_z += np.random.rand(len(fc2_z)) * 1
fc2_y += np.random.rand(len(fc2_y)) * 10

fc3_x += np.random.rand(len(fc3_x)) * 1
fc3_z += np.random.rand(len(fc3_z)) * 1
fc3_y += np.random.rand(len(fc3_y)) * 10

out_x *= 3

# Connections between layers
fc1 = model.fc1.weight.detach().numpy().T
fc2 = model.fc2.weight.detach().numpy().T
fc3 = model.fc3.weight.detach().numpy().T
out = model.fc4.weight.detach().numpy().T

fr_in, to_fc1 = (np.abs(fc1) > 0.1).nonzero()
fr_fc1, to_fc2 = (np.abs(fc2) > 0.05).nonzero()
fr_fc2, to_fc3 = (np.abs(fc3) > 0.05).nonzero()
fr_fc3, to_out = (np.abs(out) > 0.1).nonzero()

fr_fc1 += len(in_x)
to_fc1 += len(in_x)
fr_fc2 += len(in_x) + len(fc1_x)
to_fc2 += len(in_x) + len(fc1_x)
fr_fc3 += len(in_x) + len(fc1_x) + len(fc2_x)
to_fc3 += len(in_x) + len(fc1_x) + len(fc2_x)
to_out += len(in_x) + len(fc1_x) + len(fc2_x) + len(fc3_x)

# Points
x = np.hstack((in_x, fc1_x, fc2_x, fc3_x, out_x))
y = np.hstack((in_y, fc1_y, fc2_y, fc3_y, out_y))
z = np.hstack((in_z, fc1_z, fc2_z, fc3_z, out_z))

act_input = activity['input'][0]
act_fc1 = activity['fc1'][0]
act_fc2 = activity['fc2'][0]
act_fc3 = activity['fc3'][0]
act_out = activity['fc4'][0]

s = np.hstack((
    act_input.ravel() / act_input.max(),
    act_fc1 / act_fc1.max(),
    act_fc2 / act_fc2.max(),
    act_fc3 / act_fc3.max(),
    act_out / (act_out.max() if act_out.max() != 0 else 1),
))

# Plot points
points = pv.PolyData(np.c_[x, y, -z])
points['scalars'] = s
plotter.add_points(points, scalars='scalars', render_points_as_spheres=True, point_size=5, cmap='gray')

# Plot connections
lines = []
for fr, to in zip(np.hstack((fr_in, fr_fc1, fr_fc2, fr_fc3)), np.hstack((to_fc1, to_fc2, to_fc3, to_out))):
    start = np.array([x[fr], y[fr], -z[fr]])
    end = np.array([x[to], y[to], -z[to]])
    lines.append(pv.Line(start, end))

for line in lines:
    plotter.add_mesh(line, color='gray', line_width=1, opacity=0.2)

# Labels
for i, label in enumerate(range(10)):
    plotter.add_point_labels([[out_x[i], out_y[i], out_z[i]]], [str(label)], font_size=16, point_color=[1,1,1], text_color=[1,1,1])

plotter.view_3d(azimuth=0, elevation=80, distance=120, focal_point=[0,35,0])

# Animation
def update(frame):
    i = frame % len(activity['input'])
    act_input = activity['input'][i]
    act_fc1 = activity['fc1'][i]
    act_fc2 = activity['fc2'][i]
    act_fc3 = activity['fc3'][i]
    act_out = activity['fc4'][i]
    s = np.hstack((
        act_input.ravel() / act_input.max(),
        act_fc1 / act_fc1.max(),
        act_fc2 / act_fc2.max(),
        act_fc3 / act_fc3.max(),
        act_out / (act_out.max() if act_out.max() != 0 else 1),
    ))
    points['scalars'] = s
    plotter.update_coordinates(points.points, render=False)
    plotter.render()

# Run animation
for frame in tqdm(range(1600)):
    update(frame)
    plotter.screenshot(f'frames/frame{frame:04d}.png')

plotter.show()
