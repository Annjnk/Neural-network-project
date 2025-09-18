import numpy as np
from mayavi import mlab
import torch
from net import Net
import time

# --- Helper function ---
def safe_normalize(x):
    mx = np.max(x)
    if mx == 0 or np.isnan(mx):
        return np.zeros_like(x)
    return x / mx

# --- Load model and activity data ---
model = Net()
model.load_state_dict(torch.load('mnist.pth'))
model.eval()
activity = np.load('activity.npz')

# --- Figure setup ---
fig = mlab.figure(bgcolor=(13/255, 21/255, 44/255), size=(1920, 1080))

# --- Layer positions ---
in_z, in_x = np.indices((28, 28))
in_x = in_x.ravel().astype(float) - 12
in_z = in_z.ravel().astype(float) - 12
in_y = np.zeros_like(in_x)

def create_layer(shape, offset_y):
    z, x = np.indices(shape)
    x = x.ravel().astype(float) - shape[1] // 2
    z = z.ravel().astype(float) - shape[0] // 2
    y = np.ones_like(x) * offset_y
    x += np.random.rand(len(x))
    z += np.random.rand(len(z))
    y += np.random.rand(len(y)) * 10
    return x, y, z

fc1_x, fc1_y, fc1_z = create_layer((32,32), 10)
fc2_x, fc2_y, fc2_z = create_layer((32,32), 30)
fc3_x, fc3_y, fc3_z = create_layer((32,32), 50)

out_x = np.arange(10).ravel() - 5
out_y = np.ones_like(out_x) + 80
out_z = np.zeros_like(out_x)
out_x = out_x * 3

# --- Connections ---
fc1 = model.fc1.weight.detach().numpy().T
fc2 = model.fc2.weight.detach().numpy().T
fc3 = model.fc3.weight.detach().numpy().T
out = model.fc4.weight.detach().numpy().T

fr_in, to_fc1 = (np.abs(fc1) > 0.1).nonzero()
fr_fc1, to_fc2 = (np.abs(fc2) > 0.05).nonzero()
fr_fc2, to_fc3 = (np.abs(fc3) > 0.05).nonzero()
fr_fc3, to_out = (np.abs(out) > 0.1).nonzero()

to_fc1 += len(in_x)
fr_fc2 += len(in_x) + len(fc1_x)
to_fc2 += len(in_x) + len(fc1_x)
fr_fc3 += len(in_x) + len(fc1_x) + len(fc2_x)
to_fc3 += len(in_x) + len(fc1_x) + len(fc2_x)
to_out += len(in_x) + len(fc1_x) + len(fc2_x) + len(fc3_x)

# --- Create points ---
x = np.hstack((in_x, fc1_x, fc2_x, fc3_x, out_x))
y = np.hstack((in_y, fc1_y, fc2_y, fc3_y, out_y))
z = np.hstack((in_z, fc1_z, fc2_z, fc3_z, out_z))

# Initial activations
act_input = activity['input'][0]
act_fc1 = activity['fc1'][0]
act_fc2 = activity['fc2'][0]
act_fc3 = activity['fc3'][0]
act_out = activity['fc4'][0]

s = np.hstack((
    safe_normalize(act_input.ravel()),
    safe_normalize(act_fc1),
    safe_normalize(act_fc2),
    safe_normalize(act_fc3),
    safe_normalize(act_out),
))

# --- Layer visualization ---
acts = mlab.points3d(x, y, -z, s, mode='cube', scale_factor=0.5, scale_mode='none', colormap='gray')

src = mlab.pipeline.scalar_scatter(x, y, -z, s)
src.mlab_source.dataset.lines = np.vstack((
    np.hstack((fr_in, fr_fc1, fr_fc2, fr_fc3)),
    np.hstack((to_fc1, to_fc2, to_fc3, to_out))
)).T
src.update()
lines = mlab.pipeline.stripper(src)
connections = mlab.pipeline.surface(lines, colormap='gray', line_width=1, opacity=0.2)

mlab.view(azimuth=0, elevation=80, distance=100, focalpoint=[0,35,0], reset_roll=False)

# --- Open the window on macOS without blocking ---
mlab.show(stop=False)

# --- Animation loop ---
n_frames = 1600
for frame in range(n_frames):
    if frame % 16 == 0:
        i = frame // 16
        act_input = activity['input'][i]
        act_fc1 = activity['fc1'][i]
        act_fc2 = activity['fc2'][i]
        act_fc3 = activity['fc3'][i]
        act_out = activity['fc4'][i]

        s = np.hstack((
            safe_normalize(act_input.ravel()),
            safe_normalize(act_fc1),
            safe_normalize(act_fc2),
            safe_normalize(act_fc3),
            safe_normalize(act_out),
        ))

        acts.mlab_source.scalars = s
        connections.mlab_source.scalars = s

        # Print progress
        if frame % 100 == 0:
            print(f"Frame {frame} / {n_frames}")

    # Rotate view
    mlab.view(azimuth=(frame*2)%360, elevation=80, distance=120, focalpoint=[0,35,0])

    # Update GUI
    mlab.process_ui_events()

    # Small delay so animation is visible
    time.sleep(0.03)

# --- Optional: close window after finishing ---
# mlab.close()
print("Animation finished!")
