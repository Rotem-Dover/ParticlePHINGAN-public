"""
Renders a 3D volumetric visualization of GEANT4 proton-track energy deposition
in aluminum using software ray-casting through a voxelized histogram.
"""

import numpy as np
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt
import matplotlib as mpl
from common.paths import BENCHMARKS_OUTPUTS, PATH_TO_GEANT4_DATA
from common.config import apply_project_plotting_style
from data_handling.reader import read_geant4_simulation_output

apply_project_plotting_style()

# ==========================================
# 1. LOAD & PREP DATA
# ==========================================
print("Loading data...")

df = read_geant4_simulation_output(PATH_TO_GEANT4_DATA, verbose=True)

# Define bins
bins = (1000, 250, 250)

# Compute weighted 3D histogram
H, edges = np.histogramdd(
    df[['Y', 'Z', 'X']].values,
    bins=bins,
    weights=df['TotalEnergyLoss']
)

# Prepare grid for voxel edges (note: edges, not centers)
x_edges, y_edges, z_edges = edges

H_reoriented = H / 1e6  #np.transpose(H / 1e6, axes=(1, 2, 0))

x_edges_mm = x_edges * 1000.0
y_edges_mm = y_edges * 1000.0
z_edges_mm = (z_edges - z_edges.min()) * 1000.0

scale_xy = 25.0
data_x_min, data_x_max = x_edges_mm.min() * scale_xy, x_edges_mm.max() * scale_xy
data_y_min, data_y_max = y_edges_mm.min() * scale_xy, y_edges_mm.max() * scale_xy
z_min, z_max = z_edges_mm.min(), z_edges_mm.max()

box_x_min = -0.65 * scale_xy
box_x_max = 0.65 * scale_xy
box_y_min = -0.65 * scale_xy
box_y_max = 0.65 * scale_xy

print("Normalizing volume...")
log_H = np.full_like(H_reoriented, -100.0)
valid = H_reoriented > 0
log_H[valid] = np.log10(H_reoriented[valid])

vmin_log, vmax_log = -0.5, 2.0
vol = (log_H - vmin_log) / (vmax_log - vmin_log)

# ==========================================
# 2. EXACT COLORMAP & CAMERA (RGBA)
# ==========================================
cmap = plt.get_cmap('jet')

lut = np.zeros((256, 4))
lut[1:, :3] = cmap(np.linspace(0, 1, 255))[:, :3]
lut[1:, 3] = 1.0
lut[0] = [1.0, 1.0, 1.0, 0.0]

img_w, img_h = 1200, 960
fov = np.radians(30)

# CAMERA TILT: Negated the X coordinate (-45.0) to mirror the view around the vertical axis
eye = np.array([-50.0, 20.0, -30.0])
target = np.array([0.0, 0.0, 20.0])
up_vec = np.array([0.0, 1.0, 0.0])

fwd = target - eye
fwd /= np.linalg.norm(fwd)
right = np.cross(fwd, up_vec)
right /= np.linalg.norm(right)
up = np.cross(right, fwd)

u = np.linspace(-1, 1, img_w) * np.tan(fov / 2) * (img_w / img_h)
v = np.linspace(1, -1, img_h) * np.tan(fov / 2)
uu, vv = np.meshgrid(u, v)

dirs = fwd + uu[..., None] * right + vv[..., None] * up
dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
rays_flat = dirs.reshape(-1, 3)

# ==========================================
# 3. VECTORIZED RAY-MARCHING (Raw Binning)
# ==========================================
steps = 800
t_min = 20.0
t_max = 115.0
t_vals = np.linspace(t_min, t_max, steps)

eye_flat = eye.reshape(1, 3)
vol_shape = np.array(vol.shape)

max_density = np.full(img_h * img_w, -100.0)

print("Marching rays (Nearest Neighbor)...")
for dist in t_vals:
    pos = eye_flat + rays_flat * dist

    ix = (pos[:, 0] - data_x_min) / (data_x_max - data_x_min) * (vol_shape[0] - 1)
    iy = (pos[:, 1] - data_y_min) / (data_y_max - data_y_min) * (vol_shape[1] - 1)
    iz = (pos[:, 2] - z_min) / (z_max - z_min) * (vol_shape[2] - 1)
    coords = np.stack([ix, iy, iz])

    density = map_coordinates(vol, coords, order=0, cval=-100.0, prefilter=False)
    max_density = np.maximum(max_density, density)


# ==========================================
# 4. 3D TO 2D PROJECTION & ANNOTATION
# ==========================================
def project_3d_to_2d(p3d):
    v_dir = p3d - eye
    z_cam = np.dot(v_dir, fwd)
    x_cam = np.dot(v_dir, right)
    y_cam = np.dot(v_dir, up)

    u_proj = x_cam / z_cam
    v_proj = y_cam / z_cam

    tan_fov = np.tan(fov / 2)
    aspect = img_w / img_h

    px = (u_proj / (tan_fov * aspect) + 1.0) / 2.0 * img_w
    py = (-v_proj / tan_fov + 1.0) / 2.0 * img_h
    return px, py


def plot_3d_line(ax, p1, p2, color='black', lw=1.0, alpha=0.5):
    px1, py1 = project_3d_to_2d(p1)
    px2, py2 = project_3d_to_2d(p2)
    ax.plot([px1, px2], [py1, py2], color=color, linewidth=lw, alpha=alpha)


def add_3d_text(ax, p3d, text, color='black', fontsize=10, offset=(0, 0)):
    px, py = project_3d_to_2d(p3d)
    ax.text(px + offset[0], py + offset[1], text, color=color, fontsize=fontsize,
            ha='center', va='center', alpha=0.8)


# ==========================================
# 5. FINAL IMAGE ASSEMBLY & CROPPING
# ==========================================
print("Applying exact colors and drawing annotated wireframe...")

color_indices = np.zeros(img_h * img_w, dtype=int)
valid_mask = max_density > -10.0
color_indices[valid_mask] = (np.clip(max_density[valid_mask], 0.0, 1.0) * 254 + 1).astype(int)

color_acc = lut[color_indices]
img_data = color_acc.reshape(img_h, img_w, 4)

fig, ax = plt.subplots(figsize=(14, 10), facecolor='none')
ax.imshow(img_data)
ax.axis('off')

corners = [
    [box_x_min, box_y_min, z_min], [box_x_max, box_y_min, z_min], [box_x_max, box_y_max, z_min],
    [box_x_min, box_y_max, z_min],
    [box_x_min, box_y_min, z_max], [box_x_max, box_y_min, z_max], [box_x_max, box_y_max, z_max],
    [box_x_min, box_y_max, z_max]
]
edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]

for c1, c2 in edges:
    plot_3d_line(ax, np.array(corners[c1]), np.array(corners[c2]), color='black', lw=3.0, alpha=0.6)

displacement_x = 4.5
displacement_y = 4.5
displacement_z = 2

# Z-Axis (Moved to box_x_min so it stays on the right visually)
z_ticks_mm = np.linspace(z_edges_mm.min(), z_edges_mm.max(), 5)
for z_val in z_ticks_mm:
    p_tick = np.array([box_x_min, box_y_min, z_val])
    # Project the tick inwards
    p_tick_end = np.array([box_x_min + 1.0, box_y_min + 1.0, z_val])
    plot_3d_line(ax, p_tick, p_tick_end, color='black', lw=1.0, alpha=0.7)
    add_3d_text(ax, p_tick, f"{z_val:.1f}", color='black', fontsize=24, offset=(75, 35))
add_3d_text(ax, np.array([box_x_min - displacement_z, box_y_min - displacement_z, (z_min + z_max) / 2]), "Z [mm]", color='black', fontsize=32,
            offset=(130, 65))

# X-Axis (Stays on the front-bottom edge)
x_ticks_mm = np.array([-0.5, -0.25, 0.0, 0.25, 0.5])
for x_val in x_ticks_mm:
    p_tick = np.array([x_val * scale_xy, box_y_min, z_min])
    # Project the tick inwards
    p_tick_end = np.array([x_val * scale_xy, box_y_min + 1.0, z_min + 1.0])
    plot_3d_line(ax, p_tick, p_tick_end, color='black', lw=1.0, alpha=0.7)
    add_3d_text(ax, p_tick, f"{x_val:.2f}", color='black', fontsize=24, offset=(-35, 55))
add_3d_text(ax, np.array([(box_x_min + box_x_max) / 2, box_y_min, z_min - displacement_x]), "X [mm]", color='black',
            fontsize=32, offset=(-50, 100))

# Y-Axis (Moved to box_x_max so it stays on the left visually)
y_ticks_mm = np.array([-0.5, -0.25, 0.0, 0.25, 0.5])
for y_val in y_ticks_mm:
    p_tick = np.array([box_x_max, y_val * scale_xy, z_min])
    # Project the tick inwards
    p_tick_end = np.array([box_x_max - 1.0, y_val * scale_xy, z_min + 1.0])
    plot_3d_line(ax, p_tick, p_tick_end, color='black', lw=1.0, alpha=0.7)
    add_3d_text(ax, p_tick, f"{y_val:.2f}", color='black', fontsize=24, offset=(-90, -25))
add_3d_text(ax, np.array([box_x_max + displacement_y, (box_y_min + box_y_max) / 2, z_min - displacement_y]), "Y [mm]", color='black',
            fontsize=32, offset=(-145, -60))

# CANVAS CROPPING
proj_corners = [project_3d_to_2d(np.array(c)) for c in corners]
px_vals = [p[0] for p in proj_corners]
py_vals = [p[1] for p in proj_corners]

pad_x = 150
pad_y = 110

ax.set_xlim(min(px_vals) - pad_x, max(px_vals) + pad_x)
ax.set_ylim(max(py_vals) + pad_y, min(py_vals) - pad_y)

# COLORBAR
norm = mpl.colors.LogNorm(vmin=10 ** vmin_log, vmax=10 ** vmax_log)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label(r'Cumulative $\Delta E^{\rm tot}$ [MeV]', color='black', size=32, labelpad=30)
cbar.ax.yaxis.set_tick_params(color='black', labelcolor='black', labelsize=24)
cbar.outline.set_edgecolor('black')

save_kwargs = {'facecolor': 'none', 'bbox_inches': 'tight', 'pad_inches': 0.0, 'dpi': 300, 'transparent': True}

plt.savefig(BENCHMARKS_OUTPUTS / "geant4_tracks.pdf", **save_kwargs)

plt.show()
print("Done!")
