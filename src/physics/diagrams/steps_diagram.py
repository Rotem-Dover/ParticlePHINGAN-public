"""
Generates a publication-quality 3D diagram of particle propagation steps through
a bulk material using PyVista. Renders the particle trajectory with scattering
events, energy labels, and physics callout annotations. Exports high-resolution
raster (PNG) and vector (PDF) figures styled for Nature Communications.

This is a script, not a library module: importing it opens an interactive
PyVista/VTK render window and runs the whole build immediately at import
time (`plotter.show()` blocks until the window is closed). Run it as
`python -m physics.diagrams.steps_diagram`; do not import it from other code.
"""
import vtkmodules.vtkRenderingFreeType
import vtkmodules.vtkRenderingMatplotlib
import pyvista as pv
import numpy as np

# ============================================================================
# NATURE COMMUNICATIONS STYLE CONFIGURATION
# ============================================================================
# Color palette optimized for colorblind accessibility and print quality
COLORS = {
    'track_start': '#000000', #'#E63946',      # Warm red - high visibility
    'track_end': '#F4A261',        # Amber orange
    'material': '#A8DADC',         # Soft cyan-blue (semi-transparent)
    'material_edge': '#457B9D',    # Darker blue for edges
    'interaction_point': '#1D3557', # Deep navy for interaction spheres
    'annotation': '#000000',#'#264653',       # Dark slate for text
    'callout': '#000000',#'#2A9D8F',          # Teal for physics callouts
    'connector': '#6C757D',        # Neutral gray for connector lines
    'background': '#FAFAFA',       # Off-white background
    'grid': '#E9ECEF',             # Very light gray grid
}

# Typography settings matching Nature style
FONT_CONFIG = {
    'family': 'times',
    'label_size': int(1.8*36),
    'equation_size': int(1.8*42),
    'energy_label_size': int(1.8*38),
}

# Figure dimensions (Nature recommends 89mm single column, 183mm double column)
FIGURE_WIDTH_PX = 4800  # High resolution for print
FIGURE_HEIGHT_PX = 4000

# ============================================================================
# 1. PLOTTER SETUP WITH PROFESSIONAL RENDERING
# ============================================================================
pv.global_theme.font.family = FONT_CONFIG['family']
pv.global_theme.font.color = COLORS['annotation']

plotter = pv.Plotter(off_screen=False, window_size=[1600, 1200])
plotter.enable_ssao(radius=3.0, bias=0.5, kernel_size=128)
plotter.set_background(COLORS['background'])
plotter.enable_anti_aliasing('ssaa')

# Professional 3-point lighting setup
plotter.enable_lightkit()  # Adds key, fill, and back lights

# ============================================================================
# 2. PARTICLE TRAJECTORY DEFINITION
# ============================================================================
# Define key points along the particle path through the material
p_start = np.array([0, 0, -8])      # Initial position (before material)
p_entry = np.array([0, 0, -5])      # Entry into material
p_step1 = np.array([1.5, 1.0, -2])  # First scattering event
p_step2 = np.array([-1.5, -0.5, 1]) # Second scattering event
p_exit = np.array([0, 0, 5])        # Exit from material
p_departure = np.array([0, 2, 8])

# # Calculate collinear departure point
# direction = p_exit - p_step2
# unit_direction = direction / np.linalg.norm(direction)
# p_departure = p_exit + unit_direction * 3.0  # Projecting 3 units out

points = np.array([p_start, p_entry, p_step1, p_step2, p_exit, p_departure])

# ============================================================================
# 3. PARTICLE TRACK VISUALIZATION
# ============================================================================
path = pv.MultipleLines(points)
path["progress"] = np.linspace(0, 1, path.n_points)
path_tube = path.tube(radius=0.10)  # Slightly thicker for visibility


def create_arrow(start, end, shift=True):
    """Create a directional arrow cone between two points."""
    direction = end - start
    length = np.linalg.norm(direction)
    center = end - (direction / length) * 0.35 if shift else end
    return pv.Cone(
        center=center,
        direction=direction,
        height=0.70,
        radius=0.28,
        resolution=64  # Smooth cone
    )

# ============================================================================
# 4. BULK MATERIAL (Semi-transparent with subtle edges)
# ============================================================================
bulk = pv.Cube(center=(0, 0, 0), x_length=10, y_length=10, z_length=10)
plotter.add_mesh(
    bulk,
    color=COLORS['material'],
    opacity=0.18,
    pbr=True,
    metallic=0.3,
    roughness=0.6,
    smooth_shading=True
)

# Add subtle edge highlighting for depth
bulk_edges = bulk.extract_feature_edges()
plotter.add_mesh(bulk_edges, color=COLORS['material_edge'], line_width=2, opacity=0.4)

# ============================================================================
# 5. SUBTLE INTERNAL GRID (Shows material structure)
# ============================================================================
grid_points = np.linspace(-5, 5, 6)
x, y, z = np.meshgrid(grid_points, grid_points, grid_points)
struct_grid = pv.StructuredGrid(x, y, z)
grid_tubes = struct_grid.extract_all_edges().tube(radius=0.008)
plotter.add_mesh(grid_tubes, color=COLORS['grid'], opacity=0.06)

# ============================================================================
# 6. SHADOW-CATCHING FLOOR
# ============================================================================
floor = pv.Plane(center=(0, -5.01, 0), direction=(0, 1, 0), i_size=40, j_size=40)
plotter.add_mesh(floor, color="white", pbr=True, metallic=0.0, roughness=1.0)

# ============================================================================
# 7. PARTICLE TRACK AND INTERACTION MARKERS
# ============================================================================
# Color gradient track (warm colors: red -> orange for energy loss visualization)
plotter.add_mesh(
    path_tube,
    scalars="progress",
    cmap="YlOrRd_r",  # Nature-friendly colormap
    pbr=True,
    metallic=0.8,
    roughness=0.25,
    show_scalar_bar=False
)

# Directional arrows at each step
for i in range(1, len(points)):
    # dont shift the final arrow to departure point
    shift = (i != len(points) - 1)
    arrow = create_arrow(points[i - 1], points[i], shift=shift)
    plotter.add_mesh(
        arrow,
        color=COLORS['track_end'],
        pbr=True,
        metallic=0.9,
        roughness=0.2
    )

# Add interaction point markers (spheres at scattering locations)
interaction_points = [p_entry, p_step1, p_step2, p_exit]
for i, pt in enumerate(interaction_points):
    sphere = pv.Sphere(radius=0.25, center=pt)
    plotter.add_mesh(
        sphere,
        color=COLORS['interaction_point'],
        pbr=True,
        metallic=0.4,
        roughness=0.3
    )

# ============================================================================
# 8. SCIENTIFIC LABELS AND ANNOTATIONS
# ============================================================================
label_L_coords = []
label_S_coords = []
label_L_texts = [r"$L_1$", r"$L_2$", r"$L_3$"]
label_S_texts = [r"$S_0$", r"$S_1$", r"$S_2$", r"$S_3$"]  # Clean numbering

# Calculate label positions
for i in range(1, 4):
    mid = (points[i] + points[i + 1]) / 2.0
    end = points[i]
    label_L_coords.append(mid + [0, +0.6, 0])
    label_S_coords.append(end + [0, -1.0, 0])
label_S_coords.append(points[4] + [0, -1.0, 0])

# Path length labels
plotter.add_point_labels(
    label_L_coords,
    label_L_texts,
    font_size=FONT_CONFIG['label_size'],
    text_color=COLORS['annotation'],
    bold=True,
    shadow=True,
    shape=None,
    show_points=False,
    always_visible=True,
    font_family=FONT_CONFIG['family']
)

# State labels
plotter.add_point_labels(
    label_S_coords,
    label_S_texts,
    font_size=FONT_CONFIG['label_size'],
    text_color=COLORS['annotation'],
    bold=True,
    shadow=True,
    shape=None,
    show_points=False,
    always_visible=True,
    font_family=FONT_CONFIG['family']
)

# Energy labels at entry and exit
plotter.add_point_labels(
    [points[0] + [0, 1.2, 0], points[-1] + [0, 1.2, 0]],
    [r"$E_0$", r"$E_{\rm f}$"],
    font_size=FONT_CONFIG['energy_label_size'],
    text_color=COLORS['track_start'],
    bold=True,
    shadow=True,
    shape=None,
    show_points=False,
    always_visible=True,
    font_family=FONT_CONFIG['family']
)

# ============================================================================
# 9. PHYSICS CALLOUTS (Equations with connectors)
# ============================================================================
# State description callout
# p_physics_S_text = np.array([-2, 7, -4])
p_physics_S_text = np.array([-2, 7, -8])
physics_S_formula = r"$S_i = (t_i, \mathbf{r}_i, \mathbf{p}_i, E_i)$"

plotter.add_point_labels(
    [p_physics_S_text],
    [physics_S_formula],
    font_size=FONT_CONFIG['equation_size'],
    text_color=COLORS['callout'],
    bold=True,
    shadow=True,
    shape=None,
    show_points=False,
    always_visible=True,
    font_family=FONT_CONFIG['family']
)

def add_dashed_connector(plotter, start_pt, end_pt, color, n_segments=15, radius=0.03, opacity=0.7):
    """Adds a 3D dashed tube connector to the plotter."""
    points = np.linspace(start_pt, end_pt, n_segments)

    # Plot every other segment to create the 'dash'
    for i in range(0, len(points) - 1, 2):
        segment = pv.Line(points[i], points[i + 1])
        segment_tube = segment.tube(radius=radius)
        plotter.add_mesh(segment_tube, color=color, opacity=opacity)

add_dashed_connector(
    plotter,
    start_pt=p_step1 + [0, 0.3, 0],
    end_pt=p_physics_S_text,
    color=COLORS['connector']
)

# Energy loss callout
p_physics_E_text = np.array([2, 7, 2])
physics_E_formula = r"$\Delta E_i = E_i - E_{i-1}$"

plotter.add_point_labels(
    [p_physics_E_text],
    [physics_E_formula],
    font_size=FONT_CONFIG['equation_size'],
    text_color=COLORS['callout'],
    bold=True,
    shadow=True,
    shape=None,
    show_points=False,
    always_visible=True,
    font_family=FONT_CONFIG['family']
)

# Connector for energy callout
coord_dE2 = p_step1/4 + 3*p_step2/4
start_E = coord_dE2 + [0, 0.3, 0]
end_E = p_physics_E_text# - [0, 0.8, 0]
add_dashed_connector(plotter, start_E, end_E, COLORS['connector'])

# ============================================================================
# 10. CAMERA AND SCENE SETUP
# ============================================================================
# Remove default axes for cleaner look (can add custom scale bar if needed)
# plotter.add_axes()

# Professional camera angle
plotter.camera_position = [(22, 16, 22), (0, 0, 0), (0, 1, 0)]
plotter.camera.zoom(1.1)

# ============================================================================
# 11. HIGH-RESOLUTION EXPORT
# ============================================================================
plotter.show(auto_close=False)

# Export at publication resolution (300+ DPI for print)
plotter.screenshot(
    "particle_propagation_steps.png",
    transparent_background=False,
    window_size=[FIGURE_WIDTH_PX, FIGURE_HEIGHT_PX]
)

# Vector format for scalable graphics
plotter.save_graphic("particle_propagation_steps.pdf")

# Also save with transparent background for compositing
plotter.screenshot(
    "particle_propagation_steps_transparent.png",
    transparent_background=True,
    window_size=[FIGURE_WIDTH_PX, FIGURE_HEIGHT_PX]
)

print("=" * 60)
print("Nature Communications Figure Export Complete")
print("=" * 60)
print("Files saved:")
print("  - particle_propagation_steps.png (high-res raster)")
print("  - particle_propagation_steps.pdf (vector format)")
print("  - particle_propagation_steps_transparent.png (for compositing)")
print("")
print("Note: Ensure LaTeX (TeX Live or similar) is installed for")
print("proper mathematical typesetting in labels.")
print("=" * 60)

plotter.close()
