"""
Builds and exports flowchart diagrams for the paper illustrating the decision
trees used in the continuous energy-loss and secondary energy-loss fluctuation
models. Uses matplotlib patches and annotations to render publication-ready PDFs.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


class FlowchartBuilder:
    def __init__(self, figsize=(16, 9), font_size=18, xlim=(-20, 150), ylim=(-15, 105)):
        self.fig, self.ax = plt.subplots(figsize=figsize)

        # Set limits dynamically based on the specific tree's needs
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.axis('off')

        self.font_size = font_size
        self.colors = {
            'decision': '#FFF2CC', 'process': '#DAE8FC',
            'process_green': '#D5E8D4', 'terminator': '#F5F5F5',
            'connector': '#E0E0E0', 'edge': '#333333',
            'text': '#000000'
        }
        self.nodes = {}

    def add_node(self, name, label, x, y, w, h, kind='process', color_override=None):
        color = color_override if color_override else self.colors.get(kind, '#FFFFFF')
        if kind == 'circle': h = w
        self.nodes[name] = {'label': label, 'x': x, 'y': y, 'w': w, 'h': h, 'kind': kind, 'color': color}

    def draw_nodes(self):
        for name, node in self.nodes.items():
            x, y, w, h = node['x'], node['y'], node['w'], node['h']
            if node['kind'] == 'decision':
                patch = patches.Polygon([[x, y + h / 2], [x + w / 2, y + h], [x + w, y + h / 2], [x + w / 2, y]],
                                        closed=True, edgecolor=self.colors['edge'], facecolor=node['color'],
                                        linewidth=1.5, zorder=2)
            elif node['kind'] == 'ellipse':
                patch = patches.Ellipse((x + w / 2, y + h / 2), w, h, edgecolor=self.colors['edge'],
                                        facecolor=node['color'], linewidth=1.5, zorder=2)
            elif node['kind'] == 'circle':
                patch = patches.Circle((x + w / 2, y + h / 2), radius=w / 2, edgecolor=self.colors['edge'],
                                       facecolor=node['color'], linewidth=1.5, zorder=2)
            elif node['kind'] == 'container':
                patch = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=2,rounding_size=2",
                                               edgecolor='#999999', facecolor='none', linestyle='--', linewidth=1.5,
                                               zorder=1)
            else:
                patch = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=1",
                                               edgecolor=self.colors['edge'], facecolor=node['color'], linewidth=1.5,
                                               zorder=2)
            self.ax.add_patch(patch)
            if node['kind'] != 'container':
                self.ax.text(x + w / 2, y + h / 2, node['label'], ha='center', va='center', fontsize=self.font_size,
                             color=self.colors['text'], wrap=True, zorder=3)

    def connect(self, start_name, end_name, label=None, exit_side='auto', entry_side='auto', label_offset=(0, 0),
                arrow_head=True, bold_arrow=False):
        n1, n2 = self.nodes[start_name], self.nodes[end_name]
        c1 = (n1['x'] + n1['w'] / 2, n1['y'] + n1['h'] / 2)
        c2 = (n2['x'] + n2['w'] / 2, n2['y'] + n2['h'] / 2)

        # Determine Start Point
        if n1['kind'] == 'decision':
            if exit_side == 'left':
                start = (n1['x'], c1[1])
            elif exit_side == 'right':
                start = (n1['x'] + n1['w'], c1[1])
            elif exit_side == 'top':
                start = (c1[0], n1['y'] + n1['h'])
            else:
                start = (c1[0], n1['y'])  # bottom default
        elif n1['kind'] == 'circle':
            r = n1['w'] / 2
            if exit_side == 'left':
                start = (c1[0] - r, c1[1])
            elif exit_side == 'right':
                start = (c1[0] + r, c1[1])
            elif exit_side == 'bottom':
                start = (c1[0], c1[1] - r)
            elif exit_side == 'top':
                start = (c1[0], c1[1] + r)
            else:
                start = (c1[0], c1[1] - r)
        else:
            if exit_side == 'left':
                start = (n1['x'], c1[1])
            elif exit_side == 'right':
                start = (n1['x'] + n1['w'], c1[1])
            elif exit_side == 'top':
                start = (c1[0], n1['y'] + n1['h'])
            else:
                start = (c1[0], n1['y'])

        # Determine End Point
        if entry_side != 'auto':
            if entry_side == 'left':
                end = (n2['x'], c2[1])
            elif entry_side == 'right':
                end = (n2['x'] + n2['w'], c2[1])
            elif entry_side == 'top':
                end = (c2[0], n2['y'] + n2['h'])
            elif entry_side == 'bottom':
                end = (c2[0], n2['y'])
        else:
            if c2[1] > c1[1]:
                end = (c2[0], n2['y'])
            else:
                end = (c2[0], n2['y'] + n2['h'])

        # Routing Logic
        x_vals, y_vals = [], []
        if exit_side in ['left', 'right']:
            mid_x = (start[0] + end[0]) / 2
            if (exit_side == 'left' and end[0] < start[0]) or (exit_side == 'right' and end[0] > start[0]):
                x_vals = [start[0], end[0], end[0]]
                y_vals = [start[1], start[1], end[1]]
            else:
                x_vals = [start[0], mid_x, mid_x, end[0], end[0]]
                y_vals = [start[1], start[1], end[1], end[1], end[1]]
        else:
            mid_y = (start[1] + end[1]) / 2
            x_vals = [start[0], start[0], end[0], end[0]]
            y_vals = [start[1], mid_y, mid_y, end[1]]

        linewidth = 2.5 if not bold_arrow else 5.
        self.ax.plot(x_vals, y_vals, color=self.colors['edge'], linewidth=linewidth, zorder=1)

        if arrow_head:
            self.ax.annotate('', xy=(x_vals[-1], y_vals[-1]), xytext=(x_vals[-2], y_vals[-2]),
                             arrowprops=dict(arrowstyle='->', color=self.colors['edge'], lw=2))

        if label:
            lx, ly = x_vals[0], y_vals[0]
            if exit_side == 'left':
                lx -= 6; ly += 3
            elif exit_side == 'right':
                lx += 6; ly += 3
            elif exit_side == 'bottom':
                ly -= 5; lx += 1
            lx += label_offset[0]
            ly += label_offset[1]
            bbox = dict(boxstyle="round,pad=0.2", fc="none", ec="none", alpha=0.8)
            self.ax.text(lx, ly, label, fontsize=self.font_size - 2, color=self.colors['edge'],
                         fontweight='bold', ha='center', va='center', bbox=bbox, zorder=4)

    def save(self, filename):
        plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.show()


def create_continuous_tree():
    # Adjusted xlim and ylim to encompass all nodes (lowest Y is -18, furthest X is ~160)
    builder = FlowchartBuilder(figsize=(18, 12), xlim=(-30, 180), ylim=(-60, 170))

    diamond_w = 50
    diamond_h = 30

    y_0 = 135#125  # Starting Y position for the top node
    dy = 25  # Vertical spacing between rows

    n_rows = 7
    ys = np.array([y_0 - i * dy for i in range(n_rows)])
    ys[3:] -= 30
    ys[-2] -= 20
    ys[-1] -= 10


    # Row 1 (Top)
    builder.add_node('start', 'Mean loss\nbelow threshold?', 50, ys[0], diamond_w, diamond_h, 'decision')
    plt.text(47, ys[0], 'Very thin limit', fontsize=builder.font_size + 5, fontweight='bold',
             color='black',  # builder.colors['edge'],
             ha='center', va='center', zorder=4)

    # Row 2
    builder.add_node('bethe', 'Deterministic\nBethe-Bloch', 5, ys[1], 40, 15, 'process')
    builder.add_node('thick', 'Thick limit?', 100, ys[1], 35, diamond_h, 'decision')
    plt.text(138, ys[1], 'Thick limit', fontsize=builder.font_size + 5, fontweight='bold',
             color='black',  # builder.colors['edge'],
             ha='center', va='center', zorder=4)

    # Row 3
    builder.add_node('split', '+', 75, ys[2], 10, 10, 'circle', color_override='#E0E0E0')
    builder.add_node('trunc_gauss_1', 'Truncated\nGaussian', 140, ys[2], 30, 15, 'process')

    # Row 4
    builder.add_node('exc_high', 'Excitation:\nHigh rate?', 20, ys[3], diamond_w, diamond_h, 'decision')
    builder.add_node('ion_high', 'Ionization:\nHigh rate?', 90, ys[3], diamond_w, diamond_h, 'decision')
    plt.text(45, ys[2] - 5, 'Thin limit excitation', fontsize=builder.font_size + 5, fontweight='bold',
             color='black',  # builder.colors['edge'],
             ha='center', va='center', zorder=4)
    plt.text(115, ys[2] - 5, 'Thin limit ionization', fontsize=builder.font_size + 5, fontweight='bold',
             color='black',  # builder.colors['edge'],
             ha='center', va='center', zorder=4)

    # Row 5
    builder.add_node('exc_trunc',   'Truncated\nGaussian', -10, ys[4], 35, 15, 'process')
    builder.add_node('poisson_gen', 'Generate\nPoisson number', 45, ys[4], 35, 15, 'process_green')
    builder.add_node('ion_pure',    'Compound\nPoisson process', 85, ys[4], 35, 15, 'process')
    builder.add_node('ion_mix',     'Compound Poisson\nprocess +\n Truncated Gauss', 125, ys[4], 40, 22, 'process')

    # Row 6
    builder.add_node('check_pos', 'number > 0?', 45, ys[5], diamond_w, diamond_h, 'decision')

    # Row 7 (Bottom)
    builder.add_node('poisson_uni', 'Poisson\n+ Uniform', 15, ys[6], 35, 12, 'process')
    builder.add_node('zero', '0', 90, ys[6], 15, 12, 'process')

    builder.draw_nodes()

    # Connections
    builder.connect('start', 'bethe', 'Yes', exit_side='left', bold_arrow=True)
    builder.connect('start', 'thick', 'No', exit_side='right')

    builder.connect('thick', 'split', 'No', exit_side='left')
    builder.connect('thick', 'trunc_gauss_1', 'Yes', exit_side='right', bold_arrow=True)


    builder.connect('split', 'exc_high', '', exit_side='bottom', bold_arrow=True)
    builder.connect('split', 'ion_high', '', exit_side='bottom', bold_arrow=True)

    builder.connect('exc_high', 'exc_trunc', 'Yes', exit_side='left')
    builder.connect('exc_high', 'poisson_gen', 'No', exit_side='right')

    builder.connect('ion_high', 'ion_pure', 'No', exit_side='left')
    builder.connect('ion_high', 'ion_mix', 'Yes', exit_side='right')

    builder.connect('poisson_gen', 'check_pos', '')
    builder.connect('check_pos', 'poisson_uni', 'Yes', exit_side='left')
    builder.connect('check_pos', 'zero', 'No', exit_side='right')

    builder.save('continuous_decision_tree.pdf')


def create_secondary_tree():
    # Adjusted limits for the secondary tree
    builder = FlowchartBuilder(figsize=(14, 12), xlim=(10, 130), ylim=(-10, 135))

    builder.add_node('compute_psec',
                     r'Compute $P^{\rm sec} = \frac{\sigma_{\text{ion}}(E - \Delta E^{\rm cnt})}{\sigma_{\text{ion}}(E)}$',
                     55, 115, 40, 12, 'process')
    builder.add_node('sample_r', r'Sample uniform $r$', 55, 99, 40, 12, 'process_green')
    builder.add_node('decide_sec', r'$r < P^{\rm sec}$?', 55, 77, 40, 18, 'decision')
    builder.add_node('zero', '0', 105, 81, 15, 10, 'process')

    builder.add_node('sample_u', r'Sample uniform $u$', 55, 62, 40, 12, 'process_green')

    builder.add_node('container', '', 45, 2, 60, 68, 'container')
    builder.ax.text(40, 35, 'Rejection Loop', ha='center', va='center', fontsize=16, fontweight='bold', rotation=90)

    builder.add_node('calc_E', r'Compute $\Delta E_{\rm ion}^{\rm sec}(u)$', 55, 47, 40, 12, 'process')
    builder.add_node('calc_y', r'Compute $P_{\rm accept}(\Delta E_{\rm ion}^{\rm sec})$', 55, 32, 40, 12, 'process_green')
    builder.add_node('sample_q', r'Sample uniform $q_n$', 55, 17, 40, 12, 'process_green')

    builder.add_node('check', r'$P_{\rm accept} > q_n$ ?', 60, 2, 30, 12, 'decision')
    builder.add_node('term', 'Terminate', 25, 2, 25, 12, 'process')

    builder.draw_nodes()

    # Connections
    builder.connect('compute_psec', 'sample_r', '', exit_side='bottom')
    builder.connect('sample_r', 'decide_sec', '', exit_side='bottom')

    builder.connect('decide_sec', 'sample_u', 'Yes', exit_side='left', entry_side='top', label_offset=(10, -10))
    builder.connect('decide_sec', 'zero', 'No', exit_side='right', entry_side='left', arrow_head=False)

    builder.connect('sample_u', 'calc_E', '', exit_side='bottom')
    builder.connect('calc_E', 'calc_y', '', exit_side='bottom')
    builder.connect('calc_y', 'sample_q', '', exit_side='bottom')
    builder.connect('sample_q', 'check', '', exit_side='bottom')

    builder.connect('check', 'term', 'Yes', exit_side='left', entry_side='right', arrow_head=False)

    # Loop back
    n_check = builder.nodes['check']
    n_target = builder.nodes['sample_u']

    x_out = n_check['x'] + n_check['w']
    y_out = n_check['y'] + n_check['h'] / 2

    x_loop = x_out + 10
    y_target = n_target['y'] + n_target['h'] / 2
    x_target_in = n_target['x'] + n_target['w']

    path_x = [x_out, x_loop, x_loop, x_target_in]
    path_y = [y_out, y_out, y_target, y_target]

    builder.ax.plot(path_x, path_y, color='#333', lw=1.5, zorder=1)
    builder.ax.annotate('', xy=(path_x[-1], path_y[-1]), xytext=(path_x[-1] + 2, path_y[-1]),
                        arrowprops=dict(arrowstyle='->', color='#333', lw=2))

    builder.ax.text(x_out + 2, y_out + 2, 'No', fontsize=14, fontweight='bold',
                    bbox=dict(boxstyle="round", fc="none", ec="none", alpha=0.8))

    builder.save('secondary_decision_tree.pdf')


if __name__ == "__main__":
    create_continuous_tree()
    create_secondary_tree()