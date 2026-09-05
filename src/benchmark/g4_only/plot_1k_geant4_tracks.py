"""
Plots a 2D energy-deposition heatmap of 1000 GEANT4 proton tracks through
aluminum, using a logarithmic color scale. The output is saved as a PDF.
"""

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.absolute()))

    from matplotlib import pyplot as plt
    from matplotlib.colors import LogNorm

    from data_handling.reader import read_geant4_simulation_output
    from common.paths import BENCHMARKS_OUTPUTS, PATH_TO_GEANT4_DATA
    from common.enums import G4Columns
    from benchmark.system_vs_g4.pull_analysis.energy_deposition_pull_analysis import plot_tracks, organize_df, compute_hist_stats

    geant4_data = read_geant4_simulation_output(PATH_TO_GEANT4_DATA, verbose=True)
    df = organize_df(geant4_data, displace=-geant4_data.X.min())

    x_range = (df[G4Columns.X].min(), df[G4Columns.X].max())
    y_range = (df[G4Columns.Y].min(), df[G4Columns.Y].max())

    hist_2d, _, _ = compute_hist_stats(df, range_lims=[x_range, y_range])

    fig, ax = plt.subplots(1, 1, figsize=(10, 5), constrained_layout=True)
    im = plot_tracks(
        ax=ax,
        hist_2d=hist_2d,
        extent=(x_range[0], x_range[1], y_range[0], y_range[1]),
        norm=LogNorm(vmin=1, vmax=100),
        title=''
    )
    ax.set_xlabel('z (primary beam axis) [mm]')
    cbar = fig.colorbar(
        im,
        ax=ax,
        location='right',
        pad=0.01,
        aspect=20,
    )
    cbar.set_label('Energy Deposition [MeV]')
    plt.savefig(BENCHMARKS_OUTPUTS / "geant4_tracks.pdf", format="pdf", bbox_inches='tight')
    plt.show()
