"""
GEANT4-only self-consistency checks using 2D histogram comparisons.
Splits the active GEANT4 sample in half and compares the two halves against
each other, and plots a single 2D-histogram overview of the full sample.
"""

from common.paths import PATH_TO_GEANT4_DATA, BENCHMARKS_OUTPUTS
from data_handling.reader import read_geant4_simulation_output
from data_handling.plotter import plot_correlation_h2d_comparison, plot_correlation_h2d


def g4_vs_g4_comparison():
    df = read_geant4_simulation_output(PATH_TO_GEANT4_DATA, verbose=True)
    df  = df.query("EventNum < 1000")
    df1 = df.query("EventNum >= 500")
    df2 = df.query("EventNum < 500")

    fig, ax = plot_correlation_h2d_comparison(df1, df2)
    fig.show()
    fig.savefig(BENCHMARKS_OUTPUTS / "GEANT4_vs_GEANT4.pdf", format="pdf")


def hist_2d():
    df = read_geant4_simulation_output(PATH_TO_GEANT4_DATA, verbose=True)

    fig, ax = plot_correlation_h2d(df)
    fig.show()
    fig.savefig(BENCHMARKS_OUTPUTS / "GEANT4_2D_hist.pdf", format="pdf")


if __name__ == "__main__":
    # g4_vs_g4_comparison()
    hist_2d()
