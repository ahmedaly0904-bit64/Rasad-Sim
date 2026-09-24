"""Rasad on EoN — an epidemic spreading on a network.

EoN records one row per infection or recovery event, so every run returns
series of a different length, which Rasad rejects. The model below puts the
infected curve on a fixed time grid, so every run is measured at the same
instants and the runs can be compared step by step.

    python examples/eon_epidemic.py
"""

import EoN
import networkx as nx
import numpy as np

import rasad

# ——— simulation parameters ———
NODES = 500        # individuals in the network
TAU = 0.3          # infection rate
GAMMA = 1.0        # recovery rate
RHO = 0.02         # initial fraction infected

# The network is fixed across every run — the seed changes the course of the
# epidemic, not the structure of the community
GRAPH = nx.barabasi_albert_graph(NODES, 3, seed=1)


def run(params: dict, seed: int) -> dict:
    """One run of an epidemic on a network.

    t, S, I, R are all numpy arrays of the same length — but the length
    **differs between runs**, because every infection or recovery event
    records a row.
    """
    t, S, I, R = EoN.fast_SIR(
        GRAPH,
        tau=params.get("tau", TAU),
        gamma=params.get("gamma", GAMMA),
        rho=params.get("rho", RHO),
        rng=np.random.default_rng(seed),
    )

    # On a fixed time grid: every run is measured at the same instants, so
    # comparisons between runs are meaningful. Without this the lengths differ
    # and Rasad rejects them.
    grid = np.linspace(0.0, 15.0, 60)
    infected_on_grid = np.interp(grid, t, I)

    return {
        "peak_infected": I.max(),
        "final_infected_total": R[-1],
        "never_infected": S[-1],
        "epidemic_duration": t[-1],
        "took_off": 1.0 if R[-1] > 0.1 * NODES else 0.0,
        "infected_curve": infected_on_grid,
    }


def main() -> None:
    report = rasad.measure(run, params={}, runs=50)
    print(report.summary())

    if report.series:
        report.plot().write_html("eon_divergence.html")
        print("\nPlot written to: eon_divergence.html")


if __name__ == "__main__":
    main()
