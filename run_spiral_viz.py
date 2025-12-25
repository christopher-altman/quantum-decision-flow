#!/usr/bin/env python3
"""run_spiral_viz_README_2x3.py

Generate a README-friendly 2×3 grid visualization for the SPIRALS dataset under
HEISENBERG dynamics.

Important behavior notes:
- The plotting utility already includes a "base / classical" panel.
- Therefore we EXCLUDE t=0.0 from the quantum sweep to avoid 7 panels.

Panels (6 total):
- Base panel (retitled): Quantum-Deformed (t = 0.00)
- Deformed panels at: t = (0.5, 1.0, 2.0, 3.0, 5.0)

Output:
  figures/spiral_heisenberg_visualization.png
"""

from pathlib import Path

import matplotlib.pyplot as plt

from src import (
    QuantumMoonsConfig,
    DatasetType,
    HamiltonianType,
    generate_quantum_deformed_moons,
    plot_moons_grid,
)


def main() -> None:
    out_dir = Path("figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2×3 layout: base panel + 5 representative deformed times = 6 panels total
    t_values = (0.5, 1.0, 2.0, 3.0, 5.0)

    cfg = QuantumMoonsConfig(
        n_samples=400,
        dataset_type=DatasetType.SPIRALS,
        hamiltonian_type=HamiltonianType.HEISENBERG,
        t_values=t_values,
    )

    X_base, y, X_t, metrics = generate_quantum_deformed_moons(cfg, compute_metrics=True)

    # show_stats can make panels busy; keep False for a clean README figure
    fig = plot_moons_grid(X_base, y, X_t, show_stats=False)

    # Encourage a legible 2×3 layout for GitHub
    try:
        fig.set_size_inches(14, 9)
    except Exception:
        pass

    # Make geometry faithful (no stretching) across all panels
    try:
        for ax in fig.get_axes():
            ax.set_aspect("equal", adjustable="box")
    except Exception:
        pass

    # Retitle the base panel to match the "t = 0.00" convention
    # (The base panel is the first axis in typical implementations.)
    try:
        axes = fig.get_axes()
        if axes:
            axes[0].set_title("Quantum-Deformed (t = 0.00)")
    except Exception:
        pass

    # Layout tuning: prefer a tight layout, then slightly reduce whitespace
    try:
        fig.tight_layout()
    except Exception:
        pass
    try:
        fig.subplots_adjust(wspace=0.15, hspace=0.20)
    except Exception:
        pass

    out_path = out_dir / "spiral_heisenberg_visualization.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[ok] saved: {out_path}")
    print(f"[ok] metrics: {metrics}")


if __name__ == "__main__":
    main()
