"""
Visualization utilities for quantum-deformed datasets.

Provides plotting functions for:
- Grid comparison of classical vs deformed datasets
- Deformation vector fields
- Animation of time evolution
- Statistical analysis plots
"""

from typing import Dict, Tuple, Optional, List
import matplotlib.pyplot as plt
import numpy as np


def plot_moons_grid(
    X_base: np.ndarray,
    y: np.ndarray,
    X_t_dict: Dict[float, np.ndarray],
    cols: int = 3,
    figsize: Tuple[int, int] = (12, 8),
    cmap: str = "coolwarm",
    show_stats: bool = False
) -> plt.Figure:
    """
    Plot classical and quantum-deformed datasets in a grid layout.
    
    Creates a subplot grid comparing the original dataset with
    deformed variants at different evolution times.
    
    Args:
        X_base: Base classical dataset (n_samples, 2)
        y: Labels (n_samples,)
        X_t_dict: Dict mapping time values to deformed datasets
        cols: Columns in subplot grid
        figsize: Figure size (width, height) in inches
        cmap: Colormap for class coloring
        show_stats: If True, add statistical annotations
        
    Returns:
        matplotlib Figure object
        
    Raises:
        ValueError: If X_base and y have mismatched lengths
    """
    if X_base.shape[0] != len(y):
        raise ValueError(
            f"X_base and y must have same length: {X_base.shape[0]} vs {len(y)}"
        )
    
    t_values = sorted(X_t_dict.keys())
    n_plots = 1 + len(t_values)
    rows = int(np.ceil(n_plots / cols))

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).reshape(-1)

    # Plot base dataset
    ax0 = axes[0]
    scatter = ax0.scatter(
        X_base[:, 0], X_base[:, 1], 
        c=y, cmap=cmap, s=8, alpha=0.8
    )
    ax0.set_title("Classical Dataset (t = 0)", fontsize=10, fontweight='bold')
    ax0.set_xlabel("x")
    ax0.set_ylabel("y")
    ax0.grid(True, alpha=0.3, linestyle='--')
    ax0.set_aspect('equal', adjustable='box')
    
    if show_stats:
        stats = f"μ=({X_base[:,0].mean():.2f}, {X_base[:,1].mean():.2f})"
        ax0.annotate(stats, xy=(0.02, 0.98), xycoords='axes fraction',
                    fontsize=7, va='top', ha='left',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Plot deformed variants
    for idx, t in enumerate(t_values, start=1):
        if idx >= len(axes):
            break
        ax = axes[idx]
        X_def = X_t_dict[t]
        
        ax.scatter(
            X_def[:, 0], X_def[:, 1], 
            c=y, cmap=cmap, s=8, alpha=0.8
        )
        ax.set_title(f"Quantum-Deformed (t = {t:.2f})", fontsize=10)
        ax.set_xlabel("x'")
        ax.set_ylabel("y'")
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_aspect('equal', adjustable='box')
        
        if show_stats:
            stats = f"μ=({X_def[:,0].mean():.2f}, {X_def[:,1].mean():.2f})"
            ax.annotate(stats, xy=(0.02, 0.98), xycoords='axes fraction',
                       fontsize=7, va='top', ha='left',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Hide unused axes
    for j in range(n_plots, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    return fig


def plot_deformation_field(
    X_grid: np.ndarray,
    Y_grid: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    figsize: Tuple[int, int] = (10, 8),
    title: str = "Deformation Vector Field"
) -> plt.Figure:
    """
    Plot the deformation vector field as a quiver plot.
    
    Shows how quantum deformation displaces points across the
    input space, with vector magnitude indicated by color.
    
    Args:
        X_grid, Y_grid: Meshgrid coordinates
        U, V: Deformation vector components
        figsize: Figure size
        title: Plot title
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    magnitude = np.sqrt(U**2 + V**2)
    
    quiver = ax.quiver(
        X_grid, Y_grid, U, V, magnitude,
        cmap='viridis', scale=10, width=0.003
    )
    
    cbar = fig.colorbar(quiver, ax=ax, label='Displacement magnitude')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    return fig


def plot_deformation_comparison(
    X_base: np.ndarray,
    y: np.ndarray,
    X_def: np.ndarray,
    t: float,
    figsize: Tuple[int, int] = (14, 6)
) -> plt.Figure:
    """
    Side-by-side comparison with displacement vectors.
    
    Shows original and deformed datasets with arrows indicating
    the deformation applied to each point.
    
    Args:
        X_base: Original points
        y: Labels
        X_def: Deformed points
        t: Time parameter
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Original
    axes[0].scatter(X_base[:, 0], X_base[:, 1], c=y, cmap='coolwarm', s=12, alpha=0.8)
    axes[0].set_title('Original', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].grid(True, alpha=0.3)
    
    # Deformed
    axes[1].scatter(X_def[:, 0], X_def[:, 1], c=y, cmap='coolwarm', s=12, alpha=0.8)
    axes[1].set_title(f'Deformed (t={t:.2f})', fontsize=11, fontweight='bold')
    axes[1].set_xlabel("x'")
    axes[1].set_ylabel("y'")
    axes[1].grid(True, alpha=0.3)
    
    # Displacement vectors (subsample for clarity)
    step = max(1, len(X_base) // 100)
    X_sub = X_base[::step]
    X_def_sub = X_def[::step]
    y_sub = y[::step]
    
    axes[2].scatter(X_sub[:, 0], X_sub[:, 1], c=y_sub, cmap='coolwarm', s=20, alpha=0.5, marker='o')
    axes[2].scatter(X_def_sub[:, 0], X_def_sub[:, 1], c=y_sub, cmap='coolwarm', s=20, alpha=0.5, marker='x')
    
    for i in range(len(X_sub)):
        axes[2].annotate(
            '', xy=X_def_sub[i], xytext=X_sub[i],
            arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5, lw=0.5)
        )
    
    axes[2].set_title('Displacement Vectors', fontsize=11, fontweight='bold')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    axes[2].grid(True, alpha=0.3)
    
    fig.tight_layout()
    return fig


def plot_statistics(
    X_base: np.ndarray,
    X_t_dict: Dict[float, np.ndarray],
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """
    Plot statistical evolution of dataset properties over time.
    
    Shows how mean, standard deviation, and correlation evolve
    as deformation time increases.
    
    Args:
        X_base: Base dataset
        X_t_dict: Time-deformed variants
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    t_values = sorted(X_t_dict.keys())
    
    means_x = [X_t_dict[t][:, 0].mean() for t in t_values]
    means_y = [X_t_dict[t][:, 1].mean() for t in t_values]
    stds_x = [X_t_dict[t][:, 0].std() for t in t_values]
    stds_y = [X_t_dict[t][:, 1].std() for t in t_values]
    correlations = [np.corrcoef(X_t_dict[t][:, 0], X_t_dict[t][:, 1])[0, 1] for t in t_values]
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Means
    axes[0].plot(t_values, means_x, 'b-o', label='x mean', markersize=4)
    axes[0].plot(t_values, means_y, 'r-s', label='y mean', markersize=4)
    axes[0].set_xlabel('Time (t)')
    axes[0].set_ylabel('Mean')
    axes[0].set_title('Mean Evolution', fontsize=11, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Standard deviations
    axes[1].plot(t_values, stds_x, 'b-o', label='x std', markersize=4)
    axes[1].plot(t_values, stds_y, 'r-s', label='y std', markersize=4)
    axes[1].set_xlabel('Time (t)')
    axes[1].set_ylabel('Std Dev')
    axes[1].set_title('Spread Evolution', fontsize=11, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Correlation
    axes[2].plot(t_values, correlations, 'g-^', markersize=4)
    axes[2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[2].set_xlabel('Time (t)')
    axes[2].set_ylabel('Correlation')
    axes[2].set_title('X-Y Correlation Evolution', fontsize=11, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    fig.tight_layout()
    return fig


def plot_topology_preservation(
    metrics: Dict[float, float],
    figsize: Tuple[int, int] = (8, 5)
) -> plt.Figure:
    """
    Plot topology preservation scores vs time.
    
    Args:
        metrics: Dict mapping t → preservation score
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    t_values = sorted(metrics.keys())
    scores = [metrics[t] for t in t_values]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(t_values, scores, 'ko-', markersize=8, linewidth=2)
    ax.fill_between(t_values, scores, alpha=0.3)
    
    ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Perfect preservation')
    ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Random baseline')
    
    ax.set_xlabel('Evolution Time (t)', fontsize=11)
    ax.set_ylabel('k-NN Preservation Score', fontsize=11)
    ax.set_title('Topology Preservation vs Time', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig
