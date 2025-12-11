"""
Unit tests for visualization module.

Tests cover:
- Plot generation without errors
- Input validation
- Figure properties
"""

import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '..')


class TestPlotMoonsGrid:
    """Tests for grid plotting function."""
    
    def test_basic_plot(self):
        """Should create plot without errors."""
        from src.visualization import plot_moons_grid
        
        X_base = np.random.randn(50, 2)
        y = np.random.randint(0, 2, 50)
        X_t = {0.5: X_base + 0.1, 1.0: X_base + 0.2}
        
        fig = plot_moons_grid(X_base, y, X_t)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_mismatched_shapes_rejected(self):
        """Mismatched X_base and y should raise ValueError."""
        from src.visualization import plot_moons_grid
        
        X_base = np.random.randn(50, 2)
        y = np.random.randint(0, 2, 40)  # Wrong length
        X_t = {0.5: X_base}
        
        with pytest.raises(ValueError):
            plot_moons_grid(X_base, y, X_t)
    
    def test_custom_figsize(self):
        """Custom figsize should be respected."""
        from src.visualization import plot_moons_grid
        
        X_base = np.random.randn(20, 2)
        y = np.random.randint(0, 2, 20)
        X_t = {0.5: X_base}
        
        fig = plot_moons_grid(X_base, y, X_t, figsize=(15, 10))
        assert fig.get_figwidth() == 15
        assert fig.get_figheight() == 10
        plt.close(fig)
    
    def test_single_time_value(self):
        """Should work with single time value."""
        from src.visualization import plot_moons_grid
        
        X_base = np.random.randn(20, 2)
        y = np.random.randint(0, 2, 20)
        X_t = {1.0: X_base + 0.1}
        
        fig = plot_moons_grid(X_base, y, X_t, cols=2)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_many_time_values(self):
        """Should handle many time values."""
        from src.visualization import plot_moons_grid
        
        X_base = np.random.randn(20, 2)
        y = np.random.randint(0, 2, 20)
        X_t = {t: X_base + 0.1*t for t in [0.5, 1.0, 1.5, 2.0, 2.5]}
        
        fig = plot_moons_grid(X_base, y, X_t, cols=3)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotStatistics:
    """Tests for statistics plotting."""
    
    def test_basic_stats_plot(self):
        """Should create statistics plot without errors."""
        from src.visualization import plot_statistics
        
        X_base = np.random.randn(50, 2)
        X_t = {0.0: X_base, 0.5: X_base + 0.1, 1.0: X_base + 0.2}
        
        fig = plot_statistics(X_base, X_t)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotDeformationComparison:
    """Tests for deformation comparison plot."""
    
    def test_basic_comparison(self):
        """Should create comparison plot without errors."""
        from src.visualization import plot_deformation_comparison
        
        X_base = np.random.randn(50, 2)
        y = np.random.randint(0, 2, 50)
        X_def = X_base + 0.2
        
        fig = plot_deformation_comparison(X_base, y, X_def, t=1.0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotTopologyPreservation:
    """Tests for topology preservation plot."""
    
    def test_basic_topology_plot(self):
        """Should create topology plot without errors."""
        from src.visualization import plot_topology_preservation
        
        metrics = {0.5: 0.9, 1.0: 0.8, 1.5: 0.7, 2.0: 0.6}
        
        fig = plot_topology_preservation(metrics)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
