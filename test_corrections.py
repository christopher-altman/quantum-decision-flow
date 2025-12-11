#!/usr/bin/env python
"""
Test script to validate all corrections to the quantum-moons package.
Run this to verify everything works correctly.
"""

import sys
import traceback


def test_imports():
    """Test that all modules import correctly."""
    print("Testing imports...")
    try:
        from src import generator, deformation, visualization
        print("✓ All modules import successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("  Note: This is expected if PennyLane is not installed.")
        print("  Run: pip install -r requirements.txt")
        return False


def test_config_validation():
    """Test configuration validation."""
    print("\nTesting configuration validation...")
    try:
        from src.generator import QuantumMoonsConfig
        
        # Valid config
        cfg = QuantumMoonsConfig()
        print("✓ Valid configuration accepted")
        
        # Test invalid n_samples
        try:
            bad_cfg = QuantumMoonsConfig(n_samples=-1)
            print("✗ Failed to reject negative n_samples")
            return False
        except ValueError:
            print("✓ Correctly rejected negative n_samples")
        
        # Test invalid noise
        try:
            bad_cfg = QuantumMoonsConfig(noise=-0.1)
            print("✗ Failed to reject negative noise")
            return False
        except ValueError:
            print("✓ Correctly rejected negative noise")
        
        return True
    except Exception as e:
        print(f"✗ Configuration validation error: {e}")
        traceback.print_exc()
        return False


def test_type_hints():
    """Verify type hints are present."""
    print("\nTesting type hints...")
    try:
        from src.deformation import deform_points, xy_to_angles, t2_expectations
        from src.generator import generate_classical_moons, generate_quantum_deformed_moons
        from src.visualization import plot_moons_grid
        
        # Check if type hints exist
        funcs = [
            deform_points, xy_to_angles, t2_expectations,
            generate_classical_moons, generate_quantum_deformed_moons,
            plot_moons_grid
        ]
        
        for func in funcs:
            if not hasattr(func, '__annotations__') or not func.__annotations__:
                print(f"✗ {func.__name__} missing type hints")
                return False
        
        print("✓ All functions have type hints")
        return True
    except Exception as e:
        print(f"✗ Type hint check error: {e}")
        return False


def test_docstrings():
    """Verify docstrings are present."""
    print("\nTesting docstrings...")
    try:
        from src.deformation import deform_points, xy_to_angles, t2_expectations
        from src.generator import generate_classical_moons, generate_quantum_deformed_moons
        from src.visualization import plot_moons_grid
        
        funcs = [
            deform_points, xy_to_angles, t2_expectations,
            generate_classical_moons, generate_quantum_deformed_moons,
            plot_moons_grid
        ]
        
        for func in funcs:
            if not func.__doc__ or len(func.__doc__.strip()) < 20:
                print(f"✗ {func.__name__} missing or inadequate docstring")
                return False
        
        print("✓ All functions have comprehensive docstrings")
        return True
    except Exception as e:
        print(f"✗ Docstring check error: {e}")
        return False


def test_full_pipeline():
    """Test the full generation pipeline (if PennyLane is available)."""
    print("\nTesting full pipeline...")
    try:
        from src.generator import QuantumMoonsConfig, generate_quantum_deformed_moons
        from src.visualization import plot_moons_grid
        
        # Generate with small dataset for speed
        cfg = QuantumMoonsConfig(n_samples=100, t_values=(0.0, 0.5, 1.0))
        X_base, y, X_t = generate_quantum_deformed_moons(cfg)
        
        # Validate outputs
        assert X_base.shape == (100, 2), f"Unexpected X_base shape: {X_base.shape}"
        assert y.shape == (100,), f"Unexpected y shape: {y.shape}"
        assert len(X_t) == 3, f"Expected 3 time variants, got {len(X_t)}"
        
        # Test that deformation actually changes the data
        import numpy as np
        for t, X_def in X_t.items():
            if t == 0.0:
                continue
            assert not np.allclose(X_base, X_def), f"Deformation at t={t} had no effect"
        
        print("✓ Full pipeline works correctly")
        print(f"  Generated {len(X_base)} samples with {len(X_t)} time variants")
        return True
        
    except ImportError as e:
        print(f"⚠ Cannot test full pipeline (PennyLane not installed): {e}")
        return True  # Don't fail if dependencies not installed
    except Exception as e:
        print(f"✗ Pipeline error: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 70)
    print("QUANTUM MOONS - CORRECTION VALIDATION")
    print("=" * 70)
    
    tests = [
        test_imports,
        test_config_validation,
        test_type_hints,
        test_docstrings,
        test_full_pipeline,
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"\n✗ Unexpected error in {test.__name__}: {e}")
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All corrections verified successfully!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
