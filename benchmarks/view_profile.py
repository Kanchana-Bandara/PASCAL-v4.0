#!/usr/bin/env python3
"""Profile viewer for PASCAL simulation.

Usage:
    python view_profile.py profile.stats
"""

import sys
import pstats
from pstats import SortKey


def view_profile(filename='profile.stats'):
    """Display profile statistics in a readable format."""

    try:
        p = pstats.Stats(filename)
    except FileNotFoundError:
        print(f"Error: Profile file '{filename}' not found.")
        print("\nTo create a profile, run:")
        print("  python -m cProfile -o profile.stats your_script.py")
        return

    p.strip_dirs()

    # Summary
    print("\n" + "="*80)
    print("PROFILE SUMMARY")
    print("="*80)
    p.print_stats(0)  # Just the summary

    # Top functions by cumulative time
    print("\n" + "="*80)
    print("TOP 30 FUNCTIONS BY CUMULATIVE TIME (including subcalls)")
    print("="*80)
    print("This shows where the program spends most time overall")
    print("-"*80)
    p.sort_stats(SortKey.CUMULATIVE).print_stats(30)

    # Top functions by time in function itself
    print("\n" + "="*80)
    print("TOP 30 FUNCTIONS BY INTERNAL TIME (excluding subcalls)")
    print("="*80)
    print("This shows actual computation bottlenecks")
    print("-"*80)
    p.sort_stats(SortKey.TIME).print_stats(30)

    # PASCAL-specific functions
    print("\n" + "="*80)
    print("PASCAL SIMULATION FUNCTIONS (individual.py)")
    print("="*80)
    p.sort_stats(SortKey.CUMULATIVE).print_stats('individual')

    print("\n" + "="*80)
    print("PASCAL SIMULATION FUNCTIONS (coupler.py)")
    print("="*80)
    p.sort_stats(SortKey.CUMULATIVE).print_stats('coupler')

    # Growth/development module
    print("\n" + "="*80)
    print("GROWTH & DEVELOPMENT MODULE")
    print("="*80)
    p.sort_stats(SortKey.CUMULATIVE).print_stats('biology/growth')

    # Vertical migration module
    print("\n" + "="*80)
    print("VERTICAL MIGRATION MODULE")
    print("="*80)
    p.sort_stats(SortKey.CUMULATIVE).print_stats('vertical_migration')

    # Survival module
    print("\n" + "="*80)
    print("SURVIVAL MODULE")
    print("="*80)
    p.sort_stats(SortKey.CUMULATIVE).print_stats('survival')

    # Recommendations
    print("\n" + "="*80)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("="*80)

    # Get top time-consuming functions
    p.sort_stats(SortKey.TIME)
    stats = p.stats

    print("\nBased on profiling, consider optimizing:")
    count = 0
    for func, (cc, nc, tt, ct, callers) in sorted(
        stats.items(),
        key=lambda x: x[1][2],  # Sort by tottime
        reverse=True
    ):
        if count >= 10:
            break
        filename, line, func_name = func
        if 'individual' in filename or 'coupler' in filename or \
           'biology' in filename:
            count += 1
            print(f"{count:2d}. {func_name:40s} ({tt:.3f}s, {nc:,} calls)")
            if 'individual' in filename:
                print(f"    → Consider: Cythonize this method")
            if nc > 10000:
                print(f"    → High call count: Inline or vectorize")

    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("1. Parallelize: Use coupler_parallel.py (4-6x speedup)")
    print("2. Cythonize: Start with top 5 functions above (5-10x speedup)")
    print("3. Vectorize: Replace loops with NumPy operations where possible")
    print("4. Profile again: Measure improvement after each optimization")
    print("="*80)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = 'profile.stats'

    view_profile(filename)
