#!/usr/bin/env python3
"""
Usage Examples for run_all_sim_configs.py

This script allows you to run different combinations of pipe flow simulations with various
boundary conditions, collision operators, and spatial profiles.

=== Basic Usage ===

1. Run ALL possible combinations (2 BC × 2 CO × 2 SP = 8 total):
   python run_all_sim_configs.py --pipe-only

2. Dry run to see what would be executed:
   python run_all_sim_configs.py --pipe-only --dry-run

3. Run with long pipe settings:
   python run_all_sim_configs.py --pipe-only --long-pipe

=== Spatial Profile Specific Usage ===

4. Run only poiseuille profile simulations:
   python run_all_sim_configs.py --pipe-only --spatial-profile poiseuille

5. Run only blunted paraboloid (blood flow) simulations:
   python run_all_sim_configs.py --pipe-only --spatial-profile blunted_paraboloid

=== Combination Filtering ===

6. Run only time-dependent boundary conditions with all other combinations:
   python run_all_sim_configs.py --pipe-only --boundary-condition time-dependent

7. Run only non-newtonian collision operators with all other combinations:
   python run_all_sim_configs.py --pipe-only --collision-operator non-newtonian

8. Run specific combination:
   python run_all_sim_configs.py --pipe-only --boundary-condition time-dependent --collision-operator non-newtonian --spatial-profile blunted_paraboloid

=== Aneurysm Simulations ===

9. Run only aneurysm simulations:
   python run_all_sim_configs.py --aneurysm-only

10. Run both pipe and aneurysm simulations:
    python run_all_sim_configs.py

=== Output Directory Structure ===

Results will be saved with the following naming convention:
- zh = standard Zou-He boundary condition
- tdzh = time-dependent Zou-He boundary condition
- bgk = standard BGK collision operator
- nnbgk = non-newtonian BGK collision operator
- poiseuille = Poiseuille spatial profile
- blunted = blunted paraboloid spatial profile

Examples:
- zh_bgk_poiseuille/ = Standard BC + Standard CO + Poiseuille profile
- tdzh_nnbgk_blunted/ = Time-dependent BC + Non-Newtonian CO + Blunted paraboloid profile

=== Default Spatial Profile Parameters ===

Poiseuille: No additional parameters
Blunted Paraboloid: power_law_exponent = 1.7 (typical for blood flow)

=== Complete Example Commands ===

# Test all combinations with dry run
python run_all_sim_configs.py --pipe-only --dry-run

# Run all blood flow (blunted paraboloid) simulations
python run_all_sim_configs.py --pipe-only --spatial-profile blunted_paraboloid

# Run time-dependent + non-newtonian + blunted (best for blood flow)
python run_all_sim_configs.py --pipe-only --boundary-condition time-dependent --collision-operator non-newtonian --spatial-profile blunted_paraboloid

# Run long pipe simulations with all spatial profiles
python run_all_sim_configs.py --pipe-only --long-pipe
"""

if __name__ == "__main__":
    print(__doc__)
