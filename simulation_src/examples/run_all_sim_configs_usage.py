#!/usr/bin/env python3
"""
Comprehensive Usage Guide for run_all_sim_configs.py

This script provides complete documentation for running the advanced CFD blood flow simulation 
batch execution system with 18 total configurations (15 pipe + 3 aneurysm).

=== 🎯 NEW DEFAULT BEHAVIOR (2025 Update) ===

The batch system now runs BOTH standard and long pipe configurations by default:
- 12 standard pipe configurations (2 BC × 2 CO × 3 SP) 
- 3 long pipe configurations (1 BC × 1 CO × 3 SP)
- 3 aneurysm configurations (1 BC × 1 CO × 3 SP)
- Total: 18 configurations in complete study

=== ⚡ QUICK START COMMANDS ===

1. Complete simulation suite (18 configurations, ~60-70 minutes):
   python run_all_sim_configs.py

2. Validate all configurations before running:
   python run_all_sim_configs.py --dry-run

3. Pipe simulations only (15 configurations):
   python run_all_sim_configs.py --pipe-only

4. Aneurysm simulations only (3 configurations):
   python run_all_sim_configs.py --aneurysm-only

=== 🔧 PIPE TYPE CONFIGURATIONS ===

5. Standard pipe only (12 configurations, detailed analysis):
   python run_all_sim_configs.py --pipe-only --standard-pipe-only

6. Long pipe only (3 configurations, high performance):
   python run_all_sim_configs.py --pipe-only --long-pipe

7. Performance comparison (run both standard and long):
   python run_all_sim_configs.py --pipe-only

=== 🌊 SPATIAL PROFILE STUDIES ===

8. All uniform profile simulations (validation):
   python run_all_sim_configs.py --spatial-profile uniform

9. All Poiseuille profile simulations (laminar flow):
   python run_all_sim_configs.py --spatial-profile poiseuille

10. All blunted paraboloid simulations (blood flow, recommended):
    python run_all_sim_configs.py --spatial-profile blunted_paraboloid

11. Aneurysm with specific spatial profile:
    python run_all_sim_configs.py --aneurysm-only --aneurysm-spatial-profile blunted_paraboloid

=== 🔬 BOUNDARY CONDITION STUDIES ===

12. Standard Zou-He boundary conditions:
    python run_all_sim_configs.py --boundary-condition standard

13. Time-dependent boundary conditions (pulsatile flow):
    python run_all_sim_configs.py --boundary-condition time-dependent

14. Time-dependent with non-Newtonian (recommended for blood flow):
    python run_all_sim_configs.py --boundary-condition time-dependent --collision-operator non-newtonian

=== ⚙️ COLLISION OPERATOR STUDIES ===

15. Standard BGK collision operator:
    python run_all_sim_configs.py --collision-operator standard

16. Non-Newtonian BGK (Carreau-Yasuda blood model):
    python run_all_sim_configs.py --collision-operator non-newtonian

17. Non-Newtonian with blunted profile (optimal blood flow):
    python run_all_sim_configs.py --collision-operator non-newtonian --spatial-profile blunted_paraboloid

=== 🎯 FOCUSED RESEARCH CONFIGURATIONS ===

18. Blood flow validation study:
    python run_all_sim_configs.py --pipe-only --boundary-condition time-dependent --collision-operator non-newtonian --spatial-profile blunted_paraboloid

19. Performance benchmarking:
    python run_all_sim_configs.py --pipe-only --long-pipe

20. Fluid mechanics validation:
    python run_all_sim_configs.py --pipe-only --boundary-condition standard --collision-operator standard --spatial-profile poiseuille

21. Complete aneurysm blood flow study:
    python run_all_sim_configs.py --aneurysm-only

=== 📊 EXPECTED PERFORMANCE METRICS ===

Standard Pipe (751×330 nodes): 3,000-3,700 MLUPS, 2-3 min each
Long Pipe (10,001×86 nodes): 8,000-13,000 MLUPS, 3-4 min each  
Aneurysm (1751×530 nodes): 12,000-16,000 MLUPS, 4-5 min each

Complete batch run: ~60-70 minutes total on modern GPU hardware

=== 📁 OUTPUT DIRECTORY STRUCTURE ===

Results organized with comprehensive naming: {bc}_{co}_{sp}_{pipe_type}

Abbreviations:
- BC: zh (standard Zou-He), tdzh (time-dependent Zou-He)
- CO: bgk (standard BGK), nnbgk (non-Newtonian BGK)  
- SP: uniform, poiseuille, blunted
- PIPE: standard, long

Examples:
results/pipe_flow/
├── zh_bgk_uniform_standard/         # Standard BC + Standard CO + Uniform + Standard Pipe
├── zh_nnbgk_blunted_standard/       # Standard BC + Non-Newtonian CO + Blunted + Standard Pipe
├── tdzh_nnbgk_poiseuille_standard/  # Time-dependent BC + Non-Newtonian CO + Poiseuille + Standard Pipe
├── zh_nnbgk_uniform_long/           # Standard BC + Non-Newtonian CO + Uniform + Long Pipe
└── zh_nnbgk_blunted_long/           # Standard BC + Non-Newtonian CO + Blunted + Long Pipe

results/aneurysm_flow/
├── CCA_simulation_results_nnbgk_tdzh_uniform/      # Aneurysm + Uniform
├── CCA_simulation_results_nnbgk_tdzh_poiseuille/   # Aneurysm + Poiseuille  
└── CCA_simulation_results_nnbgk_tdzh_blunted/      # Aneurysm + Blunted (default)

results/logs/
├── pipe_zh_bgk_uniform_standard_2025-06-14_12-00-30.log
├── pipe_zh_nnbgk_blunted_long_2025-06-14_12-15-45.log
└── aneurysm_tdzh_nnbgk_blunted_2025-06-14_12-21-30.log

=== 🔬 CONFIGURATION MATRIX BREAKDOWN ===

Standard Pipe Configurations (12 total):
2 BC (zh, tdzh) × 2 CO (bgk, nnbgk) × 3 SP (uniform, poiseuille, blunted) = 12

Long Pipe Configurations (3 total): 
1 BC (zh) × 1 CO (nnbgk) × 3 SP (uniform, poiseuille, blunted) = 3

Aneurysm Configurations (3 total):
1 BC (tdzh) × 1 CO (nnbgk) × 3 SP (uniform, poiseuille, blunted) = 3

Total System Configurations: 18

=== ⚙️ SPATIAL PROFILE PARAMETERS ===

1. Uniform Profile:
   - Description: Flat velocity distribution (testing/validation)
   - Parameters: None
   - Use case: Numerical validation and comparison baseline

2. Poiseuille Profile:
   - Description: Parabolic velocity distribution (laminar flow)
   - Parameters: None (analytical solution)
   - Use case: Classical fluid mechanics validation

3. Blunted Paraboloid Profile:
   - Description: Power-law profile optimized for blood flow
   - Parameters: power_law_exponent = 1.7 (typical for blood)
   - Use case: Realistic cardiovascular modeling (recommended)

=== 🚀 VALIDATION AND TESTING ===

Always start with validation:
# Test configuration matrix
python run_all_sim_configs.py --dry-run

# Quick performance test  
python run_all_sim_configs.py --pipe-only --boundary-condition standard --collision-operator standard --spatial-profile uniform --dry-run

# Single high-performance test
python run_all_sim_configs.py --pipe-only --long-pipe --spatial-profile uniform

=== 🔧 ERROR HANDLING AND TROUBLESHOOTING ===

Flag conflicts automatically detected:
- Cannot use --long-pipe with --standard-pipe-only
- Cannot use --pipe-only with --aneurysm-only  
- Invalid boundary condition, collision operator, or spatial profile combinations

Common issues:
- Low MLUPS: Ensure GPU is available and drivers are current
- Memory errors: Use smaller domains or CPU backend
- Missing results: Check log files in results/logs/

=== 📈 RECOMMENDED RESEARCH WORKFLOWS ===

1. Performance Characterization:
   python run_all_sim_configs.py --pipe-only --dry-run
   python run_all_sim_configs.py --pipe-only --long-pipe
   python run_all_sim_configs.py --pipe-only --standard-pipe-only

2. Blood Flow Studies:
   python run_all_sim_configs.py --boundary-condition time-dependent --collision-operator non-newtonian --spatial-profile blunted_paraboloid
   python run_all_sim_configs.py --aneurysm-only

3. Spatial Profile Comparison:
   python run_all_sim_configs.py --pipe-only --boundary-condition time-dependent --collision-operator non-newtonian

4. Complete Parameter Study:
   python run_all_sim_configs.py

=== 📚 ADDITIONAL DOCUMENTATION ===

See also:
- README.md: Complete framework documentation
- SPATIAL_PROFILES_SUMMARY.md: Detailed spatial profiles guide
- MLUPS_FIXES_SUMMARY.md: Performance optimization details
- UNIFORM_SPATIAL_PROFILE_IMPLEMENTATION.md: Uniform profile specifics

For interactive analysis:
- visualisation_src/vtk_visualization.ipynb: Interactive field analysis
- ParaView: Professional visualization with generated VTK files

=== 🎯 2025 FEATURE HIGHLIGHTS ===

✅ Default multi-configuration execution (18 total configurations)
✅ Intelligent pipe type handling (standard vs long domains)  
✅ Comprehensive spatial profiles (uniform, Poiseuille, blunted paraboloid)
✅ Enhanced naming system prevents result overwrites
✅ Performance scaling analysis (domain-size dependent MLUPS)
✅ Robust error handling and validation systems
✅ Professional research-grade documentation
✅ Complete reproducibility with JSON parameter files

This batch execution system provides a comprehensive platform for systematic
CFD blood flow studies with professional documentation and reproducibility.
"""

if __name__ == "__main__":
    print(__doc__)
