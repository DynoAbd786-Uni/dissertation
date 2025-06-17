#!/usr/bin/env python3
"""
Comprehensive Performance Testing Script for Aneurysm LBM Simulation

This script runs performance tests for both JAX and WARP backends with proper isolation
to ensure accurate and uncontaminated results. Each backend test runs in a separate 
subprocess to avoid memory/thermal interference.

Usage:
    python run_all_performance_tests.py [options]

Features:
- Separate subprocess execution for each backend
- Thermal cool-down periods between tests
- GPU memory cleanup between tests
- Comprehensive results comparison
- System resource monitoring
"""

import subprocess
import time
import os
import json
import argparse
import sys
from datetime import datetime
from pathlib import Path


def get_gpu_temperature():
    """Get GPU temperature using nvidia-ml-py if available"""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        return temp
    except:
        return None


def wait_for_gpu_cooldown(target_temp=50, max_wait=120):
    """Wait for GPU to cool down to target temperature"""
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        temp = get_gpu_temperature()
        if temp is None:
            print("⚠️  GPU temperature monitoring not available, waiting 30s...")
            time.sleep(30)
            return
        
        print(f"🌡️  GPU Temperature: {temp}°C", end="")
        if temp <= target_temp:
            print(" ✅ Cool enough to proceed")
            return
        else:
            print(f" (waiting for ≤{target_temp}°C)")
            time.sleep(5)
    
    print(f"⚠️  Timeout waiting for GPU cooldown, proceeding anyway...")


def run_backend_test(backend, duration_seconds, warmup_seconds, verbose=False):
    """Run performance test for a specific backend in separate subprocess"""
    
    print(f"\n{'='*60}")
    print(f"🚀 STARTING {backend} PERFORMANCE TEST")
    print(f"{'='*60}")
    print(f"Duration: {duration_seconds}s analysis + {warmup_seconds}s warmup")
    print(f"Process isolation: ✅ Separate subprocess")
    
    # Build command
    cmd = [
        sys.executable, 
        "performance_testing_aneurysm_model.py",
        "--backend", backend,
        "--duration-seconds", str(duration_seconds),
        "--warmup-seconds", str(warmup_seconds)
    ]
    
    if verbose:
        print(f"Command: {' '.join(cmd)}")
    
    # Record start time and temperature
    start_time = time.time()
    start_temp = get_gpu_temperature()
    if start_temp:
        print(f"🌡️  Starting GPU temperature: {start_temp}°C")
    
    try:
        # Run the test in separate subprocess
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
            # No timeout - let it run as long as needed
        )
        
        end_time = time.time()
        duration = end_time - start_time
        end_temp = get_gpu_temperature()
        
        if result.returncode == 0:
            print(f"✅ {backend} test completed successfully in {duration:.1f}s")
            if end_temp:
                print(f"🌡️  Ending GPU temperature: {end_temp}°C")
            
            if verbose:
                print("--- STDOUT ---")
                print(result.stdout)
            
            # Parse results from JSON file
            results_file = Path(f"../results/performance_tests/aneurysm_performance_{backend.lower()}/parameters/performance_test_results_{backend.lower()}.json")
            
            if results_file.exists():
                with open(results_file, 'r') as f:
                    test_results = json.load(f)
                return {
                    "success": True,
                    "backend": backend,
                    "duration": duration,
                    "start_temp": start_temp,
                    "end_temp": end_temp,
                    "results": test_results,
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
            else:
                print(f"⚠️  Results file not found: {results_file}")
                return {
                    "success": False,
                    "backend": backend,
                    "error": "Results file not found",
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
        else:
            print(f"❌ {backend} test failed with return code {result.returncode}")
            print("--- STDERR ---")
            print(result.stderr)
            return {
                "success": False,
                "backend": backend,
                "error": f"Process failed with code {result.returncode}",
                "stdout": result.stdout,
                "stderr": result.stderr
            }
            
    except Exception as e:
        print(f"❌ {backend} test failed with exception: {e}")
        return {
            "success": False,
            "backend": backend,
            "error": str(e),
            "stdout": "",
            "stderr": ""
        }


def display_comparison_results(jax_results, warp_results):
    """Display comprehensive comparison of results"""
    
    print(f"\n{'='*80}")
    print("📊 PERFORMANCE COMPARISON RESULTS")
    print(f"{'='*80}")
    
    # Results table
    print(f"\n{'Backend':<10} {'Status':<10} {'Avg MLUPS':<12} {'Peak MLUPS':<12} {'Grid Size':<12} {'Device':<15}")
    print("-" * 80)
    
    for results in [jax_results, warp_results]:
        if results["success"]:
            perf = results["results"]["performance"]["performance_metrics"]
            grid = results["results"]["numerical"]["grid_shape"]
            backend_name = results["backend"]
            device = "GPU" if backend_name == "WARP" else "CPU"
            
            print(f"{backend_name:<10} {'✅ OK':<10} {perf['average_mlups']:<12.1f} {perf['peak_mlups']:<12.1f} {grid[0]}x{grid[1]:<7} {device:<15}")
        else:
            print(f"{results['backend']:<10} {'❌ FAIL':<10} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<15}")
    
    # Performance comparison
    if jax_results["success"] and warp_results["success"]:
        jax_mlups = jax_results["results"]["performance"]["performance_metrics"]["average_mlups"]
        warp_mlups = warp_results["results"]["performance"]["performance_metrics"]["average_mlups"]
        
        if warp_mlups > jax_mlups:
            speedup = warp_mlups / jax_mlups
            print(f"\n🚀 GPU Performance Advantage:")
            print(f"   WARP (GPU) is {speedup:.1f}x faster than JAX (CPU)")
            print(f"   Performance gain: {((speedup - 1) * 100):.1f}%")
        else:
            print(f"\n⚠️  Unexpected result: JAX outperformed WARP")
    
    # Temperature analysis
    print(f"\n🌡️  Thermal Analysis:")
    for results in [jax_results, warp_results]:
        if results["success"] and results.get("start_temp") and results.get("end_temp"):
            temp_delta = results["end_temp"] - results["start_temp"]
            print(f"   {results['backend']}: {results['start_temp']}°C → {results['end_temp']}°C (Δ{temp_delta:+.1f}°C)")
        else:
            print(f"   {results['backend']}: Temperature monitoring unavailable")


def save_comprehensive_results(jax_results, warp_results, output_file):
    """Save comprehensive test results to JSON file"""
    
    comprehensive_results = {
        "test_metadata": {
            "timestamp": datetime.now().isoformat(),
            "test_type": "comprehensive_backend_comparison",
            "isolation_method": "separate_subprocess",
            "hostname": os.uname().nodename
        },
        "jax_results": jax_results,
        "warp_results": warp_results,
        "comparison": {}
    }
    
    # Add comparison metrics if both tests succeeded
    if jax_results["success"] and warp_results["success"]:
        jax_mlups = jax_results["results"]["performance"]["performance_metrics"]["average_mlups"]
        warp_mlups = warp_results["results"]["performance"]["performance_metrics"]["average_mlups"]
        
        comprehensive_results["comparison"] = {
            "gpu_speedup_factor": warp_mlups / jax_mlups if jax_mlups > 0 else None,
            "performance_winner": "WARP" if warp_mlups > jax_mlups else "JAX",
            "jax_mlups": jax_mlups,
            "warp_mlups": warp_mlups,
            "performance_gap_percent": ((warp_mlups - jax_mlups) / jax_mlups * 100) if jax_mlups > 0 else None
        }
    
    # Save to file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(comprehensive_results, f, indent=2)
    
    print(f"\n💾 Comprehensive results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Run comprehensive performance tests for both JAX and WARP backends')
    parser.add_argument('--duration-seconds', type=float, default=1.0,
                        help='Test duration for each backend (default: 1.0s)')
    parser.add_argument('--warmup-seconds', type=float, default=2.0,
                        help='Warmup duration for each backend (default: 2.0s)')
    parser.add_argument('--cooldown-seconds', type=int, default=30,
                        help='Cooldown period between tests (default: 30s)')
    parser.add_argument('--skip-cooldown', action='store_true',
                        help='Skip thermal cooldown periods (faster but less accurate)')
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed output from each test')
    parser.add_argument('--jax-only', action='store_true',
                        help='Run only JAX backend test')
    parser.add_argument('--warp-only', action='store_true',
                        help='Run only WARP backend test')
    
    args = parser.parse_args()
    
    print("🔬 COMPREHENSIVE LBM PERFORMANCE TESTING")
    print("=" * 50)
    print(f"Test duration: {args.duration_seconds}s + {args.warmup_seconds}s warmup")
    print(f"Cooldown period: {args.cooldown_seconds}s")
    print(f"Isolation method: Separate subprocesses")
    
    results = {}
    
    # Test order: JAX first (lighter load), then WARP (heavier load)
    test_order = []
    if not args.warp_only:
        test_order.append("JAX")
    if not args.jax_only:
        test_order.append("WARP")
    
    for i, backend in enumerate(test_order):
        # Add cooldown between tests (except for first test)
        if i > 0 and not args.skip_cooldown:
            print(f"\n⏳ Cooling down for {args.cooldown_seconds}s between tests...")
            if backend == "WARP":
                # Extra care for GPU test - wait for thermal cooldown
                wait_for_gpu_cooldown(target_temp=50, max_wait=args.cooldown_seconds)
            else:
                time.sleep(args.cooldown_seconds)
        
        # Run the test
        results[backend.lower()] = run_backend_test(
            backend, 
            args.duration_seconds, 
            args.warmup_seconds,
            args.verbose
        )
    
    # Display results
    jax_results = results.get("jax", {"success": False, "backend": "JAX", "error": "Test skipped"})
    warp_results = results.get("warp", {"success": False, "backend": "WARP", "error": "Test skipped"})
    
    display_comparison_results(jax_results, warp_results)
    
    # Save comprehensive results
    output_file = "../results/performance_tests/comprehensive_performance_comparison.json"
    save_comprehensive_results(jax_results, warp_results, output_file)
    
    # Final summary
    successful_tests = sum(1 for r in results.values() if r["success"])
    total_tests = len(results)
    
    print(f"\n{'='*60}")
    print(f"📋 TEST SUMMARY: {successful_tests}/{total_tests} tests completed successfully")
    print(f"{'='*60}")
    
    if successful_tests == total_tests:
        print("🎉 All tests completed successfully!")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    exit(main())
