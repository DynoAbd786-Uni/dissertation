import jax
import jax.numpy as jnp
import numpy as np
import time
import os
from typing import List, Dict, Any

def detect_jax_devices() -> Dict[str, Any]:
    """
    Detect and analyze all available JAX devices (CPU, GPU, TPU).
    
    Returns:
        Dictionary containing device information and capabilities
    """
    print("="*60)
    print("JAX DEVICE DETECTION AND ANALYSIS")
    print("="*60)
    
    # Basic JAX information
    print(f"JAX Version: {jax.__version__}")
    print(f"JAX NumPy Version: {jnp.__version__}")
    print(f"NumPy Version: {np.__version__}")
    
    # Get all available devices
    devices = jax.devices()
    print(f"\nTotal Available Devices: {len(devices)}")
    
    device_info = {
        'jax_version': jax.__version__,
        'total_devices': len(devices),
        'devices': [],
        'cpu_devices': [],
        'gpu_devices': [],
        'tpu_devices': [],
        'default_backend': jax.default_backend(),
        'supported_platforms': []
    }
    
    # Analyze each device
    print("\nDETAILED DEVICE INFORMATION:")
    print("-" * 60)
    
    for i, device in enumerate(devices):
        print(f"\nDevice {i}: {device}")
        print(f"  Platform: {device.platform}")
        print(f"  Device Kind: {device.device_kind}")
        print(f"  ID: {device.id}")
        print(f"  Process Index: {device.process_index}")
        
        device_details = {
            'index': i,
            'device': str(device),
            'platform': device.platform,
            'device_kind': device.device_kind,
            'id': device.id,
            'process_index': device.process_index,
            'memory_stats': None,
            'compute_capability': None
        }
        
        # Get memory information if available
        try:
            if hasattr(device, 'memory_stats'):
                memory_stats = device.memory_stats()
                print(f"  Memory Stats: {memory_stats}")
                device_details['memory_stats'] = memory_stats
        except Exception as e:
            print(f"  Memory Stats: Not available ({e})")
        
        # Get additional GPU information
        if device.platform == 'gpu':
            try:
                # Try to get CUDA information
                import subprocess
                result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,compute_cap', 
                                       '--format=csv,noheader,nounits'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    gpu_lines = result.stdout.strip().split('\n')
                    if i < len(gpu_lines):
                        gpu_info = gpu_lines[i].split(', ')
                        if len(gpu_info) >= 3:
                            print(f"  GPU Name: {gpu_info[0]}")
                            print(f"  Total Memory: {gpu_info[1]} MB")
                            print(f"  Compute Capability: {gpu_info[2]}")
                            device_details['gpu_name'] = gpu_info[0]
                            device_details['total_memory_mb'] = gpu_info[1]
                            device_details['compute_capability'] = gpu_info[2]
            except Exception as e:
                print(f"  GPU Info: Could not retrieve ({e})")
        
        # Categorize devices
        if device.platform == 'cpu':
            device_info['cpu_devices'].append(device_details)
        elif device.platform == 'gpu':
            device_info['gpu_devices'].append(device_details)
        elif device.platform == 'tpu':
            device_info['tpu_devices'].append(device_details)
        
        device_info['devices'].append(device_details)
    
    # Get supported platforms
    try:
        from jax.lib import xla_bridge
        supported_platforms = xla_bridge.get_backend().platform_version
        device_info['supported_platforms'] = supported_platforms
        print(f"\nSupported Platforms: {supported_platforms}")
    except Exception as e:
        print(f"\nSupported Platforms: Could not retrieve ({e})")
    
    # Display platform summary
    print("\nPLATFORM SUMMARY:")
    print("-" * 30)
    print(f"Default Backend: {device_info['default_backend']}")
    print(f"CPU Devices: {len(device_info['cpu_devices'])}")
    print(f"GPU Devices: {len(device_info['gpu_devices'])}")
    print(f"TPU Devices: {len(device_info['tpu_devices'])}")
    
    return device_info

def test_device_performance(device_info: Dict[str, Any]) -> Dict[str, float]:
    """
    Test basic performance on each available device.
    
    Args:
        device_info: Device information from detect_jax_devices()
    
    Returns:
        Dictionary with performance results for each device
    """
    print("\n" + "="*60)
    print("DEVICE PERFORMANCE TESTING")
    print("="*60)
    
    performance_results = {}
    
    # Test matrix multiplication on each platform
    test_size = 2048
    num_iterations = 3
    
    for platform in ['cpu', 'gpu', 'tpu']:
        platform_devices = device_info[f'{platform}_devices']
        if not platform_devices:
            continue
            
        print(f"\nTesting {platform.upper()} Performance:")
        print("-" * 40)
        
        try:
            # Configure JAX to use specific platform
            with jax.default_device(jax.devices(platform)[0]):
                # Create test data
                key = jax.random.PRNGKey(42)
                A = jax.random.normal(key, (test_size, test_size), dtype=jnp.float32)
                B = jax.random.normal(key, (test_size, test_size), dtype=jnp.float32)
                
                # JIT compile the operation
                @jax.jit
                def matrix_multiply(A, B):
                    return jnp.matmul(A, B)
                
                # Warm up
                _ = matrix_multiply(A, B).block_until_ready()
                
                # Time the operation
                times = []
                for i in range(num_iterations):
                    start_time = time.time()
                    result = matrix_multiply(A, B)
                    result.block_until_ready()  # Ensure completion
                    end_time = time.time()
                    times.append(end_time - start_time)
                
                avg_time = np.mean(times)
                std_time = np.std(times)
                
                # Calculate FLOPS
                flops = 2 * test_size ** 3  # Matrix multiplication FLOPs
                gflops = flops / avg_time / 1e9
                
                print(f"  Matrix Size: {test_size}x{test_size}")
                print(f"  Average Time: {avg_time:.4f} ± {std_time:.4f} seconds")
                print(f"  Performance: {gflops:.2f} GFLOPS")
                
                performance_results[platform] = {
                    'avg_time': avg_time,
                    'std_time': std_time,
                    'gflops': gflops,
                    'matrix_size': test_size
                }
                
        except Exception as e:
            print(f"  Error testing {platform}: {e}")
            performance_results[platform] = {'error': str(e)}
    
    return performance_results

def check_jax_configuration():
    """
    Check current JAX configuration and suggest optimizations.
    """
    print("\n" + "="*60)
    print("JAX CONFIGURATION ANALYSIS")
    print("="*60)
    
    # Check important configuration flags
    configs_to_check = [
        'jax_enable_x64',
        'jax_platform_name',
        'jax_compilation_cache_dir',
        'jax_enable_compilation_cache'
    ]
    
    print("Current JAX Configuration:")
    print("-" * 30)
    
    for config_name in configs_to_check:
        try:
            value = jax.config.read(config_name)
            print(f"  {config_name}: {value}")
        except Exception:
            print(f"  {config_name}: Not set or not available")
    
    # Check environment variables
    print("\nRelevant Environment Variables:")
    print("-" * 35)
    
    env_vars = [
        'XLA_FLAGS',
        'XLA_PYTHON_CLIENT_MEM_FRACTION',
        'XLA_PYTHON_CLIENT_PREALLOCATE',
        'CUDA_VISIBLE_DEVICES',
        'JAX_TRACEBACK_FILTERING'
    ]
    
    for env_var in env_vars:
        value = os.environ.get(env_var, 'Not set')
        print(f"  {env_var}: {value}")
    
    # Provide optimization suggestions
    print("\nOPTIMIZATION SUGGESTIONS:")
    print("-" * 30)
    
    suggestions = []
    
    # Check if using float64
    if jax.config.read('jax_enable_x64'):
        suggestions.append("Consider disabling x64 for better GPU performance: jax.config.update('jax_enable_x64', False)")
    
    # Check if GPU is available but not default
    gpu_devices = [d for d in jax.devices() if d.platform == 'gpu']
    if gpu_devices and jax.default_backend() != 'gpu':
        suggestions.append("GPU detected but not default. Consider: jax.config.update('jax_platform_name', 'gpu')")
    
    # Check compilation cache
    if not jax.config.read('jax_compilation_cache_dir'):
        suggestions.append("Enable compilation caching: jax.config.update('jax_compilation_cache_dir', '/tmp/jax_cache')")
    
    if suggestions:
        for i, suggestion in enumerate(suggestions, 1):
            print(f"  {i}. {suggestion}")
    else:
        print("  Configuration looks good!")

def main():
    """
    Main function to run complete JAX device detection and analysis.
    """
    try:
        # Detect devices
        device_info = detect_jax_devices()
        
        # Test performance
        performance_results = test_device_performance(device_info)
        
        # Check configuration
        check_jax_configuration()
        
        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        
        print(f"JAX {device_info['jax_version']} detected {device_info['total_devices']} device(s)")
        print(f"Default backend: {device_info['default_backend']}")
        
        if device_info['gpu_devices']:
            print(f"✓ GPU acceleration available ({len(device_info['gpu_devices'])} GPU(s))")
        else:
            print("⚠ No GPU devices detected")
        
        if device_info['tpu_devices']:
            print(f"✓ TPU acceleration available ({len(device_info['tpu_devices'])} TPU(s))")
        
        # Performance summary
        if performance_results:
            print("\nPerformance Summary:")
            for platform, results in performance_results.items():
                if 'gflops' in results:
                    print(f"  {platform.upper()}: {results['gflops']:.2f} GFLOPS")
        
        return device_info, performance_results
        
    except Exception as e:
        print(f"Error during device detection: {e}")
        return None, None

if __name__ == "__main__":
    device_info, performance_results = main()