import jax
import jax.numpy as jnp
import time
import numpy as np
from jax import jit, random
import os

# Add JAX performance optimizations with valid flags only
try:
    # Use only stable and widely supported XLA flags
    os.environ['XLA_FLAGS'] = '--xla_gpu_enable_fast_min_max=true --xla_gpu_enable_command_buffer='
    jax.config.update("jax_enable_x64", False)  # Use float32 for better GPU performance
    jax.config.update("jax_default_matmul_precision", "high")  # Better precision for matrix ops
    
    # Only set cache if directory is writable
    cache_dir = "/tmp/jax_cache"
    if os.access("/tmp", os.W_OK):
        jax.config.update("jax_compilation_cache_dir", cache_dir)
except Exception as e:
    print(f"Warning: Could not set some JAX optimizations: {e}")
    # Fallback to basic configuration
    jax.config.update("jax_enable_x64", False)

# Add WARP imports
try:
    import warp as wp
    WARP_AVAILABLE = True
    wp.init()
    print("WARP version:", wp.__version__)
    print("WARP CUDA devices:", wp.get_cuda_device_count())
except ImportError:
    WARP_AVAILABLE = False
    print("WARP not available - skipping WARP benchmarks")

print("JAX version:", jax.__version__)
print("Available devices:", jax.devices())
print("Default device:", jax.default_backend())

@jit
def large_matrix_operations_optimized(A, B, C):
    """Optimized JIT-compiled matrix operations"""
    # Chain of operations that stress GPU - all operations stay on device
    result = jnp.matmul(A, B)  # Matrix multiplication (remove precision arg for compatibility)
    result = jnp.matmul(result, C)  # Another matrix multiplication
    result = jnp.sin(result) + jnp.cos(result)  # Element-wise operations
    result = jnp.exp(result / 10.0)  # More element-wise operations
    result = jnp.matmul(result, A.T)  # Transpose and multiply
    result = jnp.linalg.norm(result)  # Final reduction
    
    return result

def large_matrix_operations(key, size):
    """Perform intensive matrix operations"""
    # Generate large random matrices directly on device
    key1, key2, key3 = random.split(key, 3)
    
    A = random.normal(key1, (size, size), dtype=jnp.float32)
    B = random.normal(key2, (size, size), dtype=jnp.float32)
    C = random.normal(key3, (size, size), dtype=jnp.float32)
    
    # Use optimized JIT-compiled function
    result = large_matrix_operations_optimized(A, B, C)
    
    return result

@jit
def fluid_step_optimized(velocity_x, velocity_y, density):
    """JIT-compiled single fluid dynamics step"""
    # Streaming step (simplified) - vectorized operations
    velocity_x_new = jnp.roll(velocity_x, 1, axis=0) + jnp.roll(velocity_x, -1, axis=0)
    velocity_y_new = jnp.roll(velocity_y, 1, axis=1) + jnp.roll(velocity_y, -1, axis=1)
    
    # Collision step (simplified)
    velocity_x = 0.9 * velocity_x + 0.1 * velocity_x_new
    velocity_y = 0.9 * velocity_y + 0.1 * velocity_y_new
    
    # Update density - use more efficient stencil operations
    density_update = 0.01 * (
        jnp.roll(density, 1, axis=0) - 2*density + jnp.roll(density, -1, axis=0) +
        jnp.roll(density, 1, axis=1) - 2*density + jnp.roll(density, -1, axis=1)
    )
    density = density + density_update
    
    # Boundary conditions - use more efficient indexing
    velocity_x = velocity_x.at[0, :].set(0.1)  # Inlet
    velocity_x = velocity_x.at[-1, :].set(0.0)  # Outlet
    velocity_y = velocity_y.at[:, 0].set(0.0)  # Walls
    velocity_y = velocity_y.at[:, -1].set(0.0)
    
    return velocity_x, velocity_y, density

@jit
def fluid_simulation_batch(velocity_x, velocity_y, density, num_steps):
    """JIT-compiled batch fluid simulation using fori_loop for efficiency"""
    def body_fun(i, carry):
        vx, vy, d = carry
        return fluid_step_optimized(vx, vy, d)
    
    # Use fori_loop instead of scan to avoid concretization issues
    velocity_x, velocity_y, density = jax.lax.fori_loop(
        0, num_steps, body_fun, (velocity_x, velocity_y, density)
    )
    
    return velocity_x, velocity_y, density

def fluid_dynamics_simulation(key, grid_size, num_steps):
    """Simulate a simple fluid dynamics computation similar to LBM"""
    # Initialize velocity and density fields
    key1, key2 = random.split(key, 2)
    
    velocity_x = random.normal(key1, (grid_size, grid_size), dtype=jnp.float32) * 0.1
    velocity_y = random.normal(key2, (grid_size, grid_size), dtype=jnp.float32) * 0.1
    density = jnp.ones((grid_size, grid_size), dtype=jnp.float32)
    
    # Use JIT-compiled batch simulation with static num_steps
    velocity_x, velocity_y, density = fluid_simulation_batch(velocity_x, velocity_y, density, num_steps)
    
    return jnp.mean(velocity_x**2 + velocity_y**2), jnp.mean(density)

# Alternative: Create specialized functions for different step counts
@jit
def fluid_simulation_warmup(velocity_x, velocity_y, density):
    """Fixed 20-step simulation for warmup"""
    def body_fun(i, carry):
        vx, vy, d = carry
        return fluid_step_optimized(vx, vy, d)
    
    velocity_x, velocity_y, density = jax.lax.fori_loop(
        0, 20, body_fun, (velocity_x, velocity_y, density)
    )
    
    return velocity_x, velocity_y, density

@jit
def fluid_simulation_main(velocity_x, velocity_y, density):
    """Fixed 200-step simulation for main benchmark"""
    def body_fun(i, carry):
        vx, vy, d = carry
        return fluid_step_optimized(vx, vy, d)
    
    velocity_x, velocity_y, density = jax.lax.fori_loop(
        0, 200, body_fun, (velocity_x, velocity_y, density)
    )
    
    return velocity_x, velocity_y, density

def fluid_dynamics_simulation_optimized(key, grid_size, num_steps):
    """Optimized fluid simulation with fixed step counts"""
    # Initialize velocity and density fields
    key1, key2 = random.split(key, 2)
    
    velocity_x = random.normal(key1, (grid_size, grid_size), dtype=jnp.float32) * 0.1
    velocity_y = random.normal(key2, (grid_size, grid_size), dtype=jnp.float32) * 0.1
    density = jnp.ones((grid_size, grid_size), dtype=jnp.float32)
    
    # Use appropriate specialized function based on step count
    if num_steps == 20:
        velocity_x, velocity_y, density = fluid_simulation_warmup(velocity_x, velocity_y, density)
    elif num_steps == 200:
        velocity_x, velocity_y, density = fluid_simulation_main(velocity_x, velocity_y, density)
    else:
        # Fallback for other step counts - use dynamic compilation
        velocity_x, velocity_y, density = fluid_simulation_batch(velocity_x, velocity_y, density, num_steps)
    
    return jnp.mean(velocity_x**2 + velocity_y**2), jnp.mean(density)

# Optimized WARP kernel functions with better memory patterns
@wp.kernel
def warp_matrix_multiply_kernel_optimized(A: wp.array2d(dtype=float), 
                                        B: wp.array2d(dtype=float),
                                        C: wp.array2d(dtype=float),
                                        size: int):
    i, j = wp.tid()
    if i < size and j < size:
        sum_val = float(0.0)
        # Use standard multiply-add operations
        for k in range(size):
            sum_val += A[i, k] * B[k, j]
        C[i, j] = sum_val

@wp.kernel 
def warp_combined_ops_kernel(A: wp.array2d(dtype=float), 
                           result: wp.array2d(dtype=float),
                           size: int):
    """Combined element-wise operations to reduce kernel launches"""
    i, j = wp.tid()
    if i < size and j < size:
        val = A[i, j]
        # Combine sin + cos + exp operations in single kernel
        trig_result = wp.sin(val) + wp.cos(val)
        result[i, j] = wp.exp(trig_result / 10.0)

@wp.kernel
def warp_transpose_multiply_kernel(A: wp.array2d(dtype=float),
                                 B: wp.array2d(dtype=float), 
                                 C: wp.array2d(dtype=float),
                                 size: int):
    """Optimized transpose and multiply in one kernel"""
    i, j = wp.tid()
    if i < size and j < size:
        sum_val = float(0.0)
        for k in range(size):
            sum_val += B[i, k] * A[j, k]  # A.T[k,j] = A[j,k]
        C[i, j] = sum_val

@wp.kernel
def warp_reduction_kernel(A: wp.array2d(dtype=float),
                        partial_sums: wp.array(dtype=float),
                        size: int):
    """Parallel reduction for norm calculation"""
    tid = wp.tid()
    i = tid // size
    j = tid % size
    
    if i < size and j < size:
        val = A[i, j]
        wp.atomic_add(partial_sums, 0, val * val)

@wp.kernel
def warp_fluid_combined_kernel(velocity_x: wp.array2d(dtype=float),
                             velocity_y: wp.array2d(dtype=float),
                             density: wp.array2d(dtype=float),
                             rows: int,
                             cols: int):
    """Combined streaming and collision in single kernel for better performance"""
    i, j = wp.tid()
    if i < rows and j < cols:
        # Streaming with boundary checks
        i_prev = wp.max(0, i - 1)
        i_next = wp.min(rows - 1, i + 1)
        j_prev = wp.max(0, j - 1) 
        j_next = wp.min(cols - 1, j + 1)
        
        # Stream and collide in one step
        vx_stream = velocity_x[i_prev, j] + velocity_x[i_next, j]
        vy_stream = velocity_y[i, j_prev] + velocity_y[i, j_next]
        
        # Collision step
        velocity_x[i, j] = 0.9 * velocity_x[i, j] + 0.1 * vx_stream
        velocity_y[i, j] = 0.9 * velocity_y[i, j] + 0.1 * vy_stream
        
        # Density update
        density_update = 0.01 * (
            density[i_prev, j] - 2.0 * density[i, j] + density[i_next, j] +
            density[i, j_prev] - 2.0 * density[i, j] + density[i, j_next]
        )
        density[i, j] = density[i, j] + density_update
        
        # Boundary conditions
        if i == 0:
            velocity_x[i, j] = 0.1  # Inlet
        elif i == rows - 1:
            velocity_x[i, j] = 0.0  # Outlet
        if j == 0 or j == cols - 1:
            velocity_y[i, j] = 0.0  # Walls

def warp_large_matrix_operations(size):
    """Optimized WARP matrix operations with reduced memory allocations"""
    # Pre-allocate all arrays to avoid repeated allocations
    A_np = np.random.normal(0, 1, (size, size)).astype(np.float32)
    B_np = np.random.normal(0, 1, (size, size)).astype(np.float32)
    C_np = np.random.normal(0, 1, (size, size)).astype(np.float32)
    
    # Create WARP arrays with better memory layout
    A = wp.array2d(A_np, dtype=float, device='cuda')
    B = wp.array2d(B_np, dtype=float, device='cuda') 
    C = wp.array2d(C_np, dtype=float, device='cuda')
    
    # Reuse temporary arrays instead of creating new ones
    temp1 = wp.zeros((size, size), dtype=float, device='cuda')
    temp2 = wp.zeros((size, size), dtype=float, device='cuda')
    partial_sums = wp.zeros(1, dtype=float, device='cuda')
    
    # Optimize thread block dimensions for better occupancy
    block_dim = min(32, size)  # Use 32x32 blocks for better cache usage
    
    # Step 1: A * B -> temp1
    wp.launch(kernel=warp_matrix_multiply_kernel_optimized,
              dim=(size, size),
              inputs=[A, B, temp1, size],
              device='cuda')
    
    # Step 2: temp1 * C -> temp2  
    wp.launch(kernel=warp_matrix_multiply_kernel_optimized,
              dim=(size, size),
              inputs=[temp1, C, temp2, size],
              device='cuda')
    
    # Step 3: Combined element-wise operations (sin + cos + exp)
    wp.launch(kernel=warp_combined_ops_kernel,
              dim=(size, size),
              inputs=[temp2, temp1, size],
              device='cuda')
    
    # Step 4: Multiply with A.T (optimized transpose)
    wp.launch(kernel=warp_transpose_multiply_kernel,
              dim=(size, size),
              inputs=[A, temp1, temp2, size],
              device='cuda')
    
    # Step 5: Parallel reduction for norm
    wp.launch(kernel=warp_reduction_kernel,
              dim=size * size,
              inputs=[temp2, partial_sums, size],
              device='cuda')
    
    wp.synchronize()
    
    # Get result with minimal CPU transfer
    result_sum = partial_sums.numpy()[0]
    result = np.sqrt(result_sum)
    
    return result

def warp_fluid_dynamics_simulation(grid_size, num_steps):
    """Optimized WARP fluid simulation with reduced kernel launches"""
    # Initialize with better memory patterns
    velocity_x_np = np.random.normal(0, 0.1, (grid_size, grid_size)).astype(np.float32)
    velocity_y_np = np.random.normal(0, 0.1, (grid_size, grid_size)).astype(np.float32)
    density_np = np.ones((grid_size, grid_size), dtype=np.float32)
    
    # Create arrays with optimized layout
    velocity_x = wp.array2d(velocity_x_np, dtype=float, device='cuda')
    velocity_y = wp.array2d(velocity_y_np, dtype=float, device='cuda')
    density = wp.array2d(density_np, dtype=float, device='cuda')
    
    # Use combined kernel to reduce overhead
    for step in range(num_steps):
        wp.launch(kernel=warp_fluid_combined_kernel,
                  dim=(grid_size, grid_size),
                  inputs=[velocity_x, velocity_y, density, grid_size, grid_size],
                  device='cuda')
    
    wp.synchronize()
    
    # Efficient final calculation using GPU reduction
    velocity_x_final = velocity_x.numpy()
    velocity_y_final = velocity_y.numpy() 
    density_final = density.numpy()
    
    avg_velocity = np.mean(velocity_x_final**2 + velocity_y_final**2)
    avg_density = np.mean(density_final)
    
    return avg_velocity, avg_density

@jit
def memory_intensive_ops_optimized(large_array):
    """JIT-compiled memory operations for better performance"""
    # Memory-intensive operations - all vectorized
    copied = large_array + 1.0
    copied = jnp.sin(copied)
    result = jnp.sum(copied)
    return result, copied

def chunked_memory_operations(large_array, chunk_size=10_000_000):
    """Process large arrays in chunks for better memory efficiency"""
    results = []
    for i in range(0, len(large_array), chunk_size):
        chunk = large_array[i:i+chunk_size]
        result, _ = memory_intensive_ops_optimized(chunk)
        results.append(result)
    return jnp.sum(jnp.array(results))

def benchmark_computation():
    """Run intensive computations and measure performance on CPU, JAX GPU, and WARP GPU"""
    key = random.PRNGKey(42)
    
    # Test parameters - significantly increased for stress testing
    matrix_size = 4096        # Increased from 2048
    num_operations = 5        # Increased from 3
    grid_size = 1024          # Increased from 512
    num_steps = 200           # Increased from 50
    array_size = 100_000_000  # Increased from 20_000_000
    num_copies = 10           # Increased from 5
    
    results = {}
    backends = ['CPU', 'JAX_GPU']
    if WARP_AVAILABLE:
        backends.append('WARP_GPU')
    
    for backend_name in backends:
        print("\n" + "="*80)
        print(f"{backend_name} PERFORMANCE BENCHMARK")
        print("="*80)
        
        # Configure backend with error handling
        try:
            if backend_name == 'CPU':
                jax.config.update('jax_platform_name', 'cpu')
            elif backend_name == 'JAX_GPU':
                jax.config.update('jax_platform_name', 'gpu')
                # Try to enable additional GPU optimizations
                try:
                    jax.config.update('jax_enable_compilation_cache', True)
                except Exception:
                    pass  # Ignore if not supported
        except Exception as e:
            print(f"Warning: Backend configuration failed: {e}")
        
        if backend_name != 'WARP_GPU':
            print(f"Platform configured: {jax.default_backend()}")
        else:
            print(f"Platform configured: WARP CUDA")
        
        # Test 1: Large Matrix Operations
        print(f"\nTest 1: Large Matrix Operations ({backend_name})")
        print(f"Matrix size: {matrix_size}x{matrix_size}")
        print(f"Number of operations: {num_operations}")
        
        if backend_name == 'WARP_GPU':
            # WARP warm up with larger size
            _ = warp_large_matrix_operations(1024)
            
            start_time = time.time()
            for i in range(num_operations):
                result = warp_large_matrix_operations(matrix_size)
                print(f"  Operation {i+1}: Result = {result:.6f}")
            end_time = time.time()
        else:
            # JAX warm up with larger size - compile functions
            key_warmup, _ = random.split(key)
            _ = large_matrix_operations(key_warmup, 1024)
            
            # Pre-allocate arrays for better performance
            key_arrays = random.split(key, num_operations + 1)
            
            start_time = time.time()
            for i in range(num_operations):
                result = large_matrix_operations(key_arrays[i], matrix_size)
                result.block_until_ready()
                print(f"  Operation {i+1}: Result = {result:.6f}")
            end_time = time.time()
        
        elapsed_time = end_time - start_time
        
        # Calculate FLOPS
        ops_per_iteration = 6 * (matrix_size ** 3)
        total_ops = ops_per_iteration * num_operations
        flops = total_ops / elapsed_time
        
        print(f"\nMatrix Operations Results ({backend_name}):")
        print(f"  Total time: {elapsed_time:.3f} seconds")
        print(f"  Total operations: {total_ops:.2e}")
        print(f"  Performance: {flops/1e12:.2f} TFLOPS")
        
        results[f'{backend_name}_matrix_time'] = elapsed_time
        results[f'{backend_name}_matrix_tflops'] = flops/1e12
        
        # Test 2: Fluid Dynamics Simulation
        print(f"\n{'-'*60}")
        print(f"Test 2: Fluid Dynamics Simulation (LBM-like) ({backend_name})")
        
        print(f"Grid size: {grid_size}x{grid_size}")
        print(f"Time steps: {num_steps}")
        
        if backend_name == 'WARP_GPU':
            # WARP warm up with larger size
            _ = warp_fluid_dynamics_simulation(256, 20)
            
            start_time = time.time()
            avg_velocity, avg_density = warp_fluid_dynamics_simulation(grid_size, num_steps)
            end_time = time.time()
        else:
            # JAX warm up with optimized function - compile functions
            key_warmup, _ = random.split(key)
            _ = fluid_dynamics_simulation_optimized(key_warmup, 256, 20)
            
            start_time = time.time()
            key, subkey = random.split(key)
            avg_velocity, avg_density = fluid_dynamics_simulation_optimized(subkey, grid_size, num_steps)
            avg_velocity.block_until_ready()
            avg_density.block_until_ready()
            end_time = time.time()
        
        elapsed_time = end_time - start_time
        
        # Calculate MLUPS
        total_lattice_updates = grid_size * grid_size * num_steps
        mlups = total_lattice_updates / elapsed_time / 1e6
        
        print(f"\nFluid Simulation Results ({backend_name}):")
        print(f"  Final average velocity: {avg_velocity:.6f}")
        print(f"  Final average density: {avg_density:.6f}")
        print(f"  Total time: {elapsed_time:.3f} seconds")
        print(f"  Total lattice updates: {total_lattice_updates:.2e}")
        print(f"  Performance: {mlups:.2f} MLUPS")
        
        results[f'{backend_name}_fluid_time'] = elapsed_time
        results[f'{backend_name}_fluid_mlups'] = mlups
        
        # Test 3: Memory Bandwidth Test (skip for WARP as it's more complex to implement)
        if backend_name != 'WARP_GPU':
            print(f"\n{'-'*60}")
            print(f"Test 3: Memory Bandwidth Test ({backend_name})")
            
            print(f"Array size: {array_size} elements (~{array_size*4/1e6:.0f} MB)")
            print(f"Number of copy operations: {num_copies}")
            
            # Create large array
            key, subkey = random.split(key)
            large_array = random.normal(subkey, (array_size,), dtype=jnp.float32)
            
            # Warm up JIT compilation
            _ = memory_intensive_ops_optimized(large_array[:1000])
            
            start_time = time.time()
            for i in range(num_copies):
                # Use chunked processing for better memory efficiency
                if array_size > 50_000_000:  # Use chunking for very large arrays
                    result = chunked_memory_operations(large_array)
                else:
                    result, _ = memory_intensive_ops_optimized(large_array)
                result.block_until_ready()
                if i % 2 == 0:
                    print(f"  Copy {i+1}: Sum = {result:.6f}")
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            # Calculate memory bandwidth
            bytes_per_op = array_size * 4 * 3
            total_bytes = bytes_per_op * num_copies
            bandwidth_gbps = total_bytes / elapsed_time / 1e9
            
            print(f"\nMemory Bandwidth Results ({backend_name}):")
            print(f"  Total time: {elapsed_time:.3f} seconds")
            print(f"  Total data processed: {total_bytes/1e9:.2f} GB")
            print(f"  Memory bandwidth: {bandwidth_gbps:.2f} GB/s")
            
            results[f'{backend_name}_memory_time'] = elapsed_time
            results[f'{backend_name}_memory_bandwidth'] = bandwidth_gbps
    
    # Comparison Summary
    print("\n" + "="*80)
    print("CPU vs JAX GPU vs WARP GPU PERFORMANCE COMPARISON")
    print("="*80)
    
    print("\n📊 MATRIX OPERATIONS:")
    cpu_matrix_time = results['CPU_matrix_time']
    jax_gpu_matrix_time = results['JAX_GPU_matrix_time']
    cpu_matrix_tflops = results['CPU_matrix_tflops']
    jax_gpu_matrix_tflops = results['JAX_GPU_matrix_tflops']
    
    print(f"  CPU: {cpu_matrix_time:.3f}s, {cpu_matrix_tflops:.3f} TFLOPS")
    print(f"  JAX GPU: {jax_gpu_matrix_time:.3f}s, {jax_gpu_matrix_tflops:.3f} TFLOPS")
    
    if WARP_AVAILABLE:
        warp_gpu_matrix_time = results['WARP_GPU_matrix_time']
        warp_gpu_matrix_tflops = results['WARP_GPU_matrix_tflops']
        print(f"  WARP GPU: {warp_gpu_matrix_time:.3f}s, {warp_gpu_matrix_tflops:.3f} TFLOPS")
        
        jax_vs_cpu_speedup = cpu_matrix_time / jax_gpu_matrix_time
        warp_vs_cpu_speedup = cpu_matrix_time / warp_gpu_matrix_time
        jax_vs_warp_ratio = warp_gpu_matrix_time / jax_gpu_matrix_time
        
        print(f"  JAX GPU vs CPU: {jax_vs_cpu_speedup:.2f}x faster")
        print(f"  WARP GPU vs CPU: {warp_vs_cpu_speedup:.2f}x faster")
        print(f"  JAX vs WARP: {jax_vs_warp_ratio:.2f}x {'faster' if jax_vs_warp_ratio < 1 else 'slower'}")
    
    print("\n🌊 FLUID DYNAMICS (LBM):")
    cpu_fluid_time = results['CPU_fluid_time']
    jax_gpu_fluid_time = results['JAX_GPU_fluid_time']
    cpu_fluid_mlups = results['CPU_fluid_mlups']
    jax_gpu_fluid_mlups = results['JAX_GPU_fluid_mlups']
    
    print(f"  CPU: {cpu_fluid_time:.3f}s, {cpu_fluid_mlups:.2f} MLUPS")
    print(f"  JAX GPU: {jax_gpu_fluid_time:.3f}s, {jax_gpu_fluid_mlups:.2f} MLUPS")
    
    if WARP_AVAILABLE:
        warp_gpu_fluid_time = results['WARP_GPU_fluid_time']
        warp_gpu_fluid_mlups = results['WARP_GPU_fluid_mlups']
        print(f"  WARP GPU: {warp_gpu_fluid_time:.3f}s, {warp_gpu_fluid_mlups:.2f} MLUPS")
        
        jax_vs_cpu_fluid_speedup = cpu_fluid_time / jax_gpu_fluid_time
        warp_vs_cpu_fluid_speedup = cpu_fluid_time / warp_gpu_fluid_time
        jax_vs_warp_fluid_ratio = warp_gpu_fluid_time / jax_gpu_fluid_time
        
        print(f"  JAX GPU vs CPU: {jax_vs_cpu_fluid_speedup:.2f}x faster")
        print(f"  WARP GPU vs CPU: {warp_vs_cpu_fluid_speedup:.2f}x faster")
        print(f"  JAX vs WARP: {jax_vs_warp_fluid_ratio:.2f}x {'faster' if jax_vs_warp_fluid_ratio < 1 else 'slower'}")
    
    print("\n💾 MEMORY BANDWIDTH (JAX only):")
    cpu_memory_time = results['CPU_memory_time']
    jax_gpu_memory_time = results['JAX_GPU_memory_time']
    cpu_memory_bw = results['CPU_memory_bandwidth']
    jax_gpu_memory_bw = results['JAX_GPU_memory_bandwidth']
    
    speedup_memory = cpu_memory_time / jax_gpu_memory_time
    bandwidth_ratio = jax_gpu_memory_bw / cpu_memory_bw
    
    print(f"  CPU: {cpu_memory_time:.3f}s, {cpu_memory_bw:.2f} GB/s")
    print(f"  JAX GPU: {jax_gpu_memory_time:.3f}s, {jax_gpu_memory_bw:.2f} GB/s")
    print(f"  JAX GPU Speedup: {speedup_memory:.2f}x faster")
    print(f"  Bandwidth Ratio: {bandwidth_ratio:.2f}x better")
    
    print("\n🏆 OVERALL ASSESSMENT:")
    if WARP_AVAILABLE:
        print("  JAX GPU shows excellent performance for high-level array operations")
        print("  WARP GPU provides fine-grained control for custom kernel development")
        print("  Both frameworks offer significant speedups over CPU for parallel workloads")
    else:
        jax_overall_speedup = (cpu_matrix_time / jax_gpu_matrix_time + cpu_fluid_time / jax_gpu_fluid_time + speedup_memory) / 3
        print(f"  Average JAX GPU Speedup: {jax_overall_speedup:.2f}x")
        print("  WARP not available for comparison")
    
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)

if __name__ == "__main__":
    benchmark_computation()