import jax
import jax.numpy as jnp

try:
    print(jax.devices())
    x = jnp.ones((10, 10))
    y = jax.device_put(x, jax.devices('gpu')[0])
    print("GPU test successful!")
except Exception as e:
    print(f"GPU test failed: {e}")