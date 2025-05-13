import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import matplotlib.tri as mtri

# 1. Create synthetic aneurysm wall: noisy ellipse
θ = np.linspace(0, 2*np.pi, 200)
r = 1 + 0.3*np.sin(3*θ) + 0.05*np.random.randn(θ.size)  # bumpy
x = r * np.cos(θ)
y = r * np.sin(θ)

# 2. Compute tangential WSS direction (approx. as derivative along θ)
dx_dθ = np.gradient(x, θ)
dy_dθ = np.gradient(y, θ)
tangent_norm = np.hypot(dx_dθ, dy_dθ)
u = dx_dθ / tangent_norm
v = dy_dθ / tangent_norm

# 3. Create a synthetic WSS magnitude pattern
WSS_mag = 0.5 + 0.5*np.cos(5*θ)  # varies between 0 and 1

# 4. Triangulate the 2D space
points = np.vstack((x, y)).T
tri = Delaunay(points)
triang = mtri.Triangulation(x, y, triangles=tri.simplices)

# 5. Plotting
fig, ax = plt.subplots(figsize=(6,6))

# Background WSS magnitude
tcf = ax.tricontourf(triang, WSS_mag, levels=50, cmap='viridis')
fig.colorbar(tcf, ax=ax, label='WSS Magnitude')

# Quiver plot for WSS direction
ax.quiver(x, y, u, v, WSS_mag, cmap='plasma', scale=20, width=0.003)

# Aesthetics
ax.set_aspect('equal')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('WSS on 2D Aneurysm Wall\nDirection (Arrows) and Magnitude (Colour)')
plt.tight_layout()
plt.show()
