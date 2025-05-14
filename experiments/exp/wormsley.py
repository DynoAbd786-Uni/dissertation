# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.special import jv

# # Parameters
# rho = 1060       # Blood density (kg/m³)
# mu = 0.004       # Viscosity (Pa.s)
# f = 1.0          # Heart rate in Hz (~60 bpm)
# omega = 2 * np.pi * f
# R = 0.0005        # Vessel radius (5 mm)
# alpha = R * np.sqrt(omega * rho / mu)

# # Grid
# r = np.linspace(0, R, 100)
# t = np.linspace(0, 1/f, 8)  # 8 time points over one heartbeat

# # Pressure gradient amplitude (arbitrary units)
# P_amp = 1000

# # Velocity profiles over time
# fig, axs = plt.subplots(2, 4, figsize=(16, 6))
# axs = axs.flatten()

# for i, ti in enumerate(t):
#     zeta = 1j ** 1.5 * alpha * r / R
#     J0_zeta = jv(0, zeta)
#     J0_alpha = jv(0, 1j ** 1.5 * alpha)

#     u = np.real((P_amp / (1j * omega * rho)) * (1 - J0_zeta / J0_alpha) * np.exp(1j * omega * ti))

#     axs[i].plot(u, r * 1e3)  # r in mm
#     axs[i].set_title(f"t = {ti:.2f} s")
#     axs[i].set_xlabel("Velocity (m/s)")
#     axs[i].set_ylabel("Radial Position (mm)")
#     axs[i].grid()

# plt.tight_layout()
# plt.suptitle("Womersley Flow Velocity Profiles Over Time", fontsize=16, y=1.02)
# plt.show()


import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv

# CCA + Womersley setup (same as before)…
R, nu, HR = 0.0035, 3.5e-6, 60
f, omega = HR/60, 2*np.pi*(HR/60)
alpha = R*np.sqrt(omega/nu)
i32 = alpha * np.exp(1j*3*np.pi/4)
H = 1 - 2*jv(1, i32)/(i32*jv(0, i32))
T = 1/f
t = np.linspace(0, T, 500)
u_center = np.real(H * np.exp(1j*omega*t))

# Add a steady component:
U_mean = 0.4                # choose your mean velocity [m/s]
u_total = U_mean + u_center

# Plot
plt.plot(t*1000, u_center,    label='Oscillatory only')
plt.plot(t*1000, u_total,     label='Oscillatory + mean')
plt.xlabel('Time (ms)')
plt.ylabel('Centerline velocity (m/s)')
plt.title(f'CCA Womersley Flow (α={alpha:.2f})')
plt.legend()
plt.grid(True)
plt.show()

