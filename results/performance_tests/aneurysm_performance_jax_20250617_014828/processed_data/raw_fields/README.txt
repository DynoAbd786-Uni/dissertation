aneurysm_performance_jax_20250617_014828 Field Data
===================================================

Date: 2025-06-18 09:01:36
Number of frames: 11

Available fields:
- dimensions: (3,)
- rho: (11, 530, 1751)
- u_x: (11, 530, 1751)
- u_y: (11, 530, 1751)
- u_magnitude: (11, 530, 1751)
- wss_magnitude: (11, 530, 1751)
- wss_x: (11, 530, 1751)
- wss_y: (11, 530, 1751)
- wall_mask: (11, 530, 1751)
- inlet_mask: (11, 530, 1751)
- outlet_mask: (11, 530, 1751)

Notes:
- These files contain 3D arrays with dimensions [frames, height, width]
- The frame_mapping.npz file contains the mapping between frame numbers and indices
- Each field is stored as a separate .npz file for easier loading
- To load a field: data = np.load('field_name.npz')['data']
