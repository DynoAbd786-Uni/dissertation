CCA_simulation_results_nnbgk_tdzh Field Data
============================================

Date: 2025-05-10 15:09:21
Number of frames: 101

Available fields:
- dimensions: (3,)
- rho: (101, 528, 749)
- u_x: (101, 528, 749)
- u_y: (101, 528, 749)
- u_magnitude: (101, 528, 749)
- wss_magnitude: (101, 528, 749)
- wss_x: (101, 528, 749)
- wss_y: (101, 528, 749)
- wall_mask: (101, 528, 749)

Notes:
- These files contain 3D arrays with dimensions [frames, height, width]
- The frame_mapping.npz file contains the mapping between frame numbers and indices
- Each field is stored as a separate .npz file for easier loading
- To load a field: data = np.load('field_name.npz')['data']
