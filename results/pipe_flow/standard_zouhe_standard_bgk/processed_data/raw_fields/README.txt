standard_zouhe_standard_bgk Raw Field Data
=========================================

Date: 2025-05-09 23:42:13
Number of frames: 101

Available fields:
- rho: (101, 245672)
- u_x: (101, 245672)
- u_y: (101, 245672)
- u_magnitude: (101, 245672)
- wss_magnitude: (101, 245672)
- wss_x: (101, 245672)
- wss_y: (101, 245672)
- wall_mask: (101, 245672)
- dimensions: (3,)

Notes:
- These files contain the raw field data without reshaping
- The frame_mapping.npz file contains the mapping between frame numbers and indices
- Each field is stored as a separate .npz file for easier loading
- To load a field: data = np.load('field_name.npz')['data']
