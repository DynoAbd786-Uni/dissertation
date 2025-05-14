import warp as wp
import numpy as np

PARAMS_DIR = "../params"


# These will store the 64 values from the flow profile divided into 4 matrices
Y_VALUES_1 = wp.constant(wp.mat44())  # First 16 values (0-15)
Y_VALUES_2 = wp.constant(wp.mat44())  # Next 16 values (16-31)
Y_VALUES_3 = wp.constant(wp.mat44())  # Next 16 values (32-47)
Y_VALUES_4 = wp.constant(wp.mat44())  # Final 16 values (48-63)

def load_profile_values(y_values):
    """
    Load 64 y-values from a selected profile into 4 wp.mat44 matrices
    as global variables.
    
    Args:
        y_values: Array of y values from the selected profile
    """
    global Y_VALUES_1, Y_VALUES_2, Y_VALUES_3, Y_VALUES_4
    
    # Ensure we have exactly 64 values (resample if needed)
    if len(y_values) != 64:
        print(f"Profile has {len(y_values)} values, resampling to 64...")
        indices = np.linspace(0, len(y_values) - 1, 64)
        y_values = np.interp(indices, np.arange(len(y_values)), y_values)
    
    # Convert to float32 for Warp compatibility
    y_values = y_values.astype(np.float32)
    
    # Split into 4 chunks of 16 values each
    chunk1 = y_values[0:16]
    chunk2 = y_values[16:32]
    chunk3 = y_values[32:48]
    chunk4 = y_values[48:64]
    
    # Reshape each chunk into a 4x4 grid (row-major order)
    grid1 = chunk1.reshape(4, 4)
    grid2 = chunk2.reshape(4, 4)
    grid3 = chunk3.reshape(4, 4)
    grid4 = chunk4.reshape(4, 4)
    
    # Create wp.mat44 objects and store directly to global variables
    Y_VALUES_1 = wp.constant(wp.mat44(
        grid1[0, 0], grid1[0, 1], grid1[0, 2], grid1[0, 3],
        grid1[1, 0], grid1[1, 1], grid1[1, 2], grid1[1, 3],
        grid1[2, 0], grid1[2, 1], grid1[2, 2], grid1[2, 3],
        grid1[3, 0], grid1[3, 1], grid1[3, 2], grid1[3, 3]
    ))
    
    Y_VALUES_2 = wp.constant(wp.mat44(
        grid2[0, 0], grid2[0, 1], grid2[0, 2], grid2[0, 3],
        grid2[1, 0], grid2[1, 1], grid2[1, 2], grid2[1, 3],
        grid2[2, 0], grid2[2, 1], grid2[2, 2], grid2[2, 3],
        grid2[3, 0], grid2[3, 1], grid2[3, 2], grid2[3, 3]
    ))
    
    Y_VALUES_3 = wp.constant(wp.mat44(
        grid3[0, 0], grid3[0, 1], grid3[0, 2], grid3[0, 3],
        grid3[1, 0], grid3[1, 1], grid3[1, 2], grid3[1, 3],
        grid3[2, 0], grid3[2, 1], grid3[2, 2], grid3[2, 3],
        grid3[3, 0], grid3[3, 1], grid3[3, 2], grid3[3, 3]
    ))
    
    Y_VALUES_4 = wp.constant(wp.mat44(
        grid4[0, 0], grid4[0, 1], grid4[0, 2], grid4[0, 3],
        grid4[1, 0], grid4[1, 1], grid4[1, 2], grid4[1, 3],
        grid4[2, 0], grid4[2, 1], grid4[2, 2], grid4[2, 3],
        grid4[3, 0], grid4[3, 1], grid4[3, 2], grid4[3, 3]
    ))
    
    print("Profile data loaded into global mat44 matrices")
