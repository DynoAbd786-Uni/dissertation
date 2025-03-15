# main.py
import warp as wp
import numpy as np
from constants import Y_VALUES_1, Y_VALUES_2, Y_VALUES_3, Y_VALUES_4, load_profile_values

@wp.kernel
def print_matrices_kernel():
    tid = wp.tid()
    if tid == 0:
        # Directly access the Warp constants
        for i in range(4):
            for j in range(4):
                wp.printf("Y_VALUES_1[%d][%d] = %f\n", i, j, Y_VALUES_1[i][j])
        # Repeat for Y_VALUES_2, Y_VALUES_3, Y_VALUES_4...


# Launch the kernel
y_values = [i for i in range(64)]
y_values_array = np.array(y_values, dtype=np.float32)
load_profile_values(y_values_array)
from constants import Y_VALUES_1, Y_VALUES_2, Y_VALUES_3, Y_VALUES_4
wp.launch(print_matrices_kernel, dim=1)
wp.synchronize()