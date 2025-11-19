# navigation_stack/perception/numba_kernels.py
import torch
from numba import cuda
import math

@cuda.jit
def launch_ray_trace_kernel_DDA(grid, 
                                start_pos, 
                                end_indices, 
                                grid_shape, 
                                min_val):
    """
    A Numba CUDA kernel to perform a 3D DDA (Digital Differential Analyzer)
    voxel traversal algorithm in parallel.
    
    This kernel is launched with one thread for every ray.
    Every voxel it passes through is set to 'min_val' (0.0).

    This DDA implementation is adapted from "A Fast Voxel Traversal Algorithm"
    by Amanatides and Woo. It is more accurate for ray casting than Bresenham's
    as it visits *all* voxels the ray passes through.

    Args:
        grid: A Numba-view of the 3D PyTorch grid tensor.
        start_pos: A [3] array (x, y, z) of the sensor origin (in grid coords).
        end_indices: A [N, 3] array of all Lidar hit points (in grid coords).
        grid_shape: A [3] array of the grid dimensions for bounds checking.
        min_val: The value to write into the grid (e.g., 0.0 for FREE).
    """
    
    i = cuda.grid(1)
    if i >= end_indices.shape[0]:
        return

    # --- 1. Get Start/End Points (as floats) ---
    x = start_pos[0]
    y = start_pos[1]
    z = start_pos[2]
    
    end_x = end_indices[i, 0]
    end_y = end_indices[i, 1]
    end_z = end_indices[i, 2]

    # --- 2. 3D DDA (Amanatides-Woo) Setup ---
    
    # Get the current voxel index (as integer)
    curr_vx = int(math.floor(x))
    curr_vy = int(math.floor(y))
    curr_vz = int(math.floor(z))
    
    # Get the target voxel index (as integer)
    target_vx = int(math.floor(end_x))
    target_vy = int(math.floor(end_y))
    target_vz = int(math.floor(end_z))

    # Direction of the ray
    ray_dir_x = end_x - x
    ray_dir_y = end_y - y
    ray_dir_z = end_z - z

    # Step direction (which way to increment voxel index: +1 or -1)
    step_vx = 1 if ray_dir_x > 0 else -1
    step_vy = 1 if ray_dir_y > 0 else -1
    step_vz = 1 if ray_dir_z > 0 else -1

    # DeltaT: How far (in "t") we must travel along the ray
    # to cross one voxel in each dimension.
    t_delta_x = abs(1.0 / ray_dir_x) if ray_dir_x != 0 else 1e10
    t_delta_y = abs(1.0 / ray_dir_y) if ray_dir_y != 0 else 1e10
    t_delta_z = abs(1.0 / ray_dir_z) if ray_dir_z != 0 else 1e10

    # T_Max: The "t" value at which we first hit the *next*
    # voxel boundary in each dimension.
    next_bound_x = (curr_vx + 1.0) if ray_dir_x > 0 else float(curr_vx)
    next_bound_y = (curr_vy + 1.0) if ray_dir_y > 0 else float(curr_vy)
    next_bound_z = (curr_vz + 1.0) if ray_dir_z > 0 else float(curr_vz)

    t_max_x = (next_bound_x - x) / ray_dir_x if ray_dir_x != 0 else 1e10
    t_max_y = (next_bound_y - y) / ray_dir_y if ray_dir_y != 0 else 1e10
    t_max_z = (next_bound_z - z) / ray_dir_z if ray_dir_z != 0 else 1e10

    # --- 3. Voxel Traversal Loop ---
    # We will step a max of 200 voxels (or grid_shape[0] * 2, etc.)
    # to prevent infinite loops from bad inputs.
    for _ in range(200): # Should be > max(grid_dims)
        
        # 3a. Write to the current voxel (if in bounds)
        if 0 <= curr_vx < grid_shape[0] and \
           0 <= curr_vy < grid_shape[1] and \
           0 <= curr_vz < grid_shape[2]:
            grid[curr_vx, curr_vy, curr_vz] = min_val

        # 3b. Check if we reached the target voxel
        if curr_vx == target_vx and \
           curr_vy == target_vy and \
           curr_vz == target_vz:
            break # We're done with this ray

        # 3c. DDA Step: Find the *next* closest voxel boundary
        if t_max_x < t_max_y:
            if t_max_x < t_max_z:
                curr_vx += step_vx
                t_max_x += t_delta_x
            else:
                curr_vz += step_vz
                t_max_z += t_delta_z
        else:
            if t_max_y < t_max_z:
                curr_vy += step_vy
                t_max_y += t_delta_y
            else:
                curr_vz += step_vz
                t_max_z += t_delta_z

def clear_free_space_kernel(grid_tensor: torch.Tensor, 
                            start_index: torch.Tensor, 
                            end_indices: torch.Tensor, 
                            grid_shape: torch.Tensor, 
                            min_val: float):
    """
    A Python "launcher" function that configures and runs the Numba kernel.
    """
    
    # 1. Create a Numba CUDA array view of the PyTorch tensor's memory
    grid_numba_view = cuda.as_cuda_array(grid_tensor)

    # 2. Configure the kernel launch parameters
    threads_per_block = 256
    blocks_per_grid = (end_indices.shape[0] + (threads_per_block - 1)) \
                      // threads_per_block
    
    # 3. Launch the DDA kernel
    launch_ray_trace_kernel_DDA[blocks_per_grid, threads_per_block](
        grid_numba_view,
        start_index,
        end_indices,
        grid_shape,
        min_val
    )