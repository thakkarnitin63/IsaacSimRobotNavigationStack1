# In: navigation_stack/utils/height_map_processor.py

import torch
import numpy as np

# --- Auto-select GPU if available ---
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    print("HeightMapProcessor: Using GPU (CUDA)")
else:
    DEVICE = torch.device("cpu")
    print("HeightMapProcessor: Warning! CUDA not available, using CPU.")
# ------------------------------------

class HeightMapProcessor:
    """
    Creates a 2.5D Height Map from a 3D Lidar point cloud.
    
    This processor converts 3D Lidar points into a 2D grid where each
    cell (x, y) stores the Z-value (height) of the tallest obstacle
    in that cell. This is done in the WORLD frame.
    """
    def __init__(self, grid_resolution, grid_width_m, grid_height_m, robot_height, robot_ground_clearance):
        
        self.resolution = grid_resolution  # e.g., 0.05 meters per cell
        
        # Convert meters to cell counts
        self.width_cells = int(grid_width_m / self.resolution)
        self.height_cells = int(grid_height_m / self.resolution)
        
        self.robot_height = robot_height   # Max z-value to care about
        self.ground_clearance = robot_ground_clearance # Min z-value
        
        # Center the grid at (0,0) in the world
        # This is the (x, y) coordinate of the bottom-left corner
        self.origin = torch.tensor(
            [- (self.width_cells / 2.0) * self.resolution, 
             - (self.height_cells / 2.0) * self.resolution],
            device=DEVICE, dtype=torch.float32
        )
        
        # The height map tensor, initialized to 0
        self.height_map = torch.zeros((self.height_cells, self.width_cells), device=DEVICE, dtype=torch.float32)

        print(f"--- HeightMapProcessor Initialized ---")
        print(f"  Grid: {self.width_cells} (w) x {self.height_cells} (h) @ {self.resolution}m")
        print(f"  Z-Filter Range: [{self.ground_clearance}m, {self.robot_height}m]")
        print("--------------------------------------")

    def _world_to_grid(self, points_xy):
        """Converts (N, 2) world (x, y) points to (N, 2) grid (col, row) indices."""
        # Shift points by origin and scale by resolution
        indices = (points_xy - self.origin) / self.resolution
        return torch.floor(indices).long()

    def _transform_points(self, points, pos, quat_xyzw):
        """Transforms (N, 3) points by a (3,) pos and (4,) quat_xyzw."""
        
        # Convert (x, y, z, w) to (w, x, y, z) for math
        q_wxyz = torch.tensor([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], device=DEVICE)
        
        # Create a "pure" quaternion for each point (0, x, y, z)
        points_quat = torch.cat(
            (torch.zeros(points.shape[0], 1, device=DEVICE), points), dim=1
        )
        
        # Standard quaternion rotation: q * p * q_conj
        rotated_points = self._quat_multiply(q_wxyz, self._quat_multiply(points_quat, self._quat_conjugate(q_wxyz)))
        
        # Return [x, y, z] part and add translation
        return rotated_points[..., 1:4] + pos
        
    def _quat_conjugate(self, q):
        """Calculates the conjugate of a (w, x, y, z) quaternion."""
        q_conj = q.clone()
        q_conj[1:] *= -1
        return q_conj

    def _quat_multiply(self, q1, q2):
        """Multiplies two (w, x, y, z) quaternions (or batches)."""
        w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
        w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
        
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        
        return torch.stack([w, x, y, z], dim=-1)

    def process_point_cloud(self, points_raw, lidar_to_base_pos, lidar_to_base_quat, robot_world_pos, robot_world_quat):
        """
        The main public function. This MUST be run in the WORLD frame.
        
        Args:
            points_raw (np.array): [N, 3] points in Lidar frame
            lidar_to_base_pos (np.array): [3] (x,y,z)
            lidar_to_base_quat (np.array): [4] (x,y,z,w)
            robot_world_pos (np.array): [3] (x,y,z)
            robot_world_quat (np.array): [4] (x,y,z,w)
            
        Returns:
            torch.Tensor: The 2.5D height map
        """
        
        # 0. Clear the old map
        self.height_map.fill_(0.0)
        
        if points_raw.size == 0:
            return self.height_map # Return empty map
            
        # 1. Convert inputs to GPU tensors
        points_lidar_frame = torch.tensor(points_raw, dtype=torch.float32, device=DEVICE)
        
        # Static Lidar-to-Base transform
        l_to_b_pos = torch.tensor(lidar_to_base_pos, dtype=torch.float32, device=DEVICE)
        l_to_b_quat = torch.tensor(lidar_to_base_quat, dtype=torch.float32, device=DEVICE)
        
        # Dynamic Base-to-World transform
        b_to_w_pos = torch.tensor(robot_world_pos, dtype=torch.float32, device=DEVICE)
        b_to_w_quat = torch.tensor(robot_world_quat, dtype=torch.float32, device=DEVICE)
        
        # 2. Transform points from Lidar frame -> Base frame
        points_base_frame = self._transform_points(points_lidar_frame, l_to_b_pos, l_to_b_quat)
        
        # 3. Transform points from Base frame -> World frame
        points_world_frame = self._transform_points(points_base_frame, b_to_w_pos, b_to_w_quat)
        
        # 4. Filter points based on height
        z_coords = points_world_frame[..., 2]
        
        # Filter points that are on the ground OR too high to care about
        valid_mask = (z_coords > self.ground_clearance) & (z_coords < self.robot_height)
        
        filtered_points = points_world_frame[valid_mask]
        
        if filtered_points.shape[0] == 0:
            return self.height_map # Return empty map
            
        points_xy = filtered_points[..., :2]
        points_z = filtered_points[..., 2]
        
        # 5. Convert (x, y) to (col, row) grid indices
        grid_indices = self._world_to_grid(points_xy)
        
        # 6. Filter out-of-bounds indices
        col_indices = grid_indices[..., 0]
        row_indices = grid_indices[..., 1]
        
        bounds_mask = (col_indices >= 0) & (col_indices < self.width_cells) & \
                      (row_indices >= 0) & (row_indices < self.height_cells)
                      
        valid_indices = grid_indices[bounds_mask]
        valid_z = points_z[bounds_mask]
        
        if valid_indices.shape[0] == 0:
            return self.height_map # Return empty map

        # 7. Update the height map
        # This is a "scatter max" operation.
        # We use a trick: reverse the z-values, scatter_min, then reverse back.
        # This finds the max z-value for each cell.
        
        # Note: PyTorch uses (row, col) which is (y, x)
        row_idx = valid_indices[:, 1]
        col_idx = valid_indices[:, 0]
        
        # Create a linear index for scatter
        linear_indices = row_idx * self.width_cells + col_idx
        
        # We want the max Z, so we scatter the *negative* Z and find the *min*
        # (min(-z) is the same as max(z))
        
        # Fill map with a large number (so min works)
        self.height_map.fill_(1e10)
        
        # Scatter the negative z-values
        self.height_map.view(-1).scatter_reduce_(
            0, 
            linear_indices, 
            -valid_z, 
            reduce="amin", # 'amin' finds the minimum
            include_self=False
        )
        
        # Invert the values back
        # Cells that were never touched are 1e10, set them to 0
        self.height_map[self.height_map == 1e10] = 0.0
        # Cells that were touched are negative, flip them back
        self.height_map = -self.height_map
        
        return self.height_map