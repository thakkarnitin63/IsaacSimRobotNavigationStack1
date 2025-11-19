# navigation_stack/perception/frustum_model.py

"""
Three-Dimensional Lidar Frustum Model

"""
import torch
import math
from typing import Tuple
import kaolin.math.quat as Kq

class ThreeDimensionalLidarFrustum:
    """
    Defines a 3D viewing frustum for a lidar sensor.
    It can test whether points in world coordinates fall within
    the sensor's field of view.
    """
    def __init__(self, 
                 v_fov: float, 
                 v_fov_padding: float, 
                 h_fov: float,
                 min_d: float, 
                 max_d: float, 
                 device: torch.device):
        """
        Initialize the frustum parameters.
        
        Args:
            v_fov: Vertical field of view in RADIANS
                   Example: 0.7 rad ≈ 40 degrees
                   This is the TOTAL vertical angle (top to bottom)
                   
            v_fov_padding: Vertical FOV padding in METERS (not radians!)
                          This expands the vertical FOV slightly
                          Example: 0.05m = 5cm padding
                          Purpose: Account for sensor mounting uncertainty
                          
            h_fov: Horizontal field of view in RADIANS
                   Example: 6.28 rad ≈ 360 degrees (full rotation lidar)
                   Example: 1.04 rad ≈ 60 degrees (narrow FOV lidar)
                   
            min_d: Minimum detection distance in METERS
                   Example: 0.1m = ignore points < 10cm away
                   Purpose: Sensors have a "blind spot" very close
                   
            max_d: Maximum detection distance in METERS  
                   Example: 10.0m = ignore points > 10m away
                   Purpose: Sensor range limit, or computational limit
                   
            device: torch.device('cuda:0') or torch.device('cpu')
        
        Implementation Notes:
        --------------------
        - All angles stored in radians internally
        - Pre-compute expensive operations (tan, squared values)
        """
        self.device = device
        
        # === Store Raw Parameters (matching C++ member names) ===
        self._vFOV = v_fov              # Vertical FOV (radians)
        self._vFOVPadding = v_fov_padding  # Vertical padding (meters)
        self._hFOV = h_fov              # Horizontal FOV (radians)
        self._min_d = min_d             # Min range (meters)
        self._max_d = max_d             # Max range (meters)
        
        # === Pre-compute Values  ===
        
        # Horizontal FOV half-angle
        # Used in horizontal FOV check
        self._hFOVhalf = h_fov / 2.0
        
        # Tangent of vertical FOV half-angle
        # Used for vertical FOV cone check
        # Why tangent? Because tan(angle) = opposite/adjacent
        # In our case: tan(v_fov/2) = height_offset / radial_distance
        self._tan_vFOVhalf = math.tan(v_fov / 2.0)
        
        # Squared tangent (avoid repeated squaring)
        self._tan_vFOVhalf_squared = self._tan_vFOVhalf * self._tan_vFOVhalf
        
        # Squared distances (avoid repeated squaring)
        self._min_d_squared = min_d * min_d
        self._max_d_squared = max_d * max_d
        
        # Check if sensor is 360 degrees (full rotation)
        # If h_fov > 6.27 rad (≈ 359°), treat as full circle
        # This optimization skips horizontal FOV checks
        self._full_hFOV = (h_fov > 6.27)
        
        # === Transform (set by set_transform) ===
        # These are updated each frame with sensor's pose
        self._position = None           # Sensor position in world frame [x, y, z]
        self._orientation_inverse = None # Conjugate (for inverse rotation)
        
        print(f"   Frustum initialized:")
        print(f"     Vertical FOV: {math.degrees(v_fov):.1f}° (±{math.degrees(v_fov/2):.1f}°)")
        print(f"     Horizontal FOV: {math.degrees(h_fov):.1f}° ({'360°' if self._full_hFOV else 'directional'})")
        print(f"     Range: {min_d:.2f}m to {max_d:.2f}m")
        print(f"     Vertical padding: {v_fov_padding:.3f}m")
    
    def set_transform(self, position: torch.Tensor, orientation_quat: torch.Tensor):
        """
        Set the sensor's pose in world coordinates.
        
        Args:
            position: [3] tensor (x, y, z) in world frame
                     
            orientation_quat: [4] tensor (x, y, z, w) quaternion
        
        """
        self._position = position
        self._orientation_inverse = Kq.quat_inverse(orientation_quat)

    def is_inside(self, points: torch.Tensor) -> torch.Tensor:
        """
        Check if points are inside the frustum.
        """
        # Handle empty input
        if points.shape[0] == 0:
            return torch.zeros(0, device=self.device, dtype=torch.bool)

        if self._position is None or self._orientation_inverse is None:
            return torch.zeros(points.shape[0], device=self.device, dtype=torch.bool)
        
        # === STEP 1: Transform to Sensor Frame ===
        points_translated = points - self._position  # [N, 3]
        
        # quat_rotate expects: rotation [b, 4], point [b, 3]
        N = points_translated.shape[0]
        quat_expanded = self._orientation_inverse.unsqueeze(0).expand(N, -1)  # [N, 4]
        
        transformed_pts = Kq.quat_rotate(quat_expanded, points_translated)  # [N, 3]
        
        print(f"  Frustum Debug - First point:")
        print(f"    Original: {points[0].cpu().numpy()}")
        print(f"    Translated: {points_translated[0].cpu().numpy()}")
        print(f"    Transformed: {transformed_pts[0].cpu().numpy()}")
        
        # === STEP 2: Radial Distance Check ===
        radial_distance_squared = (
            transformed_pts[:, 0] * transformed_pts[:, 0] +
            transformed_pts[:, 1] * transformed_pts[:, 1]
        )
        
        print(f"    Radial dist²: {radial_distance_squared[0].item():.4f} (should be in [{self._min_d_squared:.4f}, {self._max_d_squared:.4f}])")
        
        # === STEP 3: Range Check ===
        range_ok = (radial_distance_squared <= self._max_d_squared) & \
                (radial_distance_squared >= self._min_d_squared)
        
        print(f"    Range check: {range_ok[0].item()}")
        
        # === STEP 4: Vertical FOV Check ===
        v_padded = torch.abs(transformed_pts[:, 2]) + self._vFOVPadding
        v_fov_ok = (v_padded * v_padded / (radial_distance_squared + 1e-8)) <= \
                self._tan_vFOVhalf_squared
        
        ratio = (v_padded[0] * v_padded[0] / (radial_distance_squared[0] + 1e-8)).item()
        print(f"    V-FOV check: {v_fov_ok[0].item()}, ratio={ratio:.4f} vs threshold={self._tan_vFOVhalf_squared:.4f}")
        
        # === STEP 5: Horizontal FOV Check ===
        if not self._full_hFOV:
            x = transformed_pts[:, 0]
            y = transformed_pts[:, 1]
            
            half_pi = math.pi / 2.0
            
            angles_1 = torch.abs(torch.atan(y / (x + 1e-8)))
            mask_1 = (x > 0) & (angles_1 <= self._hFOVhalf)
            
            angles_2 = torch.abs(torch.atan(x / (y + 1e-8))) + half_pi
            mask_2 = (x <= 0) & (angles_2 <= self._hFOVhalf)
            
            h_fov_ok = mask_1 | mask_2
        else:
            h_fov_ok = torch.ones(points.shape[0], device=self.device, dtype=torch.bool)
        
        print(f"    H-FOV check: {h_fov_ok[0].item()} (full 360°)")
        
        # === STEP 6: Combine All Checks ===
        inside = range_ok & v_fov_ok & h_fov_ok
        
        print(f"    FINAL: {inside[0].item()}")
        print(f"  Total passing: {inside.sum().item()}/{N}")
        
        return inside
    

