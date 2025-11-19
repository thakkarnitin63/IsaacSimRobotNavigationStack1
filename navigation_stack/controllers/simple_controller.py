# In: navigation_stack/controllers/simple_controller.py

import numpy as np

class SimpleController:
    """Dead simple proportional controller for testing"""
    
    def __init__(self):
        self.max_v = 0.25
        self.max_w = 0.8
        
    def compute_control_command(self, current_pose, path, height_map):
        if len(path) == 0:
            return [0.0, 0.0]
        
        # Get target
        target = np.array(path[0])
        robot_pos = current_pose[:2]
        robot_heading = current_pose[2]
        
        # Calculate errors
        dx = target[0] - robot_pos[0]
        dy = target[1] - robot_pos[1]
        distance = np.sqrt(dx**2 + dy**2)
        
        desired_heading = np.arctan2(dy, dx)
        heading_error = desired_heading - robot_heading
        
        # Normalize to [-pi, pi]
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
        
        # Simple proportional control
        if abs(heading_error) > 0.5:  # 28 degrees
            # Turn in place
            v = 0.05
            w = np.clip(2.0 * heading_error, -self.max_w, self.max_w)
        else:
            # Move forward
            v = np.clip(0.5 * distance, 0.0, self.max_v)
            w = np.clip(1.5 * heading_error, -self.max_w, self.max_w)
        
        return [float(v), float(w)]