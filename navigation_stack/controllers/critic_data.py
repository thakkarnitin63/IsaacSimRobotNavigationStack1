from dataclasses import dataclass
import torch
from typing import Optional



# class CriticData:
#     """
#     Unified data structure for all MPPI critics.
#     """

#     # Trajectories
#     trajectories: torch.Tensor      # [K, T, 3] - Sampled trajectories (x, y, theta)
#     v_samples: torch.Tensor         # [K, T] - Linear velocity samples
#     w_samples: torch.Tensor         # [K, T] - Angular velocity samples

#     # Path state
#     path: torch.Tensor              # [P, 3] - Global path waypoints (x,y, theta)
#     goal: torch.Tensor              # [2] - Final goal position (XY)
#     goal_heading: Optional[torch.Tensor] = None        # Final goal heading (from path[-1,2]) 

#     # PathTracker state (monotonic progress tracking)
#     furthest_reached_idx: int       # Furthest point reached on path (never decreases!)
#     path_valid_flags: torch.Tensor  # [P] - bool, True if path point not blocked
#     path_integrated_distances: torch.Tensor  # [P] Cumulative distance along path (XY ONLY)
#     local_path_length:float         # Remaining path distance to goal (XY only)

#     # Robot State
#     current_pose : torch.Tensor     # [3] - Curret robot pose (x, y, theta)

#     # Perception
#     costmap: torch.Tensor           # [H, W] - STVL costmap (values 0.0 to 1.0)
#     grid_origin: torch.tensor       # [2] - Costmap origin in world frame (robot-centric!)

#     # Time
#     dt: float                       # Timestep duration (seconds)

#     # Robot Limits
#     v_max: float                # MAX vel
#     v_min: float  # Allow reversing # MIN vel
#     w_max: float                  # MAX angular vel

@dataclass
class CriticData:
    """
    Unified data structure for all MPPI critics.
    Matches Nav2's CriticData architecture.
    
    Field names chosen to match PathTracker attributes exactly:
    - path_valid_flags (NOT path_pts_valid)
    - goal_heading as float (NOT tensor)
    """

    # =========================================================================
    # REQUIRED FIELDS (no defaults)
    # =========================================================================
    
    # Trajectories
    trajectories: torch.Tensor      # [K, T, 3] - Sampled trajectories (x, y, theta)
    v_samples: torch.Tensor         # [K, T] - Linear velocity samples
    w_samples: torch.Tensor         # [K, T] - Angular velocity samples

    # Path & Goal
    path: torch.Tensor              # [N, 3] - Global path waypoints (x, y, theta)
    goal: torch.Tensor              # [2] - Final goal position (x, y)

    # Robot State
    current_pose: torch.Tensor      # [3] - Current robot pose (x, y, theta)

    # Perception (Costmap)
    costmap: torch.Tensor           # [H, W] - STVL costmap (values 0.0 to 1.0)
    grid_origin: torch.Tensor       # [2] - Costmap origin in world frame

    # Time & Constraints
    dt: float                       # Timestep duration (seconds)
    v_max: float                    # Max linear velocity (m/s)
    v_min: float                    # Min linear velocity (m/s)
    w_max: float                    # Max angular velocity (rad/s)

    # =========================================================================
    # OPTIONAL FIELDS (with defaults) - Must come after required fields!
    # =========================================================================
    
    # Goal heading (SCALAR float from path, not tensor!)
    goal_heading: float = 0.0       # ✅ Changed from Optional[torch.Tensor]

    # PathTracker State (field names match PathTracker exactly!)
    furthest_reached_idx: int = 0
    path_valid_flags: Optional[torch.Tensor] = None  # ✅ Changed from path_pts_valid
    path_integrated_distances: Optional[torch.Tensor] = None

    # State Flags
    fail_flag: bool = False
    local_path_length: float = 0.0  # ✅ This is a VALUE (not method call)

