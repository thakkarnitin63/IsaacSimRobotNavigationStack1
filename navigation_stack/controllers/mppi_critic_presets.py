# In: navigation_stack/controllers/mppi_critic_presets.py

"""Preset configurations for different navigation scenarios"""

# BALANCED (default) - Works well in most scenarios
BALANCED_WEIGHTS = {
    "cost_critic_weight": 50.0,
    "obstacles_critic_weight": 50.0,
    "path_tracking_weight": 10.0,
    "path_angle_weight": 5.0,
    "path_follow_weight": 8.0,
    "goal_weight": 15.0,
    "goal_angle_weight": 8.0,
    "constraint_weight": 10.0,
    "smoothness_weight": 5.0,
    "twirling_weight": 15.0,
    "prefer_forward_weight": 5.0,
}

# CAUTIOUS - Very conservative, avoids obstacles aggressively
CAUTIOUS_WEIGHTS = {
    "cost_critic_weight": 100.0,        # ↑ Double obstacle avoidance
    "obstacles_critic_weight": 100.0,   # ↑ Double obstacle avoidance
    "path_tracking_weight": 8.0,        # ↓ Less strict on path
    "path_angle_weight": 4.0,
    "path_follow_weight": 6.0,
    "goal_weight": 12.0,
    "goal_angle_weight": 6.0,
    "constraint_weight": 15.0,          # ↑ Stricter limits
    "smoothness_weight": 8.0,           # ↑ Smoother motion
    "twirling_weight": 20.0,            # ↑ Less rotation
    "prefer_forward_weight": 3.0,
}

# AGGRESSIVE - Fast navigation, less cautious
AGGRESSIVE_WEIGHTS = {
    "cost_critic_weight": 30.0,         # ↓ Less obstacle paranoia
    "obstacles_critic_weight": 30.0,    # ↓ Less obstacle paranoia
    "path_tracking_weight": 15.0,       # ↑ Follow path strictly
    "path_angle_weight": 8.0,           # ↑ Point toward goal
    "path_follow_weight": 10.0,
    "goal_weight": 20.0,                # ↑ Rush to goal
    "goal_angle_weight": 5.0,
    "constraint_weight": 5.0,           # ↓ Less strict limits
    "smoothness_weight": 3.0,           # ↓ Allow jerkier motion
    "twirling_weight": 10.0,
    "prefer_forward_weight": 8.0,       # ↑ Strongly prefer forward
}

# SMOOTH - Prioritize comfort over speed
SMOOTH_WEIGHTS = {
    "cost_critic_weight": 50.0,
    "obstacles_critic_weight": 50.0,
    "path_tracking_weight": 10.0,
    "path_angle_weight": 5.0,
    "path_follow_weight": 8.0,
    "goal_weight": 15.0,
    "goal_angle_weight": 8.0,
    "constraint_weight": 12.0,
    "smoothness_weight": 15.0,          # ↑↑ Very smooth motion
    "twirling_weight": 20.0,            # ↑ Minimal rotation
    "prefer_forward_weight": 5.0,
}

# ANTI_SWIRL - For fixing swirling/spinning issues
ANTI_SWIRL_WEIGHTS = {
    "cost_critic_weight": 50.0,
    "obstacles_critic_weight": 50.0,
    "path_tracking_weight": 15.0,       # ↑ Follow path more
    "path_angle_weight": 10.0,          # ↑↑ Point toward target!
    "path_follow_weight": 12.0,
    "goal_weight": 15.0,
    "goal_angle_weight": 8.0,
    "constraint_weight": 10.0,
    "smoothness_weight": 10.0,          # ↑ Reduce oscillation
    "twirling_weight": 30.0,            # ↑↑ CRITICAL: Stop spinning!
    "prefer_forward_weight": 8.0,
}