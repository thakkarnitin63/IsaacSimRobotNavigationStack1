import numpy as np
from isaacsim.storage.native import get_assets_root_path
import carb

try:
    from isaacsim.robot.policy.examples.robots.h1 import H1FlatTerrainPolicy
    H1_AVAILABLE = True
except ImportError:
    H1_AVAILABLE = False
    carb.log_warn("H1FlatTerrainPolicy not available")

class H1Humanoid:
    """Simple H1 humanoid that walks straight - based on working example."""
    
    def __init__(self, world, spawn_position, walk_distance=3.0):
        self.world = world
        self.spawn_position = np.array(spawn_position)
        self.h1_policy = None
        self._physics_ready = False
        
        # Simple straight walk parameters
        self._step_count = 0
        self._forward_speed = 0.4  # Slower than 0.75 m/s
        self._walk_distance = walk_distance
        
        # Calculate how many steps needed
        physics_dt = 1.0 / 200.0
        time_needed = self._walk_distance / self._forward_speed
        self._target_steps = int(time_needed / physics_dt)
        
        print(f"H1 will walk {self._walk_distance:.2f}m at {self._forward_speed}m/s")
        print(f"This will take {time_needed:.2f} seconds ({self._target_steps} steps)")
        
    def spawn(self):
        """Spawn the H1 robot."""
        if not H1_AVAILABLE:
            carb.log_error("H1 robot not available")
            return False
        
        try:
            assets_root = get_assets_root_path()
            if not assets_root:
                carb.log_error("Assets path not found")
                return False
            
            print(f"Spawning H1 humanoid at {self.spawn_position}...")
            
            self.h1_policy = H1FlatTerrainPolicy(
                prim_path="/World/H1_Walker",
                name="H1_Walker",
                usd_path=assets_root + "/Isaac/Robots/Unitree/H1/h1.usd",
                position=self.spawn_position,
            )
            
            print(f"✓ H1 humanoid spawned")
            return True
            
        except Exception as e:
            carb.log_error(f"Failed to spawn H1: {e}")
            return False
    
    def on_physics_step(self, step_size):
        """
        Physics callback - exactly like working example.
        """
        if not self.h1_policy:
            return
        
        if self._physics_ready:
            # Default: stop
            command = [0.0, 0.0, 0.0]
            
            # If haven't reached target, walk forward
            if self._step_count < self._target_steps:
                command = [self._forward_speed, 0.0, 0.0]
                self._step_count += 1
            
            # Send command to H1 policy
            self.h1_policy.forward(step_size, command)
            
            # Print once when reached
            if self._step_count == self._target_steps:
                print(f"   🤖 H1 finished walking {self._walk_distance:.2f}m")
                self._step_count += 1  # Increment to avoid re-printing
                
        else:
            # First physics step - initialize
            print("Initializing H1 robot policy...")
            self._physics_ready = True
            self.h1_policy.initialize()
            self.h1_policy.post_reset()
            self.h1_policy.robot.set_joints_default_state(self.h1_policy.default_pos)
    
    def on_timeline_event(self, event):
        """Reset when timeline is played/stopped."""
        if self.h1_policy:
            self._physics_ready = False