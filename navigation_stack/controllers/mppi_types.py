import torch
from dataclasses import dataclass 


@dataclass
class MPPIConfig:
    """All hyperparameters in one place"""
    # Sampling 
    num_samples: int = 2000 # K: Number of Trajectory Samples
    horizon_steps: int = 56 # T : Planning horizon (timesteps)
    dt: float = 0.05 # 20Hz control   # deltaT: timestep duration (seconds)
    grid_resolution: float = 0.1 # COSTMAP RESOLUTION

    # Robot Limits
    v_max: float = 0.5                # MAX vel
    v_min: float = -0.35 # Allow reversing # MIN vel
    w_max: float = 1.0                  # MAX angular vel

    # Noise (exploration)
    v_std: float = 0.1           # Linear Velocity noise std dev
    w_std: float = 0.2              # Angular Velocity noise std dev
 

    # Temperature (lower = more greedy)
    lambda_: float = 0.3 

    gamma: float = 0.015          

    

    # Device 
    device: str = "cuda:0"

    @property
    def planning_horizon_seconds(self)-> float:
        """Total planning time in seconds"""
        return self.horizon_steps * self.dt

class ControlSequence:
    """
    A Sequence of controls over the planning horizon.
    This is what MPPI optimizes. each timestep has a linear velocity (vx)
    and angular velocity (wz)
    vx: velocity in X direction (body frame)
    wz: angular velocity around Z axis (yaw rate)
    """
    def __init__(self, T:int, device: str):
        """
        T : Number of timesteps (horizon length)
        device : Pytorch device (cuda:0)
        """
        self.T =T
        self.device = device 

        # Init with Zeros
        self.vx = torch.zeros(T, device=device) # Linear vel
        self.wz = torch.zeros(T, device=device) # Angular vel

    def shift(self):
        """Warm--start : Shift left and repeat last so after choosing first control
        we dont dispose the rest of calcualted time step just compute last one"""
        
        if self.T <2:
            return # Nothing Shift
        
        # Shift left (roll by -1)
        self.vx = torch.roll(self.vx, shifts=-1, dims=0)
        self.wz = torch.roll(self.wz, shifts=-1, dims=0)

        # Repeat last control dont let it zero
        self.vx[-1] = self.vx[-2]
        self.wz[-1] = self.wz[-2]

    def initalize_with_forward_motion(self, v:float = 0.2):
        """
        Initialize sequence with constant forward motion
        """
        self.vx[:] = v
        self.wz[:] = 0.0

    def clamp(self, v_min: float, v_max: float, w_max: float):
        """Enforce Control Limits"""
        self.vx = torch.clamp(self.vx, v_min, v_max)
        self.wz = torch.clamp(self.wz, -w_max, w_max)

    def get_first_command(self) -> tuple[float, float]:
        """
        Extract first control command to send to robot.
        Returns: (v,w) as Python Floats
        """
        return self.vx[0].item(), self.wz[0].item()
    
    def __repr__(self) -> str:
        return (f"ControlSequence(T={self.T}, "
                f"v_range=[{self.vx.min():.2f}, {self.vx.max():.2f}], "
                f"w_range=[{self.wz.min():.2f}, {self.wz.max():.2f}])")
    

@dataclass
class MPPIState:
    """ Robot state at one instant"""

    x: float
    y: float
    theta: float
    v: float # Current Velocity
    w: float # Current Angular Velocity

    def to_tensor(self, device:str) -> torch.Tensor:
        """Convert to Pytorch tensor [x, y, theta, v, w]"""
        return torch.tensor(
            [self.x, self.y, self.theta, self.v, self.w],
            dtype=torch.float32,
            device=device
        )
