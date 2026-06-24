"""SoftVG-PINN: soft-layer VG-constrained PINN for layered soil water flow."""
from .config import TrainingConfig, load_config, save_config, set_seed
from .model import SoilPINN

__all__ = ["TrainingConfig", "load_config", "save_config", "set_seed", "SoilPINN"]
__version__ = "0.2.0"
