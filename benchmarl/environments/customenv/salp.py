from dataclasses import dataclass, MISSING


@dataclass
class TaskConfig:
    max_steps: int = MISSING
    map_size: list = MISSING
    use_order: bool = MISSING
    obs_space_dim: int = MISSING
    action_space_dim: int = MISSING
    rovers: list = MISSING
    pois: list = MISSING
