from dataclasses import dataclass, MISSING


@dataclass
class TaskConfig:
    max_steps: int = MISSING
    map_size: list = MISSING
    obs_space_dim: int = MISSING
    action_space_dim: int = MISSING
    agents: list = MISSING
    targets: list = MISSING
    shuffle_agents_positions: bool = MISSING

