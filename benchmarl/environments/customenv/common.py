#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from typing import Callable, Dict, List, Optional

from benchmarl.environments.common import Task
from benchmarl.utils import DEVICE_TYPING

from torchrl.envs import EnvBase
from torchrl.data import Composite

# from torchrl.envs.libs import YourTorchRLEnvConstructor
from vmas import make_env

# from .domain.salp_domain import SalpDomain
# from .domain.create_env import create_env
# from vmas_salp.domain.salp_domain import SalpDomain
from vmas_salp.domain.create_env import create_env
from pathlib import Path


class CustomEnvTask(Task):
    # Your task names.
    # Their config will be loaded from conf/task/customenv

    TASK_1 = None  # Loaded automatically from conf/task/customenv/task_1
    # TASK_2 = None  # Loaded automatically from conf/task/customenv/task_2

    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
        device: DEVICE_TYPING,
    ) -> Callable[[], EnvBase]:
        # return lambda: YourTorchRLEnvConstructor(
        #     scenario=self.name.lower(),
        #     num_envs=num_envs,  # Number of vectorized envs (do not use this param if the env is not vectorized)
        #     continuous_actions=continuous_actions,  # Ignore this param if your env does not have this choice
        #     seed=seed,
        #     device=device,
        #     categorical_actions=True,  # If your env has discrete actions, they need to be categorical (TorchRL can help with this)
        #     **self.config,  # Pass the loaded config (this is what is in your yaml
        # )
        # return lambda: make_env(
        #                 scenario=self.name.lower,
        #                 num_envs=num_envs,
        #                 device=device,
        #                 continuous_actions=continuous_actions,
        #                 seed=seed,
        #                 # dict_spaces=dict_spaces,
        #                 wrapper=None,
        #                 seed=None,
        #                 # Environment specific variables
        #                 # **kwargs,
        #                 **self.config,
        #             )
        here = Path(__file__).resolve()
        bm = here.parents[2]
        return lambda: create_env(
            batch_dir=bm / "conf" / "task" / "customenv",
            n_envs=num_envs,
            device=device,
            seed=seed,
            benchmark=True,
        )

    def supports_continuous_actions(self) -> bool:
        return True

    def supports_discrete_actions(self) -> bool:
        return True

    def has_render(self, env: EnvBase) -> bool:
        return True

    def max_steps(self, env: EnvBase) -> int:
        return self.config["max_steps"]

    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        if hasattr(env, "group_map"):
            return env.group_map
        return {"agents": [agent.name for agent in env.agents]}

    def state_spec(self, env: EnvBase) -> Optional[Composite]:
        return None

    def action_mask_spec(self, env: EnvBase) -> Optional[Composite]:
        return None

    def observation_spec(self, env: EnvBase) -> Composite:
        observation_spec = env.unbatched_observation_spec.clone()
        for group in self.group_map(env):
            if "info" in observation_spec[group]:
                del observation_spec[(group, "info")]
        return observation_spec

    def info_spec(self, env: EnvBase) -> Optional[Composite]:
        info_spec = env.unbatched_observation_spec.clone()
        for group in self.group_map(env):
            del info_spec[(group, "observation")]
        for group in self.group_map(env):
            if "info" in info_spec[group]:
                return info_spec
        else:
            return None

    def action_spec(self, env: EnvBase) -> Composite:
        return env.unbatched_action_spec

    @staticmethod
    def env_name() -> str:
        return "customenv"
