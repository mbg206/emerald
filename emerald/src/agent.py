import torch
import numpy as np
from os import path, getcwd

from action import EmeraldLookupAction
from policy.discrete import DiscreteFF
from obs.observation import EmeraldObs

DETERMINISTIC = True

class EmeraldAgentPPO:
    def __init__(self, layer_sizes: tuple[int], use_leakyRELU: bool = False) -> None:
        self.layer_sizes = layer_sizes
        self.use_leakyRELU = use_leakyRELU

    def init(self, obs_builder: EmeraldObs, action_parser: EmeraldLookupAction, policy_path: str):
        self.action_parser = action_parser

        device = torch.device("cpu")
        self.policy = DiscreteFF(obs_builder.obs_size, len(self.action_parser._lookup_table), self.layer_sizes, device, self.use_leakyRELU)
        
        with open(path.join(getcwd(), "policy", policy_path), "rb") as file:
            self.policy.load_state_dict(torch.load(file, map_location=device))

        torch.set_num_threads(1)
    
    def act(self, observations: dict[int, np.ndarray], mirrored: dict[int, bool]) -> np.ndarray:
        indices = []
        first_obs = next(iter(observations.values()))
        full_obs = np.empty((len(observations), len(first_obs)), dtype=np.float32)

        for i, (idx, obs) in enumerate(observations.items()):
            indices.append(idx)
            full_obs[i] = obs

        with torch.no_grad():
            action_idx = self.policy.get_action(full_obs, DETERMINISTIC)

        actions = {idx: action_idx[i] for i, idx in enumerate(indices)}
        
        return self.action_parser.parse_actions(actions, mirrored)