"""
File: discrete_policy.py
Author: Matthew Allen

Description:
    An implementation of a feed-forward neural network which parametrizes a discrete distribution over a space of actions.
"""


from torch.distributions import Categorical
import torch.nn as nn
import torch
import numpy as np


class DiscreteFF(nn.Module):
    def __init__(self, input_shape, n_actions, layer_sizes, device, use_leakyRELU: bool = False):
        super().__init__()
        self.device = device

        assert len(layer_sizes) != 0, "AT LEAST ONE LAYER MUST BE SPECIFIED TO BUILD THE NEURAL NETWORK!"
        activation = nn.LeakyReLU() if use_leakyRELU else nn.ReLU()
        layers = [nn.Linear(input_shape, layer_sizes[0]), activation]
        prev_size = layer_sizes[0]
        for size in layer_sizes[1:]:
            layers.append(nn.Linear(prev_size, size))
            layers.append(activation)
            prev_size = size

        layers.append(nn.Linear(layer_sizes[-1], n_actions))
        layers.append(nn.Softmax(dim=-1))
        self.model = nn.Sequential(*layers).to(self.device)

        self.n_actions = n_actions


    def get_action(self, obs, deterministic=True):
        """
        Function to get an action and the log of its probability from the policy given an observation.
        :param obs: Observation to act on.
        :param deterministic: Whether the action should be chosen deterministically.
        :return: Chosen action and its logprob.
        """

        obs = torch.from_numpy(obs).to(self.device)
        probs = self.model(obs)
        probs = probs.view(-1, self.n_actions)
        probs = torch.clamp(probs, min=1e-11, max=1)

        if deterministic:
            return probs.cpu().numpy().argmax(axis=-1)

        action = torch.multinomial(probs, 1, True)
        return action.flatten().cpu()