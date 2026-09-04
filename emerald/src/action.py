from typing import Dict, Any, Tuple, List

import numpy as np
import math


class EmeraldLookupAction:
    """
    World-famous discrete action parser which uses a lookup table to reduce the number of possible actions from 1944 to 102
    """

    def __init__(self, tick_skip: int = 8, include_wall_act: bool = False):
        super().__init__()
        self._lookup_table = self.make_lookup_table(tick_skip, include_wall_act)
        self.tick_skip = tick_skip

    def parse_actions(self, actions: Dict[int, int], mirrored: Dict[int, bool]) -> Dict[int, np.ndarray]:
        parsed_actions = {}
        for i, action in actions.items():
            parsed_action = self._lookup_table[action]

            if (mirrored[i]):
                parsed_action = parsed_action.copy()
                parsed_action[:, 1] *= -1
                parsed_action[:, 3] *= -1
                parsed_action[:, 4] *= -1

            parsed_actions[i] = parsed_action

        return parsed_actions

    @staticmethod
    def make_lookup_table(tick_skip: int, include_wall_act: bool):
        actions = []
        # Ground
        for throttle in (-1, 0, 1):
            for steer in (-1, 0, 1):
                for boost in (0, 1):
                    for handbrake in (0, 1):
                        if boost == 1 and throttle != 1:
                            continue
                        actions.append(
                            np.repeat([[throttle, steer, 0, steer, 0, 0, boost, handbrake]], tick_skip, axis=0)
                        )
        # Aerial
        for pitch in (-1, 0, 1):
            for yaw in (-1, 0, 1):
                for roll in (-1, 0, 1):
                    for jump in (0, 1):
                        for boost in (0, 1):
                            if jump == 1 and yaw != 0:  # Only need roll for sideflip
                                continue
                            if pitch == roll == jump == 0:  # Duplicate with ground
                                continue
                            # Enable handbrake for potential wavedashes
                            handbrake = jump == 1 and (pitch != 0 or yaw != 0 or roll != 0)
                            actions.append(
                                np.repeat([[boost, yaw, pitch, yaw, roll, jump, boost, handbrake]], tick_skip, axis=0)
                            )
        
        # Wall
        if (include_wall_act):
            for steer1 in (-1, 0, 1):
                for steer2 in (-1, 0, 1):
                    if steer1 == steer2:
                        continue
                    for boost in (0, 1):
                        actions.append(np.vstack([
                            np.repeat([[1, steer1, 0, steer1, 0, 0, boost, 0]], math.floor(tick_skip / 2), axis=0),
                            np.repeat([[1, steer2, 0, steer2, 0, 1, boost, 0]], math.ceil(tick_skip / 2), axis=0)
                        ]))

        return np.array(actions, dtype=np.float32)