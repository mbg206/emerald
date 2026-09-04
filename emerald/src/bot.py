from __future__ import annotations

from rlbot import flat
from rlbot.managers import Hivemind
import sys

from vers import VERSIONS, EmeraldVersion

import numpy as np
import traceback


def toss(err, packet = None):
    with open("error_log.txt", "w") as sys.stdout:
        if (packet is not None):
            print("ERROR DURING PACKET HANDLING")
            print(packet)
            print()
        else:
            print("ERROR DURING INIT\n")
        traceback.print_exception(err, file=sys.stdout)
    exit(99)

class Emeralds(Hivemind):
    def __init__(self, version: EmeraldVersion):
        super().__init__()

        self.agent = version.agent
        self.agent.init(version.obs_builder, version.action_parser, version.policy_path)
        self.obs_builder = version.obs_builder
        self.action_idx = 999
        self.last_tick = 0
        self.tick_skip = version.action_parser.tick_skip

    
    def initialize(self):
        mutators = self.match_config.mutators
        if (mutators is None):
            inf_boost = False
        else:
            inf_boost = mutators.boost_amount == flat.BoostAmountMutator.UnlimitedBoost
        self.obs_builder.init(
            inf_boost,
            len(self.field_info.boost_pads) != 34
        )

        self.actions: dict[int, np.ndarray] = {i: np.zeros(8) for i in self.indices}
        self.controllers: dict[int, flat.ControllerState] = {i: flat.ControllerState() for i in self.indices}

    def get_outputs(self, packet: flat.GamePacket) -> dict[int, flat.ControllerState]:
        try:
            return self._get_outputs(packet)
        except Exception as e:
            toss(e, packet)

    def _get_outputs(self, packet: flat.GamePacket) -> dict[int, flat.ControllerState]:
        tick_diff = packet.match_info.frame_num - self.last_tick
        self.last_tick = packet.match_info.frame_num
        self.action_idx += round(tick_diff)

        self.obs_builder.update(packet, tick_diff * (1/120))

        if (
            packet.match_info.match_phase
            not in {
                flat.MatchPhase.Active,
                flat.MatchPhase.Kickoff,
                flat.MatchPhase.Countdown
            }
            or len(packet.balls) == 0
        ):
            return self.controllers

        if (self.action_idx >= self.tick_skip):
            obs, mirrored = self.obs_builder.build_obs(packet, self.controllers, self.indices, self.ball_prediction)
            self.actions = self.agent.act(obs, mirrored)

            self.action_idx = 0


        for i, act in self.actions.items():
            controller = self.controllers[i]
            c_act = act[self.action_idx]
            controller.throttle = c_act[0]
            controller.steer = c_act[1]
            controller.pitch = c_act[2]
            controller.yaw = c_act[3]
            controller.roll = c_act[4]
            controller.jump = c_act[5] > 0
            controller.boost = c_act[6] > 0
            controller.handbrake = c_act[7] > 0

        self.obs_builder.update_boosts(packet)

        return self.controllers

    def _render_pred(self):
        pts = []
        
        for pred in self.ball_prediction.slices:
            lo = pred.physics.location
            pts.append(flat.Vector3(lo.x, lo.y, lo.z))

        self.renderer.begin_rendering()
        self.renderer.draw_polyline_3d(pts, self.renderer.purple)
        self.renderer.end_rendering()



def run(ver: str):
    try:
        version = VERSIONS[ver]()
        bot = Emeralds(version)
        bot.initialize()

        bot.run()
    except Exception as e:
        toss(e)

if __name__ == "__main__":
    run(sys.argv[1])