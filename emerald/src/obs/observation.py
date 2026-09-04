from rlbot import flat

import numpy as np
from dataclasses import dataclass
from typing import Optional
import random

from .physics import Physics
from .common_values import *



POS_COEF = 1 / np.array((SIDE_WALL_X, BACK_WALL_Y, CEILING_Z), dtype=np.float32)
VEL_COEF = 1 / np.float32(CAR_MAX_SPEED)
ANG_VEL_COEF = 1 / np.float32(CAR_MAX_ANG_VEL)

PAD_TIMER_COEF = 1 / np.repeat((4, 10, 4, 10, 4, 10, 4, 10, 4), (3, 2, 10, 1, 2, 1, 10, 2, 3))

FAKE_PLAYER = flat.PlayerInfo()

fake = Physics()
fake.pos = np.array((0, 5320, 17.01), dtype=np.float32)
fake.vel = fake.angvel = np.zeros(3, dtype=np.float32)
fake.rot_mtx = np.array((
    (0, -1, 0),
    (1, 0, 0),
    (0, 0, 1)
), dtype=np.float32)
FAKE_PHYS_TM = fake
FAKE_PHYS_OPP = fake.invert()



def mirror_pads(pads: np.ndarray):
    a = pads.copy()
    a[1], a[2] = a[2], a[1]
    a[3], a[4] = a[4], a[3]
    a[5], a[6] = a[6], a[5]
    a[8], a[9] = a[9], a[8]
    a[10], a[11] = a[11], a[10]
    a[12], a[14] = a[14], a[12]
    a[15], a[18] = a[18], a[15]
    a[16], a[17] = a[17], a[16]
    a[19], a[21] = a[21], a[19]
    a[22], a[23] = a[23], a[22]
    a[24], a[25] = a[25], a[24]
    a[27], a[28] = a[28], a[27]
    a[29], a[30] = a[30], a[29]
    a[31], a[32] = a[32], a[31]
    return a

class EmeraldObs:
    def __init__(self,
                 min_size: int = 1,
                 max_size: int = 3,
                 pred_slices: Optional[list[int]] = None,
                 pred_before_player: bool = False,
                 mirror: bool = False,
                 pads_inf_boost: bool = False,
                 inf_boost_indicator: bool = False,
                 use_v4_data: bool = True,
                 teammate_indicator: bool = False,
                 demo_indicator: bool = False):
        self.min_size = min_size
        self.max_size = max_size
        self.pred_slices = pred_slices
        self.pred_before_player = pred_before_player
        self.mirror = mirror
        self.pads_inf_boost = pads_inf_boost
        self.inf_boost_indicator = inf_boost_indicator
        self.use_v4_data = use_v4_data
        self.teammate_indicator = teammate_indicator
        self.demo_indicator = demo_indicator

        self.n = (min_size * 2) + 1 # extra 0 slot for fake players
        self.prev_boosts = [0] * self.n
        self.jump_times = [0] * self.n
        self.air_times = [0] * self.n # does NOT include time jumping
        self.flip_times = [0] * self.n


        base_size = 51
        if (self.mirror):
            base_size += 1
        if (self.inf_boost_indicator):
            base_size += 1

        ind_car_size = 34 if self.use_v4_data else 36
        other_car_size = ind_car_size + 18
        if (self.teammate_indicator):
            other_car_size += 1

        pred_size = 0 if self.pred_slices is None else len(self.pred_slices) * 9

        self.obs_size = base_size + ind_car_size + pred_size + ((self.max_size * 2 - 1) * other_car_size)
        self.zp_size = other_car_size

    def init(self, infinite_boost: bool, non_standard: bool):
        self.infinite_boost = infinite_boost
        self.is_non_standard = non_standard

    def update(self, packet: flat.GamePacket, time_elapsed: float):
        n_players = len(packet.players)
        if (self.n <= n_players):
            addition = [0] * (n_players - self.n + 1)
            self.n = n_players + 1

            self.prev_boosts = self.prev_boosts + addition
            self.jump_times = self.jump_times + addition
            self.air_times = self.air_times + addition
            self.flip_times = self.flip_times + addition

        for i, player in enumerate(packet.players):
            player = packet.players[i]
            
            if (player.air_state == flat.AirState.Jumping):
                self.jump_times[i] += time_elapsed
            elif (player.has_jumped and player.air_state in {flat.AirState.Dodging, flat.AirState.DoubleJumping, flat.AirState.InAir}):
                self.jump_times[i] += time_elapsed
                self.air_times[i] += time_elapsed
            else:
                self.jump_times[i] = 0
                self.air_times[i] = 0

    def update_boosts(self, packet: flat.GamePacket):
        for i, player in enumerate(packet.players):
            self.prev_boosts[i] = player.boost

    def build_obs(self, packet: flat.GamePacket, controllers: dict[int, flat.ControllerState], indices: list[int], ball_pred: flat.BallPrediction) -> tuple[dict[int, np.ndarray], dict[int, bool]]:
        player_obs = {}
        player_mirrored = {}

        # physics handling could be made better (probably)
        # TODO hivemind process spawns once for each *team*, so only one
        # of each list is used at all i think
        nor_player_phys = [Physics(player.physics) for player in packet.players]
        inv_player_phys = [phys.invert() for phys in nor_player_phys]
        nor_balls = [Physics(ball.physics) for ball in packet.balls]
        inv_balls = [phys.invert() for phys in nor_balls]

        if (self.infinite_boost or self.is_non_standard):
            nor_pads = np.ones(34, dtype=np.float32) if self.pads_inf_boost else np.zeros(34, dtype=np.float32)
        else:
            p = [pad.timer for pad in packet.boost_pads]
            nor_pads = np.array(p, dtype=np.float32) * PAD_TIMER_COEF

        inv_pads = np.flip(nor_pads)

        nor_pred_balls = []
        if (self.pred_slices is not None):    
            for i in self.pred_slices:
                pred_slice = ball_pred.slices[i]
                nor_pred_balls.append(Physics(pred_slice.physics))
        inv_pred_balls = [phys.invert() for phys in nor_pred_balls]

        for idx in indices:
            player = packet.players[idx]
            is_blue = player.team == BLUE_TEAM

            if (is_blue):
                player_phys = nor_player_phys
                balls = nor_balls
                pads = nor_pads
                pred_balls = nor_pred_balls
            else:
                player_phys = inv_player_phys
                balls = inv_balls
                pads = inv_pads
                pred_balls = inv_pred_balls

            physics = player_phys[idx]
            ball = self._get_car_ball(physics, balls)

            controller = controllers[idx]
            prev_action = [
                controller.throttle,
                controller.steer,
                controller.pitch,
                controller.yaw,
                controller.roll,
                controller.jump,
                controller.boost,
                controller.handbrake,
            ]

            obs = []

            if (self.mirror):
                mirrored = physics.pos[0] < 0
                if (mirrored):
                    player_phys = [phys.mirror() for phys in player_phys]
                    ball = ball.mirror()
                    pads = mirror_pads(pads)
                    pred_balls = [phys.mirror() for phys in pred_balls]
                    physics = player_phys[idx]

                    prev_action[1] *= -1
                    prev_action[3] *= -1
                    prev_action[4] *= -1

                obs.append([mirrored and packet.match_info.match_phase == flat.MatchPhase.Kickoff])
            else:
                mirrored = False

            player_mirrored[idx] = mirrored
                    
            if (self.inf_boost_indicator):
                obs.append([self.infinite_boost])
            
            obs.extend([
                ball.pos * POS_COEF,
                ball.vel * VEL_COEF,
                ball.angvel * ANG_VEL_COEF,
                pads,
                prev_action
            ])

            if (not self.pred_before_player):
                obs.extend(self._build_car_obs(player, physics, ball, idx))

            for pred_slice in pred_balls:
                obs.extend([
                    pred_slice.pos * POS_COEF,
                    ((pred_slice.pos - physics.pos) @ physics.rot_mtx) * POS_COEF,
                    pred_slice.vel * VEL_COEF
                ])

            if (self.pred_before_player):
                obs.extend(self._build_car_obs(player, physics, ball, idx))

            teammates = []
            opponents = []

            for i, other_player in enumerate(packet.players):
                if (i == idx):
                    continue

                other_phys = player_phys[i]
                if (player.team == other_player.team):
                    if (len(teammates) == (self.max_size - 1)):
                        continue
                    team_obs = teammates
                else:
                    if (len(opponents) == self.max_size):
                        continue
                    team_obs = opponents

                team_obs.append(self._build_car_obs(other_player, other_phys, ball, i, physics))

            while (len(teammates) < self.min_size - 1):
                teammates.append(self._build_car_obs(FAKE_PLAYER, FAKE_PHYS_TM, ball, -1, physics))
            while (len(opponents) < self.min_size):
                opponents.append(self._build_car_obs(FAKE_PLAYER, FAKE_PHYS_OPP, ball, -1, physics))

            if (self.max_size > 1):
                zeros = np.zeros((1, self.zp_size), dtype=np.float32)
                while (len(teammates) < self.max_size - 1):
                    teammates.append(zeros)
                while (len(opponents) < self.max_size):
                    opponents.append(zeros)

            random.shuffle(teammates)
            random.shuffle(opponents)

            for p in teammates:
                obs.extend(p)
            for p in opponents:
                obs.extend(p)

            player_obs[idx] = np.concatenate(obs)

        return player_obs, player_mirrored



    def _get_car_ball(self, player: Physics, balls: list[Physics]) -> Physics:
        if (len(balls) == 1):
            return balls[0]

        ball = None

        for p_ball in balls:
            dist = np.linalg.norm(p_ball.pos - player.pos)
            if (ball is not None):
                if (dist >= ball_dist):
                    continue

            ball = p_ball
            ball_dist = dist
        return ball


    def _build_car_obs(self, player: flat.PlayerInfo, physics: Physics, ball: Physics, idx: int, main_phys: Optional[Physics] = None):
        if (player.demolished_timeout != -1 and self.demo_indicator):
            obs = [[0] * 27]
        else:
            ball_pos_diff = ball.pos - physics.pos
            ball_vel_diff = ball.vel - physics.vel

            obs = [
                physics.pos * POS_COEF,
                physics.vel * VEL_COEF,
                (physics.angvel @ physics.rot_mtx) * ANG_VEL_COEF,
                physics.forward,
                physics.up,

                ball_pos_diff * POS_COEF,
                (ball_pos_diff @ physics.rot_mtx) * POS_COEF,

                ball_vel_diff * VEL_COEF,
                (ball_vel_diff @ physics.rot_mtx) * VEL_COEF,
            ]

        
        obs.append([
            1 if self.infinite_boost else player.boost * 0.01,
            max(0, player.demolished_timeout * (1/3)),
            player.air_state == flat.AirState.OnGround,            
            player.has_jumped
        ])

        # is_boosting = 0 < player.boost < self.prev_boosts[idx]
        is_boosting = player.last_input.boost and (self.infinite_boost or player.boost > 0)

        if (self.use_v4_data):
            time_since_jump = self.jump_times[idx]
            obs.append([
                time_since_jump/(time_since_jump + 1),
                player.has_double_jumped or player.has_dodged,
                is_boosting
            ])

        else:
            time_since_jump = self.air_times[idx]
            flip_time = player.dodge_elapsed * (1/0.95) if player.dodge_elapsed <= 0.95 else 0

            obs.append([
                player.has_double_jumped or player.has_dodged,
                flip_time,
                
                min(DOUBLEJUMP_MAX_DELAY, time_since_jump),
                time_since_jump/(time_since_jump + 1),
                is_boosting
            ])

        if (main_phys is None):
            return obs

        pos_diff = physics.pos - main_phys.pos
        vel_diff = physics.vel - main_phys.vel
        rot_mtx = main_phys.rot_mtx

        obs.extend([
            pos_diff * POS_COEF,
            (pos_diff @ rot_mtx) * POS_COEF,

            vel_diff * VEL_COEF,
            (vel_diff @ rot_mtx) * VEL_COEF,

            physics.forward @ rot_mtx,
            physics.up @ rot_mtx,
        ])

        if (self.teammate_indicator):
            obs.append([1])

        return obs
