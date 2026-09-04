from dataclasses import dataclass
from typing import Callable

try:
    from action import EmeraldLookupAction
    from obs.observation import EmeraldObs
    from agent import EmeraldAgentPPO
except ImportError:
    class a:
        def __init__(self, *args, **kwargs):
            pass
    EmeraldLookupAction = EmeraldObs = EmeraldAgentPPO = a

TICKS_PER_SECOND = 120


REAL_T3_BOOST = False

@dataclass
class EmeraldVersion:
    policy_path: str
    agent: Callable
    action_parser: EmeraldLookupAction
    obs_builder: EmeraldObs


T4_PRED_SLICES = [
    int(   0.5    * TICKS_PER_SECOND - 1),
    int(   1      * TICKS_PER_SECOND - 1),
    int(   1.5    * TICKS_PER_SECOND - 1),
    int(   2      * TICKS_PER_SECOND - 1),
    int(   3      * TICKS_PER_SECOND - 1),
    int(   4      * TICKS_PER_SECOND - 1)
]
MAIN_PRED_SLICES = [
    int(   0.5    * TICKS_PER_SECOND - 1),
    int(   1      * TICKS_PER_SECOND - 1),
    int(   1.5    * TICKS_PER_SECOND - 1),
    int(   2      * TICKS_PER_SECOND - 1),
    int(   2.5    * TICKS_PER_SECOND - 1),
    int(   3      * TICKS_PER_SECOND - 1),
    int(   3.5    * TICKS_PER_SECOND - 1),
    int(   5      * TICKS_PER_SECOND - 1)
]

VERSIONS = {
    "t1": lambda: EmeraldVersion(
        policy_path="PPO_POLICY_T1.pt",
        agent=EmeraldAgentPPO((512, 256, 256)),
        action_parser=EmeraldLookupAction(),
        obs_builder=EmeraldObs(max_size=1)),



    "t2": lambda: EmeraldVersion(
        policy_path="PPO_POLICY_T2.pt",
        agent=EmeraldAgentPPO((1024, 512, 512, 512, 256)),
        action_parser=EmeraldLookupAction(),
        obs_builder=EmeraldObs(max_size=1)),
    "t2-a": lambda: EmeraldVersion(
        policy_path="PPO_POLICY_T2-A.pt",
        agent=EmeraldAgentPPO((1024, 512, 512, 512, 256)),
        action_parser=EmeraldLookupAction(),
        obs_builder=EmeraldObs(max_size=1)),
    "t2-b": lambda: EmeraldVersion(
        policy_path="PPO_POLICY_T2-B.pt",
        agent=EmeraldAgentPPO((1024, 512, 512, 512, 256)),
        action_parser=EmeraldLookupAction(),
        obs_builder=EmeraldObs(max_size=1)),
    "t2-c": lambda: EmeraldVersion(
        policy_path="PPO_POLICY_T2-C.pt",
        agent=EmeraldAgentPPO((1024, 512, 512, 512, 256)),
        action_parser=EmeraldLookupAction(),
        obs_builder=EmeraldObs(max_size=1)),
    "t2-d": lambda: EmeraldVersion(
        policy_path="PPO_POLICY_T2-D.pt",
        agent=EmeraldAgentPPO((1024, 512, 512, 512, 256)),
        action_parser=EmeraldLookupAction(),
        obs_builder=EmeraldObs(min_size=2, max_size=2)),
    "t2-myst": lambda: EmeraldVersion(
        policy_path="PPO_POLICY_T2-MYST.pt",
        agent=EmeraldAgentPPO((1024, 512, 512, 512, 256)),
        action_parser=EmeraldLookupAction(),
        obs_builder=EmeraldObs(max_size=1)),



    "t3": lambda: EmeraldVersion(
        policy_path="PPO_POLICY_T3.pt",
        agent=EmeraldAgentPPO((1024, 512, 512, 512, 256)),
        action_parser=EmeraldLookupAction(include_wall_act=True),
        obs_builder=EmeraldObs(max_size=1,
                               pads_inf_boost=REAL_T3_BOOST)),
    "t3p": lambda: EmeraldVersion(
        policy_path="PPO_POLICY_T3P.pt",
        agent=EmeraldAgentPPO((1024, 512, 512, 512, 256)),
        action_parser=EmeraldLookupAction(include_wall_act=True),
        obs_builder=EmeraldObs(max_size=1,
                               pads_inf_boost=REAL_T3_BOOST)),
    "t3-fr": lambda: EmeraldVersion(
        policy_path="PPO_POLICY_T3-FR.pt",
        agent=EmeraldAgentPPO((1024, 512, 512, 512, 256)),
        action_parser=EmeraldLookupAction(include_wall_act=True),
        obs_builder=EmeraldObs(max_size=1,
                               pads_inf_boost=REAL_T3_BOOST)),
    "t3-a": lambda: EmeraldVersion(
        policy_path="PPO_POLICY_T3-A.pt",
        agent=EmeraldAgentPPO((1024, 512, 512, 512)),
        action_parser=EmeraldLookupAction(include_wall_act=True),
        obs_builder=EmeraldObs(max_size=1,
                               pads_inf_boost=REAL_T3_BOOST)),
    "t3-b": lambda: EmeraldVersion(
        policy_path="PPO_POLICY_T3-B.pt",
        agent=EmeraldAgentPPO((1024, 512, 512, 512)),
        action_parser=EmeraldLookupAction(include_wall_act=True),
        obs_builder=EmeraldObs(max_size=1,
                               pads_inf_boost=REAL_T3_BOOST)),
    "t3-c": lambda: EmeraldVersion(
        policy_path="PPO_POLICY_T3-C.pt",
        agent=EmeraldAgentPPO((1024, 512, 512, 512)),
        action_parser=EmeraldLookupAction(include_wall_act=True),
        obs_builder=EmeraldObs(max_size=1,
                               pads_inf_boost=REAL_T3_BOOST)),
    "t3-d": lambda: EmeraldVersion(
        policy_path="PPO_POLICY_T3-D.pt",
        agent=EmeraldAgentPPO((1024, 512, 512, 512)),
        action_parser=EmeraldLookupAction(include_wall_act=True),
        obs_builder=EmeraldObs(max_size=1,
                               pads_inf_boost=REAL_T3_BOOST)),
    "t3-e": lambda: EmeraldVersion(
        policy_path="PPO_POLICY_T3-E.pt",
        agent=EmeraldAgentPPO((1024, 512, 512, 512)),
        action_parser=EmeraldLookupAction(include_wall_act=True),
        obs_builder=EmeraldObs(max_size=1,
                               pads_inf_boost=REAL_T3_BOOST)),
    "t3-f": lambda: EmeraldVersion(
        policy_path="PPO_POLICY_T3-F.pt",
        agent=EmeraldAgentPPO((1024, 512, 512, 512)),
        action_parser=EmeraldLookupAction(include_wall_act=True),
        obs_builder=EmeraldObs(max_size=1,
                               pads_inf_boost=REAL_T3_BOOST)),
    "t3-g": lambda: EmeraldVersion(
        policy_path="PPO_POLICY_T3-G.pt",
        agent=EmeraldAgentPPO((1024, 512, 512, 512)),
        action_parser=EmeraldLookupAction(include_wall_act=True),
        obs_builder=EmeraldObs(max_size=1,
                               pads_inf_boost=REAL_T3_BOOST)),
    "t3-h": lambda: EmeraldVersion(
        policy_path="PPO_POLICY_T3-H.pt",
        agent=EmeraldAgentPPO((1024, 512, 512, 512)),
        action_parser=EmeraldLookupAction(include_wall_act=True),
        obs_builder=EmeraldObs(max_size=1,
                               pads_inf_boost=REAL_T3_BOOST)),



    "t4": lambda: EmeraldVersion(
        policy_path="PPO_POLICY_T4.pt",
        agent=EmeraldAgentPPO((1024, 512, 512, 512, 256)),
        action_parser=EmeraldLookupAction(include_wall_act=True),
        obs_builder=EmeraldObs(max_size=1,
                               pads_inf_boost=True,
                               pred_slices=T4_PRED_SLICES,
                               pred_before_player=True,
                               teammate_indicator=True)),
    "t4.1": lambda: EmeraldVersion(
        policy_path="PPO_POLICY_T4.1.pt",
        agent=EmeraldAgentPPO((1024, 512, 512, 512, 256)),
        action_parser=EmeraldLookupAction(include_wall_act=True),
        obs_builder=EmeraldObs(max_size=3,
                               pads_inf_boost=True,
                               pred_slices=MAIN_PRED_SLICES,
                               pred_before_player=True,
                               teammate_indicator=True)),
    "t4.2": lambda: EmeraldVersion(
        policy_path="PPO_POLICY_T4.2.pt",
        agent=EmeraldAgentPPO((1024, 1024, 512)),
        action_parser=EmeraldLookupAction(include_wall_act=True),
        obs_builder=EmeraldObs(max_size=2,
                               pads_inf_boost=True,
                               pred_slices=MAIN_PRED_SLICES,
                               pred_before_player=True,
                               teammate_indicator=True)),



    "t5": lambda: EmeraldVersion(
        policy_path="PPO_POLICY_T5.pt",
        agent=EmeraldAgentPPO((1024, 1024, 512, 512, 256), use_leakyRELU=True),
        action_parser=EmeraldLookupAction(include_wall_act=True),
        obs_builder=EmeraldObs(max_size=3,
                               mirror=True,
                               pads_inf_boost=True,
                               inf_boost_indicator=True,
                               use_v4_data=False,
                               teammate_indicator=True,
                               demo_indicator=True,
                               pred_slices=MAIN_PRED_SLICES)),
    "t5-o": lambda: EmeraldVersion(
        policy_path="PPO_POLICY_T5-O.pt",
        agent=EmeraldAgentPPO((1024, 1024, 512, 512, 256), use_leakyRELU=True),
        action_parser=EmeraldLookupAction(include_wall_act=True),
        obs_builder=EmeraldObs(max_size=3,
                               mirror=True,
                               inf_boost_indicator=True,
                               use_v4_data=False,
                               teammate_indicator=True,
                               demo_indicator=True,
                               pred_slices=MAIN_PRED_SLICES)),
    "t5.1": lambda: EmeraldVersion(
        policy_path="actor_T5.1.pt",
        agent=EmeraldAgentPPO((1024, 1024, 512, 512, 256), use_leakyRELU=True),
        action_parser=EmeraldLookupAction(include_wall_act=True),
        obs_builder=EmeraldObs(max_size=3,
                               mirror=True,
                               inf_boost_indicator=True,
                               use_v4_data=False,
                               teammate_indicator=True,
                               demo_indicator=True,
                               pred_slices=MAIN_PRED_SLICES)),
}