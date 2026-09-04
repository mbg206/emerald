# Emerald

A bot that plays Rocket League via the [RLBot API](https://rlbot.org/). Trained using proximal policy optimization, with earlier versions using Matthew Allen 's [rlgym-ppo](https://github.com/AechPro/rlgym-ppo) implementation.

**Note** for those not involved in RLBot/Rocket League: due to developers who use bots to create cheats for the game, Psyonix has asked bot developers within the RLBot community to not publish any bots better than Nexto, a grand champion-level bot. Therefore, checkpoints above Nexto-level will not be uploaded. In addition, I cannot currently publish any code used to train Emerald (which is very unfortunate as it is some of my best work...)

### Installation

~~To add some of the bots in this repo as a botpack in the RLBot V5 GUI, go to the Bots screen, click `Add/Remove`, then `Add Botpack`, then tick the `Custom` bubble, and in the text box that appears, type `mbg206/emerald-rlbot` and hit `Confirm`~~ Automatic installation coming soon...

`bobgen.py` is a helper script that automatically creates bob files for all `bot.toml` files. Bob support is not well tested (yet)

To manually add *every* bot in this repo, you will need to download this repository and run `setup.bat` or `setup.sh` (depending on platform + Python must be installed). Afterwards, you will need to add the main folder (parent of the `emerald` folder and the newly-created venv) as a bot folder in the GUI.

### Inclusion guide

✅ - Included
✔️ - Included in repo; not in botpack
❌ - Not included
⚠️ - See notes

| Version  | Mode | Est. Rank    | Included? | Notes |
|----------|------|--------------|-----------|-------|
| T1       | 1s   | Diamond      | ✅ |  |
| T2       | 1s   | Grand Champ  | ⚠️ | Latest version of T2 is significantly better than Nexto, and therefore should not be released publicly. An older checkpoint has been included that plays similar (though worse) |
| T2-A     | 1s   | Champion     | ✅ | Pulled back on demo rewards to reduce demochasing behavior in late training |
| T2-B     | 1s   | Diamond      | ✅ | Trained from scratch using T2's final rewards and hyperparameters (no curriculum learning). Learned thing *except* aerials, despite active encouragement. |
| T2-C     | 1s   | Grand Champ  | ✅ | An experiment run from mid-training, using extremely large batch sizes and low learning rates. Did not improve significantly |
| T2-D     | 2s   | Diamond      | ✅ | Created after T4; uses T2's curriculum in a 2v2 bot. Didn't learn dribbling well, but still considered a success. |
| T2-Myst  | 2s   | Champion     | ✅ | A mystery checkpoint, likely from earlier in training |
| T3       | 1s   | Grand Champ  | ⚠️ | First version with air dribble encouragement. Rewards underwent extensive changes and experiments during development, with smaller models being used to test new reward ideas (see versions below). Eventually abandoned due to a critical bug in infinite boost episodes causing collapses in learning. Older, more capable checkpoint is included |
| T3+      | 1s   | Diamond?     | ✅ | Alternative/scrapped version of T3 that learned more air dribbling. Likes being in the air more than T3 |
| T3-FR    | 1s   | ---          | ✔️ | Alternate T3 run with stronger flip reset rewards. Failed due to the same infinite boost episode bug |
| T3-A     | 1s   | ---          | ✔️ | Air dribble test (failure) |
| T3-B     | 1s   | ---          | ✔️ | Air dribble test (failure) |
| T3-D     | 1s   | ---          | ✔️ | Air dribble test (success) |
| T3-E     | 1s   | ---          | ✔️ | Flip reset test (success) |
| T3-F     | 1s   | ---          | ✔️ | Double tap test (partial success) |
| T3-G     | 1s   | ---          | ✔️ | Flip reset test (partial failure) |
| T3-H     | 1s   | ---          | ✔️ | Flip reset test (failure) |
| T4       | 1s   | Grand Champ  | ⚠️ | First version with ball pred. Never learned dribbling, but still developed decent skill. Newest checkpoints play better than Nexto, so an older checkpoint is included |
| T4.1     | 1s, 2s, 3s | Gold   | ✅ | Teamplay test |
| T4.2     | 2s   | Bronze       | ✔️ | Short-lived teamplay test |
| T5       | ?    | ?            | ❌ | Coming soon... |
| T5.1     | ?    | ?            | ❌ | Coming soon... |

For higher-level versions: T2 > T4 > T3 > T2-C > T2-A > T2-Myst

Note about T3 versions: when using the unlimited boost mutator, T3 versions will still play semi-normal. If you want to see how they play when they actually know about the mutator setting (which causes their behavior to break significantly), set `REAL_T3_BOOST` to `True` in `src/vers.py`