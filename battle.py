import os
import shutil
import time
import json
import torch
import multiprocessing

import mjx
from mjx.agents import Agent as MjxAgent

from model.simple_mlp_model import SimpleMlpModel
from model.split_mlp_model import SplitMlpModel
from feature.feature_vector import BoardFeature, ActionFeature, DiscardActionFeature, OptionalActionFeature
from actor.simple_mlp_actor import SimpleMlpActor, SimpleMlpMenzenActor, SimpleMlpShantenActor
from actor.split_mlp_actor import SplitMlpActor, SplitMlpMenzenActor, SplitMlpShantenActor
from learn import SimpleMlpLearner, SplitMlpLearner

class SingleGame:
    def __init__(
        self,
        agents: list[MjxAgent]
    ):
        if len(agents) != 4:
            raise Exception("len(agents) is not 4")
        self.agents = {f"player_{idx}": agent for idx, agent in enumerate(agents)}
        self.env = mjx.MjxEnv()
        self.start_time = time.time()   # ログ用

    def run(self):
        obs_dict = self.env.reset()
        while not self.env.done(done_type="game"):
            actions = {player_id: self.agents[player_id].act(obs) for player_id, obs in obs_dict.items()}
            obs_dict = self.env.step(actions)

            if self.env.done(done_type="round"):    # 局終了時
                state = json.loads(self.env.state().to_json())

                player_ids = state["publicObservation"]["playerIds"]
                init_scores = state["publicObservation"]["initScore"]["tens"]
                final_scores = state["roundTerminal"]["finalScore"]["tens"]
                score_diffs = [final - init for init, final in zip(init_scores, final_scores)]
                score_diffs = [
                    [score_diffs[0], score_diffs[1], score_diffs[2], score_diffs[3]],
                    [score_diffs[1], score_diffs[2], score_diffs[3], score_diffs[0]],
                    [score_diffs[2], score_diffs[3], score_diffs[0], score_diffs[1]],
                    [score_diffs[3], score_diffs[0], score_diffs[1], score_diffs[2]],
                    ]
                ranks = [sorted(final_scores, reverse=True).index(x) + 1 for x in final_scores]
                yakus = [[False for _ in range(55)] for _ in range(4)]

                if "wins" in state["roundTerminal"]:
                    wins = state["roundTerminal"]["wins"]
                    for win in wins:
                        changes = win["tenChanges"]
                        winner = None   # なぜかwhoがないことがあるので点数の変更から決める
                        for idx, change in enumerate(changes):
                            if change > 0:
                                winner = idx
                        if "yakus" in win:
                            for idx in win["yakus"]:
                                yakus[winner][idx] = True
                        elif "yakumans" in win:
                            for idx in win["yakumans"]:
                                yakus[winner][idx] = True
                        else:
                            raise Exception(f"yakus or yakumans is not in win: {win}")

                for player_id, diffs, yaku, rank in zip(player_ids, score_diffs, yakus, ranks):
                    self.agents[player_id].end_round(diffs, rank, yaku)

                if self.env.done(done_type="game"): # ゲーム終了時
                    for player_id, final, rank in zip(player_ids, final_scores, ranks, strict=True):
                        self.agents[player_id].end_game(final, rank)

    def visualize_one_round(
        self,
        save_dir:str
    ):
        if os.path.isdir(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        obs_dict = self.env.reset()
        Actor.LOG_FILE = f"{save_dir}/actor_log.txt"

        turn = 0
        while not self.env.done(done_type="round"):
            actions = {player_id: self.agents[player_id].act(obs) for player_id, obs in obs_dict.items()}
            obs_dict = self.env.step(actions)
            self.env.state().save_svg(f"{save_dir}/{turn}.svg")
            turn += 1
        Actor.LOG_FILE = None

        state = json.loads(self.env.state().to_json())
        if "wins" in state["roundTerminal"]:
            print(state["roundTerminal"]["wins"])
        if "noWinner" in state["roundTerminal"]:
            print(state["roundTerminal"]["noWinner"])

def battle_SimpleMlp(
    rule_index: int = 0,
    log_time: float = 0.0
):
    learn_dir = "./learn"

    model = SimpleMlpModel(BoardFeature.SIZE, ActionFeature.SIZE)
    model_path = None

    # epsilon = 0.15
    epsilon = 0.05

    agents = None
    if log_time is not None:    # log
        agents = [
            SimpleMlpActor(model, epsilon=0.0),
            SimpleMlpActor(model, epsilon=0.0),
            SimpleMlpActor(model, epsilon=0.0),
            SimpleMlpShantenActor(model),
        ]
    else:                       # 自己対戦
        if rule_index == 0:
            agents = [
                SimpleMlpActor(model, epsilon=epsilon),
                SimpleMlpActor(model, epsilon=epsilon),
                SimpleMlpActor(model, epsilon=epsilon),
                SimpleMlpShantenActor(model)
            ]
        elif rule_index == 1:
            agents = [
                SimpleMlpActor(model, epsilon=epsilon),
                SimpleMlpActor(model, epsilon=epsilon),
                SimpleMlpActor(model, epsilon=epsilon),
                SimpleMlpMenzenActor(model)
            ]
    
    game = SingleGame(agents)
    cnt = 0
    while True:
        with open(os.path.join(learn_dir, f"modelpath.txt"), mode="r") as f:
            model_path = f.readline()
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu"), weights_only=False))

        for _ in range(10):
            game.run()
        
        for player_id in range(4):
            agents[player_id].export(os.path.join(learn_dir, f"learndata"), SimpleMlpLearner.FILE_SIZE, SimpleMlpLearner.BIN_NUM)
        if log_time != None:
            log_dict = {"time": time.time() - game.start_time + log_time, "scores": [agent.get_score() for agent in agents], "ranks": [agent.get_rank() for agent in agents], "epsilon": epsilon}
            print(log_dict, flush=True)

        epsilon *= 0.998
        if log_time is None:
            for agent in agents:
                agent.set_random_parameter(epsilon=epsilon)

def battle_SplitMlp(
    rule_index: int = 0,
    log_time: float = 0.0
):
    learn_dir = "./learn"

    model = SplitMlpModel(BoardFeature.SIZE, DiscardActionFeature.SIZE, OptionalActionFeature.SIZE)
    model_path = None

    # epsilon = 0.15
    epsilon = 0.05

    agents = None
    if log_time is not None:    # log
        agents = [
            SplitMlpActor(model, discard_softmax=False, optional_epsilon=0.0),
            SplitMlpActor(model, discard_softmax=False, optional_epsilon=0.0),
            SplitMlpActor(model, discard_softmax=False, optional_epsilon=0.0),
            SplitMlpShantenActor(model),
        ]
    else:                       # 自己対戦
        if rule_index == 0:
            agents = [
                SplitMlpActor(model, discard_softmax=True, optional_epsilon=epsilon),
                SplitMlpActor(model, discard_softmax=True, optional_epsilon=epsilon),
                SplitMlpActor(model, discard_softmax=True, optional_epsilon=epsilon),
                SplitMlpShantenActor(model)
            ]
        elif rule_index == 1:
            agents = [
                SplitMlpActor(model, discard_softmax=True, optional_epsilon=epsilon),
                SplitMlpActor(model, discard_softmax=True, optional_epsilon=epsilon),
                SplitMlpActor(model, discard_softmax=True, optional_epsilon=epsilon),
                SplitMlpMenzenActor(model)
            ]
    
    game = SingleGame(agents)
    cnt = 0
    while True:
        with open(os.path.join(learn_dir, f"modelpath.txt"), mode="r") as f:
            model_path = f.readline()
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu"), weights_only=False))

        for _ in range(10):
            game.run()
        
        for player_id in range(4):
            agents[player_id].export(os.path.join(learn_dir, f"learndata"), SplitMlpLearner.FILE_SIZE, SplitMlpLearner.BIN_NUM)
        if log_time != None:
            log_dict = {"time": time.time() - game.start_time + log_time, "scores": [agent.get_score() for agent in agents], "ranks": [agent.get_rank() for agent in agents], "epsilon": epsilon}
            print(log_dict, flush=True)

        epsilon *= 0.998
        if log_time is None:
            for agent in agents:
                agent.set_random_parameter(discard_softmax=None, optional_epsilon=epsilon)

def main():
    torch.set_num_threads(1)
    num_subprocess = 9  # ログを表示しないプロセスの数
    temperature = 1.0

    battle = battle_SimpleMlp
    # battle = battle_SplitMlp
    
    log_time = 0.0
    main_process = multiprocessing.Process(target=battle, args=(0, log_time))
    main_process.start()

    sub_processes = []
    rule_index = 0
    while True:
        while len(sub_processes) < num_subprocess:
            process = multiprocessing.Process(target=battle, args=(rule_index, None))
            sub_processes.append(process)
            process.start()
            rule_index = 1 - rule_index
            # rule_index = (rule_index + 1) % num_subprocess
            # rule_index = (rule_index + 1) % 3

        for process in sub_processes:
            process.join(timeout=1)
            if not process.is_alive():
                sub_processes.remove(process)

if __name__ == "__main__":
    main()