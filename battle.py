import os
import time
import json
import torch
import multiprocessing

import mjx

from learn import LearningConstants
from actor import ShantenActor, Actor
from model import Model
from feature import BoardFeature, ActionFeature

class SingleGame:
    def __init__(
        self,
        agents: list[Actor]
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
                for player_id, init, final, yaku in zip(player_ids, init_scores, final_scores, yakus):
                    self.agents[player_id].end_round(final - init, yaku)

                if self.env.done(done_type="game"): # ゲーム終了時
                    ranks = [sorted(final_scores, reverse=True).index(x) + 1 for x in final_scores]
                    for player_id, final, rank in zip(player_ids, final_scores, ranks, strict=True):
                        self.agents[player_id].end_game(final, rank)

    def visualize_one_round(
        self,
        save_path: str
    ):
        obs_dict = self.env.reset()
        while not self.env.done(done_type="round"):
            actions = {player_id: self.agents[player_id].act(obs) for player_id, obs in obs_dict.items()}
            obs_dict = self.env.step(actions)

        state = json.loads(self.env.state().to_json())
        if "wins" in state["roundTerminal"]:
            print(state["roundTerminal"]["wins"])
        if "noWinner" in state["roundTerminal"]:
            print(state["roundTerminal"]["noWinner"])
        self.env.state().save_svg("./visualize.svg")

    def repeat(
        self,
        repeat: int = -1,
        log_time: float = None  # 入ってるとログを表示する
    ):
        # 引数なしで永遠に繰り返す
        if repeat < -1:
            raise Exception(f"invalid repeat number: {repeat}")
        cnt = 0
        while True:
            if cnt == repeat:
                break
            self.run()
            if log_time != None and cnt == 0:
                log_dict = {"time": time.time() - self.start_time + log_time, "scores": [agent.get_score() for agent in self.agents.values()], "ranks": [agent.get_rank() for agent in self.agents.values()]}
                print(log_dict, flush=True)
            cnt += 1

def battle(
    iteration = -1,
    temperature: float = None,
    log_time: float = None  # 入ってるとログを表示する
):
    # とりあえずmodel0のみ
    learn_dir = "./learn"

    model = Model(BoardFeature.SIZE, ActionFeature.SIZE)
    model_path = ""

    # battle_agents = [
    #     ShantenActor(model),
    #     ShantenActor(model),
    #     ShantenActor(model),
    #     ShantenActor(model)
    # ]
    battle_agents = [
        Actor(model),
        Actor(model),
        Actor(model, temperature),
        ShantenActor(model)
    ]
    log_agents = [
        Actor(model),
        Actor(model),
        Actor(model),
        ShantenActor(model)
    ]
    agents = None
    if log_time is not None:
        agents = log_agents
    else:
        agents = battle_agents
    game = SingleGame(agents)
    cnt = 0
    while True:
        if cnt == iteration:
            break

        with open(os.path.join(learn_dir, f"modelpath_{0}.txt"), mode="r") as f:
            model_path = f.readline()

        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu"), weights_only=False))
        game.repeat(LearningConstants.BATTLE_NUM, log_time)
        agents[0].export(os.path.join(learn_dir, f"learndata_{0}"), LearningConstants.FILE_SIZE, LearningConstants.BIN_NUM)

        cnt += 1

def main():
    torch.set_num_threads(1)
    num_subprocess = 9  # ログを表示しないプロセスの数
    temperature = 1.0
    
    log_time = 0.0
    main_process = multiprocessing.Process(target=battle, args=(-1, None, log_time))
    main_process.start()

    sub_processes = []
    while True:
        while len(sub_processes) < num_subprocess:
            process = multiprocessing.Process(target=battle, args=(10, temperature, None))
            sub_processes.append(process)
            process.start()

        for process in sub_processes:
            process.join(timeout=1)
            if not process.is_alive():
                sub_processes.remove(process)

if __name__ == "__main__":
    main()