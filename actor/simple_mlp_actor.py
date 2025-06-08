import os
import random
import struct
import numpy as np
import torch
import pickle
from io import BytesIO
from collections import deque
from enum import IntEnum, IntFlag, unique

import mjx
from mjx.agents import Agent as MjxAgent
from mjx.action import Action as MjxAction
from mjx.observation import Observation as MjxObservation
from mjx.agents import ShantenAgent as MjxShantenAgent

from board import Board, Action, ActionKind, TileKind, Tile, Relation
from feature.feature_vector import FeatureVector, BoardFeature, ActionFeature
from model.simple_mlp_model import SimpleMlpModel

class TrainingData:
    def __init__(
        self,
        board_vector: FeatureVector,
        action_vector: FeatureVector,
        value_inferred: float,
        value_label: float = None,
        yaku_label: list[bool] = None,
        policy_label: float = None,
        score_label: list[float] = None,
        round_reward: float = None,
        game_reward: float = None,
        round_final: bool = False,
    ):
        self.board_vector = board_vector
        self.action_vector = action_vector
        self.value_inferred = value_inferred
        self.value_label = value_label
        self.yaku_label = yaku_label
        self.policy_label = policy_label
        self.score_label = score_label
        self.round_reward = round_reward
        self.game_reward = game_reward
        self.round_final = round_final

class Dataset:
    def __init__(
        self,
        data: list[TrainingData] = None
    ):
        if data == None:
            data = []
        self.data = data
    
    def add(
        self,
        data: TrainingData
    ):
        self.data.append(data)

    def addrange(
        self,
        data: list[TrainingData]
    ):
        self.data.extend(data)
    
    def get_data(self):
        return self.data

    def split(
        self,
        size: int,
    ):
        # size個ごとに区切って新しいDatasetにする
        # 余りは無視する
        fullsize = len(self.data)
        datasets = [Dataset(self.data[idx-size:idx]) for idx in range(size, fullsize, size)]
        return datasets

    def make_inputs(self):
        board_indexes, board_offsets = [], []
        action_indexes, action_offsets = [], []
        for entry in self.data:
            board_offsets.append(len(board_indexes))
            action_offsets.append(len(action_indexes))
            board_indexes += entry.board_vector.get_indexes()
            action_indexes += entry.action_vector.get_indexes()
        return board_indexes, board_offsets, action_indexes, action_offsets

    def make_labels(self):
        value_labels, yaku_labels, policy_labels, score_labels = [], [], [], []
        for entry in self.data:
            value_labels.append(entry.value_label)
            yaku_labels.append(self.make_yaku_labels([1 if flag else 0 for flag in entry.yaku_label]))
            policy_labels.append(entry.policy_label)
            score_labels.append(entry.score_label)
        return value_labels, yaku_labels, policy_labels, score_labels

    def make_yaku_labels(self, yaku_raw):
        # 成立した役の一覧からyaku_labelsを作る
        '''
        1. 役が成立しなかった（＝上がれなかった）
        2. 面前役のみで上がった
        3. 副露役を含めて上がった
        の3つのラベルに分ける
        '''
        open_yaku = [       # Mjxの副露役のindex
            3,  # 槍槓
            4,  # 嶺上開花
            5,  # 海底撈月
            6,  # 河底撈魚
            8,  # 断幺九
            10, # 自風（東）
            11, # 自風（南）
            12, # 自風（西）
            13, # 自風（北）
            14, # 場風（東）
            15, # 場風（南）
            16, # 場風（西）
            17, # 場風（北）
            18, # 役牌（白）
            19, # 役牌（發）
            20, # 役牌（中）
            23, # 混全帯幺九
            24, # 一気通貫
            25, # 三色同順
            26, # 三色同刻
            27, # 三槓子
            28, # 対々和
            29, # 三暗刻
            30, # 小三元
            31, # 混老頭
            33, # 純全帯幺九
            34, # 混一色
            35, # 清一色
            39, # 大三元
            42, # 字一色
            43, # 緑一色
            44, # 清老頭
            49, # 大四喜
            50, # 小四喜
            51, # 四槓子
        ]

        cnt = 0
        for idx, flag in enumerate(yaku_raw):
            if flag:
                cnt += 1
                if idx in open_yaku:    # 副露役を含めて上がり
                    return [0, 0, 1]
        if cnt > 0: # 面前役のみで上がり
            return [0, 1, 0]
        else:       # 上がれなかった
            return [1, 0, 0]

    def __len__(self):
        return len(self.data)

class Episode(Dataset):
    DISCOUNT_RATE = 0.95                        # 報酬の割引率（value_inferredで埋める）
    RESULT_RATE = 0.2                           # valueにresultを反映させる割合
    POLICY_MAX = 0.4
    POLICY_MIN = -POLICY_MAX
    REWARD_ROUND_RATE = 0.3                     # 報酬全体のうち、局ごとのものの割合
    REWARD_GAME_RATE = 1.0 - REWARD_ROUND_RATE

    def __init__(self):
        super().__init__()
    
    def set_round_result(
        self,
        round_reward: float,
        yaku: list[bool],
        score_diffs: list[int],
    ):
        next_round_value = round_reward
        if len(self.data) == 0: # 東一局の九種九牌？
            return
        self.data[-1].round_final = True
        for entry in reversed(self.data):
            if entry.round_reward is not None:
                if not entry.round_final:
                    raise Exception("unexpected round final")
                break   # 前の局のデータに来たら終わり
            entry.yaku_label = yaku
            entry.score_label = [diff / 48000.0 for diff in score_diffs]
            entry.round_reward = round_reward
            next_round_value = next_round_value * self.DISCOUNT_RATE + entry.value_inferred * (1.0 - self.DISCOUNT_RATE)

    def set_game_result(
        self,
        game_reward: float,
    ):
        next_round_value, next_game_value = None, game_reward
        next_value_inferred = None
        for entry in reversed(self.data):
            if entry.yaku_label is None or entry.score_label is None:
                Exception("yaku/score label is not setted")
            if entry.round_reward is None:
                Exception("round_reward is None")
            if entry.round_final:
                next_round_value = entry.round_reward
                next_value_inferred = next_round_value * self.REWARD_ROUND_RATE + next_game_value * self.REWARD_GAME_RATE
            entry.game_reward = game_reward
            full_reward = entry.round_reward * self.REWARD_ROUND_RATE + entry.game_reward * self.REWARD_GAME_RATE
            next_full_value = next_round_value * self.REWARD_ROUND_RATE + next_game_value * self.REWARD_GAME_RATE
            entry.value_label = next_full_value * (1.0 - self.RESULT_RATE) + full_reward * self.RESULT_RATE

            policy = max(self.POLICY_MIN, min(next_value_inferred - entry.value_inferred, self.POLICY_MAX))
            # policy = max(self.POLICY_MIN, min(next_full_value - entry.value_inferred, self.POLICY_MAX))

            # print(f"finish: {entry.round_final}, next_round_value: {next_round_value}, next_game_value: {next_game_value}, next_value_inffered: {next_value_inferred}, round_reward: {entry.round_reward}, game_reward: {entry.game_reward}, full_reward: {full_reward}, next_full_value: {next_full_value}")
            entry.policy_label = policy

            next_value_inferred = entry.value_inferred
            next_round_value = next_round_value * self.DISCOUNT_RATE + entry.value_inferred * (1.0 - self.DISCOUNT_RATE)
            next_game_value = next_game_value * self.DISCOUNT_RATE + entry.value_inferred * (1.0 - self.DISCOUNT_RATE)

class Inferred:
    def __init__(
        self,
        board: Board,
        board_vector: FeatureVector,
        mjx_action: MjxAction,
        action_vector: FeatureVector,
        value: float,
        yaku: list[float],
        policy: float
    ):
        if len(yaku) != 3:
            raise Exception("unexpected length of list")
        self.board = board
        self.board_vector = board_vector
        self.value = value
        self.yaku = yaku
        self.mjx_action = mjx_action
        self.action_vector = action_vector
        self.policy = policy

    def __repr__(self):
        if self.mjx_action is None:
            return f"Action: no, value: {self.value}, policy: {self.policy}"
        action = Action.from_mjx(self.mjx_action, self.board)
        string = "{" + f"Action: \"{action}\", value: {self.value}, policy: {self.policy}" + "}"
        return string

class Trainer:
    def __init__(self):
        self.episodes = []
        self.current_episode = Episode()
        self.extra_data = Dataset()
        self.bin_idx = 0

    def add(
        self,
        inferred: Inferred
    ):
        training_data = TrainingData(inferred.board_vector, inferred.action_vector, inferred.value)
        self.current_episode.add(training_data)

    def end_round(
        self,
        score_diffs: list[int],
        rank: int,
        yaku: list[bool],
    ):
        self.current_episode.set_round_result([1.0, 0.3, -0.3, -1.0][rank - 1], yaku, score_diffs)
    
    def end_game(
        self,
        final_rank: int,
    ):
        self.current_episode.set_game_result([1.0, 0.3, -0.3, -1.0][final_rank - 1])
        self.episodes.append(self.current_episode)
        self.current_episode = Episode()

    def export(
        self,
        dir_path: str,
        file_size: int,
        bin_num: int
    ):
        alldata = sum([episode.get_data() for episode in self.episodes], [])
        alldata += self.extra_data.get_data()
        random.shuffle(alldata)

        idx, size = 0, len(alldata)
        while size - idx >= file_size:
            r = random.getrandbits(64)
            dataset = Dataset(alldata[idx:idx+file_size])
            with open(os.path.join(dir_path, f"bin_{self.bin_idx}/data_{r}.pkl"), "wb") as f:
                pickle.dump(dataset, f)
            idx += file_size
            self.bin_idx = (self.bin_idx + 1) % bin_num
        self.extra_data = Dataset(alldata[idx:])
        self.episodes = []
        
    def reset(self):
        self.episodes.clear()
        self.current_episode = Episode()
    
    @staticmethod
    def calc_reward(
        result: int
    ):
        reward = result / 48000
        return reward

class Evaluator:
    MAX_BATTLE = 100

    def __init__(self):
        self.games = deque()
        self.score_sum = 0
        self.rank_sum = 0
    
    def add(
        self,
        score: int,
        rank: int
    ):
        self.games.append((score, rank))
        self.score_sum += score
        self.rank_sum += rank
        if len(self.games) > self.MAX_BATTLE:
            score_popped, rank_popped = self.games.popleft()
            self.score_sum -= score_popped
            self.rank_sum -= rank_popped

    def get_moving_average(self):
        if len(self.games) == 0:
            return 0.0, 0.0
        return self.score_sum / len(self.games), self.rank_sum / len(self.games)

class SimpleMlpActor(MjxAgent):
    LOG_FILE = None

    def __init__(
        self,
        model:SimpleMlpModel,
        epsilon: float = 0.0   # ランダムに選択する確率
        ) -> None:
        super().__init__()
        self.evaluator = Evaluator()
        self.trainer = Trainer()
        self.model = model
        self.model.eval()
        self.epsilon = epsilon
    
    def set_random_parameter(
        self,
        epsilon: float,
    ):
        self.epsilon = epsilon
    
    def dump(
        self,
        file_path: str,
        text: str
    ):
        # テキストをファイルに書き込む
        with open(file_path, "a") as f:
            f.write(text + "\n")

    def act(
        self,
        observation: MjxObservation
        ) -> MjxAction:

        # DUMMYはすぐ返す
        legal_actions = observation.legal_actions()
        if len(legal_actions) >= 1 and legal_actions[0].type() == mjx.ActionType.DUMMY:
            return legal_actions[0]

        return self._inner_act(observation)
    
    def check(
        self,
        observation: MjxObservation
        ):

        # DUMMYはすぐ返す
        legal_actions = observation.legal_actions()
        if len(legal_actions) >= 1 and legal_actions[0].type() == mjx.ActionType.DUMMY:
            print("DUMMY")
            return legal_actions[0]

        return self._inner_act(observation, dump=True)

    def _infer(
        self,
        observation: MjxObservation
    ):
        legal_actions = observation.legal_actions()
        board_indexes, board_offsets = [], []
        action_indexes, action_offsets = [], []
        board_vectors, action_vectors = [], []

        board = Board.from_mjx(observation)
        for action in observation.legal_actions():
            board_vector = BoardFeature.make(board)
            action_vector = ActionFeature.make(Action.from_mjx(action, board), board)

            board_offsets.append(len(board_indexes))
            action_offsets.append(len(action_indexes))
            board_indexes += board_vector.get_indexes()
            action_indexes += action_vector.get_indexes()
            board_vectors.append(board_vector)
            action_vectors.append(action_vector)

        board_indexes = torch.LongTensor(board_indexes)
        board_offsets = torch.LongTensor(board_offsets)
        action_indexes = torch.LongTensor(action_indexes)
        action_offsets = torch.LongTensor(action_offsets)

        value, yaku, policy, score = self.model(board_indexes, board_offsets, action_indexes, action_offsets)

        inferred = [Inferred(board, board_vector, mjx_action, action_vector, v[0], y, p[0]) for mjx_action, action_vector, v, y, p in zip(legal_actions, action_vectors, value.tolist(), yaku.tolist(), policy.tolist(), strict=True)]
        return inferred

    def _inner_act(
        self,
        observation: MjxObservation
        ) -> MjxAction:

        inferred = sorted(self._infer(observation), key=lambda x: x.policy, reverse=True)

        selected = None
        if random.random() < self.epsilon:
            selected = random.choice(inferred)     # ランダムに選択
        else:
            selected = inferred[0]                 # 最善手を選ぶ

        self.trainer.add(selected)

        return selected.mjx_action

    def export(
        self,
        dir_path: str,
        file_size: int,
        bin_num: int
    ):
        self.trainer.export(dir_path, file_size, bin_num)

    def end_round(
        self,
        score_diffs: list[int],
        rank: int,
        yaku: list[bool]
    ):
        self.trainer.end_round(score_diffs, rank, yaku)

    def end_game(
        self,
        final_score: int,
        final_rank: int
    ):
        self.trainer.end_game(final_rank)
        self.evaluator.add(final_score, final_rank)

    def get_score(
        self
    ):
        score, _ = self.evaluator.get_moving_average()
        return score

    def get_rank(
        self
    ):
        _, rank = self.evaluator.get_moving_average()
        return rank

class SimpleMlpMenzenActor(SimpleMlpActor):
    def __init__(
        self,
        model: SimpleMlpModel
        ) -> None:
        super().__init__(model)
    
    def _select_mjx_action(
        self,
        observation: MjxObservation
        ) -> MjxAction:

        # ShantenActorの副露しない版
        legal_actions = observation.legal_actions()
        win_actions = [action for action in legal_actions if action.type() in [mjx.ActionType.TSUMO, mjx.ActionType.RON]]
        if len(win_actions) >= 1:
            return win_actions[0]
        
        riichi_actions = [action for action in legal_actions if action.type() == mjx.ActionType.RIICHI]
        if len(riichi_actions) >= 1:
            return riichi_actions[0]
        
        legal_discard_actions = [action for action in legal_actions if action.type() in [mjx.ActionType.DISCARD, mjx.ActionType.TSUMOGIRI]]
        effective_discard_types = observation.curr_hand().effective_discard_types()
        effective_discard_actions = [a for a in legal_discard_actions if a.tile().type() in effective_discard_types]
        if len(effective_discard_actions) > 0:
            return random.choice(effective_discard_actions)

        pass_actions = [action for action in legal_actions if action.type() == mjx.ActionType.PASS]
        if len(pass_actions) >= 1:
            return pass_actions[0]
        
        return random.choice(legal_discard_actions)
    
    def _inner_act(
        self,
        observation: MjxObservation
        ) -> MjxAction:
        mjx_selected = self._select_mjx_action(observation)

        inferred = self._infer(observation)

        for entry in inferred:
            if entry.mjx_action == mjx_selected:
                self.trainer.add(entry)
                return entry.mjx_action

        raise Exception("selected action not found")

class SimpleMlpShantenActor(SimpleMlpMenzenActor):
    def __init__(
        self,
        model: SimpleMlpModel
    ) -> None:
        super().__init__(model)
        self.shanten_agent = MjxShantenAgent()

    def _select_mjx_action(
        self,
        observation: MjxObservation
    ) -> MjxAction:
        return self.shanten_agent.act(observation)
