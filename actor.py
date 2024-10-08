import os
import random
import struct
import numpy as np
import torch
from io import BytesIO
from collections import deque

import mjx
from mjx.agents import Agent as MjxAgent
from mjx.action import Action as MjxAction
from mjx.observation import Observation as MjxObservation
from mjx.agents import ShantenAgent as MjxShantenAgent

from board import Board, Action
from feature import FeatureVector, BoardFeature, ActionFeature
from model import Model

class TrainingData:
    def __init__(
        self,
        board_vector: FeatureVector,
        action_vector: FeatureVector,
        value_inferred: float,
        value_label: float = None,
        yaku_label: list[bool] = None,
        policy_label: float = None
    ):
        self.board_vector = board_vector
        self.action_vector = action_vector
        self.value_inferred = value_inferred
        self.value_label = value_label
        self.yaku_label = yaku_label
        self.policy_label = policy_label
    
    def to_bytes(self):
        b = struct.pack("H", len(self.board_vector))
        for idx in self.board_vector.get_indexes():
            b += struct.pack("H", idx)
        b += struct.pack("H", len(self.action_vector))
        for idx in self.action_vector.get_indexes():
            b += struct.pack("H", idx)
        b += struct.pack("d", self.value_inferred)
        b += struct.pack("d", self.value_label)
        b += struct.pack("H", len(self.yaku_label))
        for lab in self.yaku_label:
            b += struct.pack("?", lab)
        b += struct.pack("d", self.policy_label)
        return b

    @classmethod
    def from_bytes(
        cls,
        buffer: BytesIO
    ):
        len_bv, = struct.unpack("H", buffer.read(2))
        board_vector = FeatureVector([struct.unpack("H", buffer.read(2))[0] for _ in range(len_bv)])
        len_av, = struct.unpack("H", buffer.read(2))
        action_vector = FeatureVector([struct.unpack("H", buffer.read(2))[0] for _ in range(len_av)])
        value_inferred, = struct.unpack("d", buffer.read(8))
        value_label, = struct.unpack("d", buffer.read(8))
        len_yl, = struct.unpack("H", buffer.read(2))
        yaku_label = [struct.unpack("?", buffer.read(1))[0] for _ in range(len_yl)]
        policy_label, = struct.unpack("d", buffer.read(8))
        return cls(board_vector, action_vector, value_inferred, value_label, yaku_label, policy_label)

    def __repr__(self):
        return f"inferred: (value: {self.value_inferred}), label: (value: {self.value_label}, yaku: {self.yaku_label}, policy: {self.policy_label})"

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

    def get_data(self):
        return self.data

    def export(
        self,
        file_path: str
    ):
        b = b"".join([td.to_bytes() for td in self.data])
        with open(file_path, "wb") as f:
            f.write(b)
    
    def split(
        self,
        size: int
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
        value_labels, yaku_labels, policy_labels = [], [], []
        for entry in self.data:
            value_labels.append(entry.value_label)
            yaku_labels.append([1 if flag else 0 for flag in entry.yaku_label])
            policy_labels.append(entry.policy_label)
        return value_labels, yaku_labels, policy_labels
    
    @classmethod
    def from_bytes(
        cls,
        b: bytes
    ):
        size = len(b)
        buffer = BytesIO(b)
        data = []
        while buffer.tell() < size:
            data.append(TrainingData.from_bytes(buffer))
        return cls(data)

    def __len__(self):
        return len(self.data)

class Episode(Dataset):
    DISCOUNT_RATE = 0.9 # 報酬の割引率（value_inferredで埋める）
    RESULT_RATE = 0.4   # valueにresultを反映させる割合
    POLICY_MAX = 0.4
    POLICY_MIN = -POLICY_MAX

    def __init__(self):
        super().__init__()

    def result(
        self,
        reward: int,
        yaku: list[bool]
    ):
        next_value = reward
        for entry in reversed(self.data):
            entry.value_label = (next_value + entry.value_inferred) / 2.0 * (1.0 - self.RESULT_RATE) + reward * self.RESULT_RATE 
            entry.yaku_label = yaku
            entry.policy_label = max(self.POLICY_MIN, min(next_value - entry.value_inferred, self.POLICY_MAX))
            next_value = next_value * self.DISCOUNT_RATE + entry.value_inferred * (1.0 - self.DISCOUNT_RATE)

    def __repr__(self):
        return f"Episode: {self.data}"

class Inferred:
    def __init__(
        self,
        action: MjxAction,
        board_vector: FeatureVector,
        action_vector: FeatureVector,
        value: float,
        yaku: list[float],
        policy: float
    ):
        self.action = action
        self.board_vector = board_vector
        self.action_vector = action_vector
        self.value = value
        self.yaku = yaku
        self.policy = policy

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

    def end(
        self,
        result: int,
        yaku: list[bool]
    ):
        self.current_episode.result(self.calc_reward(result), yaku)
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
            dataset.export(os.path.join(dir_path, f"bin_{self.bin_idx}/data_{r}.dat"))
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

class Actor(MjxAgent):
    def __init__(
        self,
        model: Model,
        temperature: float = None   # 引数なしでランダム性なし
        ) -> None:
        super().__init__()
        self.evaluator = Evaluator()
        self.trainer = Trainer()
        self.temperature = temperature
        self.model = model
        self.model.eval()

    def softmax(
        self,
        inferred: list[Inferred]
    ):
        if self.temperature == None:
            raise Exception("temperature is None")
        else:
            x = np.array([entry.policy for entry in inferred])
            exp_x = np.exp(x / self.temperature)
            p = exp_x / np.sum(exp_x)
            return p.tolist()


    def act(
        self,
        observation: MjxObservation
        ) -> MjxAction:

        # DUMMYはすぐ返す
        legal_actions = observation.legal_actions()
        if len(legal_actions) >= 1 and legal_actions[0].type() == mjx.ActionType.DUMMY:
            return legal_actions[0]

        return self._inner_act(observation)

    def _inner_act(
        self,
        observation: MjxObservation
        ) -> MjxAction:

        inferred = sorted(self._infer(observation), key=lambda x: x.policy, reverse=True)

        if self.temperature == None:    # 最善手を選ぶ
            selected = inferred[0]
        else:                           # policyのsoftmaxで選ぶ
            probabilites = self.softmax(inferred)
            selected = random.choices(inferred, k=1, weights=probabilites)[0]
            # print(f"{[entry.policy for entry in inferred]} -> {probabilites}")
            # board = Board.from_mjx(observation)
            # print([(Action.from_mjx(inf.action, board), prob) for inf, prob in zip(inferred, probabilites)])

        self.trainer.add(selected)

        return selected.action

    def export(
        self,
        dir_path: str,
        file_size: int,
        bin_num: int
    ):
        self.trainer.export(dir_path, file_size, bin_num)

    def end_round(
        self,
        score_diff: int,
        yaku: list[bool]
    ):
        self.trainer.end(score_diff, yaku)

    def end_game(
        self,
        final_score: int,
        final_rank: int
    ):
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

        out = self.model(board_indexes, board_offsets, action_indexes, action_offsets)

        return [Inferred(action, v_board, v_action, outlist[0], outlist[1:-1], outlist[-1]) for action, v_board, v_action, outlist in zip(legal_actions, board_vectors, action_vectors, out.tolist(), strict=True)]

class ShantenActor(Actor):
    def __init__(
        self,
        model: Model
        ) -> None:
        super().__init__(model)
        self.shanten_agent = MjxShantenAgent()

    def _inner_act(
        self,
        observation: MjxObservation
        ) -> MjxAction:

        selected = self.shanten_agent.act(observation)
        inferred = self._infer(observation)

        for entry in inferred:
            if entry.action == selected:
                self.trainer.add(entry)
                return entry.action

        raise Exception("selected action not found")
