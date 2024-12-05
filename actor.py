import os
import random
import struct
import numpy as np
import torch
from io import BytesIO
from collections import deque
from enum import IntEnum, IntFlag, unique

import mjx
from mjx.agents import Agent as MjxAgent
from mjx.action import Action as MjxAction
from mjx.observation import Observation as MjxObservation
from mjx.agents import ShantenAgent as MjxShantenAgent

from board import Board, Action, ActionKind, TileKind, Tile
from feature import FeatureVector, BoardFeature, ActionFeature
from model import Model

@unique
class DataType(IntEnum):
    DISCARD = 0
    OPTIONAL = 1

class DiscardTrainingData:
    def __init__(
        self,
        board_vector: FeatureVector,
        value_inferred: float,
        discard_index: int,
        illegal_indexes: list[int],
        value_label: float = None,
        yaku_label: list[bool] = None,
        policy_label: float = None,
    ):
        self.data_type = DataType.DISCARD
        self.board_vector = board_vector
        self.value_inferred = value_inferred
        self.discard_index = discard_index
        self.illegal_indexes = illegal_indexes
        self.value_label = value_label
        self.yaku_label = yaku_label
        self.policy_label = policy_label
    
    def to_bytes(self):
        b = struct.pack("H", len(self.board_vector))
        for idx in self.board_vector.get_indexes():
            b += struct.pack("H", idx)
        b += struct.pack("d", self.value_inferred)
        b += struct.pack("H", self.discard_index)
        b += struct.pack("H", len(self.illegal_indexes))
        for idx in self.illegal_indexes:
            b += struct.pack("H", idx)
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
        value_inferred, = struct.unpack("d", buffer.read(8))
        discard_index, = struct.unpack("H", buffer.read(2))
        len_ii, = struct.unpack("H", buffer.read(2))
        illegal_indexes = [struct.unpack("H", buffer.read(2))[0] for _ in range(len_ii)]
        value_label, = struct.unpack("d", buffer.read(8))
        len_yl, = struct.unpack("H", buffer.read(2))
        yaku_label = [struct.unpack("?", buffer.read(1))[0] for _ in range(len_yl)]
        policy_label, = struct.unpack("d", buffer.read(8))
        return cls(board_vector, value_inferred, discard_index, illegal_indexes, value_label, yaku_label, policy_label)

class OptionalTrainingData:
    def __init__(
        self,
        board_vector: FeatureVector,
        action_vector: FeatureVector,
        value_inferred: float,
        value_label: float = None,
        yaku_label: list[bool] = None,
        policy_label: float = None,
    ):
        self.data_type = DataType.OPTIONAL
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

class Dataset:
    def __init__(
        self,
        data: list[DiscardTrainingData] | list[OptionalTrainingData] = None
    ):
        if data == None:
            data = []
        self.data = data
    
    def add(
        self,
        data: DiscardTrainingData | OptionalTrainingData
    ):
        self.data.append(data)

    def addrange(
        self,
        data: DiscardTrainingData | OptionalTrainingData
    ):
        self.data.extend(data)
    
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
        size: int,
    ):
        # size個ごとに区切って新しいDatasetにする
        # 余りは無視する
        fullsize = len(self.data)
        datasets = [Dataset(self.data[idx-size:idx]) for idx in range(size, fullsize, size)]
        return datasets

    def make_inputs(self):
        raise NotImplementedError()

    def make_labels(self):
        raise NotImplementedError()
        
    def from_bytes(
        cls,
        b: bytes
    ):
        raise NotImplementedError()

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

class DiscardDataset(Dataset):
    def __init__(
        self,
        data: list[DiscardTrainingData] = None
    ):
        super().__init__(data)

    def add(
        self,
        data: DiscardTrainingData
    ):
        if data.data_type != DataType.DISCARD:
            raise Exception("DataType is not matched")
        self.data.append(data)

    def make_inputs(self):
        board_indexes, board_offsets = [], []
        for entry in self.data:
            board_offsets.append(len(board_indexes))
            board_indexes += entry.board_vector.get_indexes()
        return board_indexes, board_offsets
    
    def make_labels(self):
        value_labels, yaku_labels, discard_labels = [], [], []
        illegal_labels = []
        for entry in self.data:
            value_labels.append(entry.value_label)
            yaku_labels.append(self.make_yaku_labels([1 if flag else 0 for flag in entry.yaku_label]))
            discard_labels.append([(entry.policy_label if entry.policy_label > 0 else entry.policy_label / 10) if idx == entry.discard_index else 0.0 for idx in range(34)])
            illegal_labels.append([1.0 if idx in entry.illegal_indexes else 0.0 for idx in range(34)])
        return value_labels, yaku_labels, discard_labels, illegal_labels

    def split(
        self,
        size: int,
    ):
        # size個ごとに区切って新しいDatasetにする
        # 余りは無視する
        fullsize = len(self.data)
        datasets = [DiscardDataset(self.data[idx-size:idx]) for idx in range(size, fullsize, size)]
        return datasets

    @classmethod
    def from_bytes(
        cls,
        b: bytes
    ):
        size = len(b)
        buffer = BytesIO(b)
        data = []
        while buffer.tell() < size:
            data.append(DiscardTrainingData.from_bytes(buffer))
        return cls(data)

class OptionalDataset(Dataset):
    def __init__(
        self,
        data: list[OptionalTrainingData] = None
    ):
        super().__init__(data)

    def add(
        self,
        data: OptionalTrainingData
    ):
        if data.data_type != DataType.OPTIONAL:
            raise Exception("DataType is not matched")
        self.data.append(data)

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
            yaku_labels.append(self.make_yaku_labels([1 if flag else 0 for flag in entry.yaku_label]))
            policy_labels.append(entry.policy_label)
        return value_labels, yaku_labels, policy_labels

    def split(
        self,
        size: int,
    ):
        # size個ごとに区切って新しいDatasetにする
        # 余りは無視する
        fullsize = len(self.data)
        datasets = [OptionalDataset(self.data[idx-size:idx]) for idx in range(size, fullsize, size)]
        return datasets

    @classmethod
    def from_bytes(
        cls,
        b: bytes
    ):
        size = len(b)
        buffer = BytesIO(b)
        data = []
        while buffer.tell() < size:
            data.append(OptionalTrainingData.from_bytes(buffer))
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

class BaseInferred:
    def __init__(
        self,
        board: Board,
        board_vector: FeatureVector,
        value: float,
        yaku: list[float],
        b_to_d,
        b_to_p
    ):
        if len(yaku) != 3:
            raise Exception("unexpected length of list")
        self.board = board
        self.board_vector = board_vector
        self.value = value
        self.yaku = yaku
        self.b_to_d = b_to_d
        self.b_to_p = b_to_p

class DiscardInferred:
    def __init__(
        self,
        mjx_actions: list[MjxAction],
        base_inferred: BaseInferred,
        discard: list[float],
    ):
        if len(discard) != 34:
            raise Exception("unexpected length of list")
        self.mjx_actions = mjx_actions
        self.base_inferred = base_inferred
        self.discard = discard
        self.selected_idx = None    # self.select()で最後に選択された牌のindex

    def get_illegal_indexes(
        self
    ):
        # 手牌にない牌のindexのリストを返す
        tiles = [Tile.from_mjx(mjx_action.tile()) for mjx_action in self.mjx_actions]
        legal = [tile.tile_kind for tile in tiles]
        return [idx for idx in range(34) if idx not in legal]

    def set_selected(
        self,
        action: Action
    ):
        # 学習用、selectedを直接セットする
        self.selected_idx = action.discard_tile.tile_kind

    def select(
        self,
        temperature: float = None
    ):
        # 可能なアクションだけのlistを作る
        table = [None for _ in range(34)]
        # 通常と赤があるとき、通常から捨てる
        for mjx_action in self.mjx_actions:
            action = Action.from_mjx(mjx_action, self.base_inferred.board)
            if action.action_kind != ActionKind.DISCARD:
                raise Exception(f"unexpected action kind: {action.action_kind}")
            tile_idx = action.discard_tile.tile_kind
            if (table[tile_idx] is None) or (not action.discard_tile.red):
                table[tile_idx] = mjx_action
        
        actionlist, probs = [], []
        for idx, mjx_action in enumerate(table):
            if mjx_action is None:
                continue
            actionlist.append((idx, mjx_action))
            probs.append(self.discard[idx])

        # 温度がNoneなら最大値を返す
        if temperature == None:
            idx, mjx_action = actionlist[probs.index(max(probs))]
            self.selected_idx = idx
            return mjx_action

        x = np.array(probs)
        exp_x = np.exp(x / temperature)
        p = exp_x / np.sum(exp_x)

        # selectedに選んだindexを入れる
        idx, mjx_action = random.choices(actionlist, k=1, weights=p)[0]
        self.selected_idx = idx
        return mjx_action

class OptionalInferred:
    def __init__(
        self,
        mjx_action: MjxAction,
        base_inferred: BaseInferred,
        action_vector: FeatureVector,
        policy: float
    ):
        self.mjx_action = mjx_action
        self.base_inferred = base_inferred
        self.action_vector = action_vector
        self.policy = policy

class Trainer:
    def __init__(self):
        self.episodes = []
        self.current_episode = Episode()
        self.extra_data = Dataset()
        self.discard_bin_idx = 0
        self.optional_bin_idx = 0

    def add_discard(
        self,
        inferred: DiscardInferred
    ):
        if inferred.selected_idx is None:
            raise Exception("selected_idx is None")
        training_data = DiscardTrainingData(inferred.base_inferred.board_vector, inferred.base_inferred.value, inferred.selected_idx, inferred.get_illegal_indexes())
        self.current_episode.add(training_data)
    
    def add_optional(
        self,
        inferred: OptionalInferred
    ):
        training_data = OptionalTrainingData(inferred.base_inferred.board_vector, inferred.action_vector, inferred.base_inferred.value)
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

        discard, optional = [], []
        for data in alldata:
            if data.data_type == DataType.DISCARD:
                discard.append(data)
            elif data.data_type == DataType.OPTIONAL:
                optional.append(data)
            else:
                raise Exception(f"unexpected datatype: {data.data_type}")
        
        idx, size = 0, len(discard)
        while size - idx >= file_size:
            r = random.getrandbits(64)
            dataset = Dataset(discard[idx:idx+file_size])
            dataset.export(os.path.join(dir_path, f"discard_{self.discard_bin_idx}/data_{r}.dat"))
            idx += file_size
            self.discard_bin_idx = (self.discard_bin_idx + 1) % bin_num
        self.extra_data = Dataset(discard[idx:])

        idx, size = 0, len(optional)
        while size - idx >= file_size:
            r = random.getrandbits(64)
            dataset = Dataset(optional[idx:idx+file_size])
            dataset.export(os.path.join(dir_path, f"optional_{self.optional_bin_idx}/data_{r}.dat"))
            idx += file_size
            self.optional_bin_idx = (self.optional_bin_idx + 1) % bin_num
        self.extra_data.addrange(optional[idx:])
        self.episodes.clear()
        
    def reset(self):
        self.episodes.clear()
        self.current_episode = Episode()
    
    @staticmethod
    def calc_reward(
        result: int
    ):
        reward = result / 24000
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

    def act(
        self,
        observation: MjxObservation
        ) -> MjxAction:

        # DUMMYはすぐ返す
        legal_actions = observation.legal_actions()
        if len(legal_actions) >= 1 and legal_actions[0].type() == mjx.ActionType.DUMMY:
            return legal_actions[0]

        return self._inner_act(observation)

    def _infer_base(
        self,
        observation: MjxObservation
    ):
        board_indexes, board_offsets = [], []
        board_vectors = []

        board = Board.from_mjx(observation)
        board_vector = BoardFeature.make(board)
        
        board_vectors.append(board_vector)
        board_offsets.append(len(board_indexes))
        board_indexes += board_vector.get_indexes()

        board_indexes = torch.LongTensor(board_indexes)
        board_offsets = torch.LongTensor(board_offsets)

        value, yaku, b_to_d, b_to_p = self.model.forward_base(board_indexes, board_offsets)
        inferred = BaseInferred(board, board_vector, value.tolist()[0][0], yaku.tolist()[0], b_to_d, b_to_p)

        return inferred

    def _infer_discard(
        self,
        base_inferred: BaseInferred,
        mjx_actions: list[MjxAction]
        ):
        
        out = self.model.forward_discard(base_inferred.b_to_d)

        inferred = DiscardInferred(mjx_actions, base_inferred, out.tolist()[0])

        return inferred

    def _infer_optional(
        self,
        base_inferred: BaseInferred,
        mjx_actions: list[MjxAction],
        no: bool = False    # NOがないとき、NOを追加する
        ):

        add_no = no
        action_indexes, action_offsets = [], []
        action_vectors = []

        board = base_inferred.board
        for mjx_action in mjx_actions:
            action = Action.from_mjx(mjx_action, board)
            action_vector = ActionFeature.make(action, board)
            action_offsets.append(len(action_indexes))
            action_indexes += action_vector.get_indexes()
            action_vectors.append(action_vector)
            if action.action_kind == ActionKind.NO:
                add_no = False
        if add_no:
            mjx_actions.append(None)
            action = Action.no()
            action_vector = ActionFeature.make(action, board)
            action_offsets.append(len(action_indexes))
            action_indexes += action_vector.get_indexes()
            action_vectors.append(action_vector)
            add_no = False

        action_indexes = torch.LongTensor(action_indexes)
        action_offsets = torch.LongTensor(action_offsets)

        out = self.model.forward_optional(action_indexes, action_offsets, base_inferred.b_to_p)

        return [OptionalInferred(action, base_inferred, action_vector, outlist[0]) for action, action_vector, outlist in zip(mjx_actions, action_vectors, out.tolist(), strict=True)]

    def _inner_act(
        self,
        observation: MjxObservation
        ):

        # 打牌とそれ以外のアクションを分ける
        discard_actions, optional_actions = [], []  # mjxのアクション
        board = Board.from_mjx(observation)
        for mjx_action in observation.legal_actions():
            if mjx_action.to_idx() <= 73:   # 打牌
                discard_actions.append(mjx_action)
            else:
                optional_actions.append(mjx_action)
        can_discard = len(discard_actions) > 0

        # value, yakuを推論する
        base_inferred = self._infer_base(observation)

        # 打牌以外を推論する
        if len(optional_actions) > 0:
            optional_inferred = sorted(self._infer_optional(base_inferred, optional_actions, no=can_discard), key=lambda x:x.policy, reverse=True)

            optional = optional_inferred[0]
            if optional.mjx_action is None:
                # NOかつdiscardに進む場合
                if not can_discard:
                    raise Exception("mjx_action is None")
                self.trainer.add_optional(optional)
            else:
                # 選んだactionを返す場合
                self.trainer.add_optional(optional)
                return optional.mjx_action

        # 打牌を推論する
        discard_inferred = self._infer_discard(base_inferred, discard_actions)
        mjx_action = discard_inferred.select()
        self.trainer.add_discard(discard_inferred)
        return mjx_action

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

        mjx_selected = self.shanten_agent.act(observation)

        # 打牌とそれ以外のアクションを分ける
        discard_actions, optional_actions = [], []  # mjxのアクション
        board = Board.from_mjx(observation)
        for mjx_action in observation.legal_actions():
            if mjx_action.to_idx() <= 73:   # 打牌
                discard_actions.append(mjx_action)
            else:
                optional_actions.append(mjx_action)
        can_discard = len(discard_actions) > 0

        selected = Action.from_mjx(mjx_selected, board)

        # value, yakuを推論する
        base_inferred = self._infer_base(observation)
        
        # 打牌以外を推論する
        if len(optional_actions) > 0:
            optional_inferred = sorted(self._infer_optional(base_inferred, optional_actions, no=can_discard), key=lambda x:x.policy, reverse=True)

            for entry in optional_inferred:
                if entry.mjx_action is None:
                    if selected.action_kind == ActionKind.DISCARD:
                        self.trainer.add_optional(entry)
                        break
                elif entry.mjx_action == mjx_selected:
                    self.trainer.add_optional(entry)
                    return mjx_selected

        # 打牌を推論する
        discard_inferred = self._infer_discard(base_inferred, discard_actions)
        discard_inferred.set_selected(selected)
        self.trainer.add_discard(discard_inferred)
        return mjx_selected
