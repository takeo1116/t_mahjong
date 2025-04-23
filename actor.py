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
from feature import FeatureVector, BoardFeature, DiscardActionFeature, OptionalActionFeature
from model import Model

@unique
class DataType(IntEnum):
    DISCARD = 0
    OPTIONAL = 1

class DiscardTrainingData:
    def __init__(
        self,
        board_vector: FeatureVector,
        action_vector: FeatureVector,
        value_inferred: float,
        discard_inferred: list[float],
        discard_index: int,
        valid_mask: list[float],
        value_label: float = None,
        yaku_label: list[bool] = None,
        policy_label: float = None,
        score_label: list[float] = None,
    ):
        self.data_type = DataType.DISCARD
        self.board_vector = board_vector
        self.action_vector = action_vector
        self.value_inferred = value_inferred
        self.discard_inferred = discard_inferred
        self.discard_index = discard_index
        self.valid_mask = valid_mask
        self.value_label = value_label
        self.yaku_label = yaku_label
        self.policy_label = policy_label
        self.score_label = score_label
    
    # def to_bytes(self):
    #     b = struct.pack("H", len(self.board_vector))
    #     for idx in self.board_vector.get_indexes():
    #         b += struct.pack("H", idx)
    #     b += struct.pack("H", len(self.action_vector))
    #     for idx in self.action_vector.get_indexes():
    #         b += struct.pack("H", idx)
    #     b += struct.pack("d", self.value_inferred)
    #     b += struct.pack("H", len(self.discard_inferred))
    #     for val in self.discard_inferred:
    #         b += struct.pack("d", val)
    #     b += struct.pack("H", self.discard_index)
    #     b += struct.pack("H", len(self.valid_mask))
    #     for val in self.valid_mask:
    #         b += struct.pack("?", val)
    #     b += struct.pack("d", self.value_label)
    #     b += struct.pack("H", len(self.yaku_label))
    #     for lab in self.yaku_label:
    #         b += struct.pack("?", lab)
    #     b += struct.pack("d", self.policy_label)
    #     b += struct.pack("H", len(self.score_label))
    #     for val in self.score_label:
    #         b += struct.pack("d", val)
    #     return b

    # @classmethod
    # def from_bytes(
    #     cls,
    #     buffer: BytesIO
    # ):
    #     len_bv, = struct.unpack("H", buffer.read(2))
    #     board_vector = FeatureVector([struct.unpack("H", buffer.read(2))[0] for _ in range(len_bv)])
    #     len_av, = struct.unpack("H", buffer.read(2))
    #     action_vector = FeatureVector([struct.unpack("H", buffer.read(2))[0] for _ in range(len_av)])
    #     value_inferred, = struct.unpack("d", buffer.read(8))
    #     len_di, = struct.unpack("H", buffer.read(2))
    #     discard_inferred = [struct.unpack("d", buffer.read(8))[0] for _ in range(len_di)]
    #     discard_index, = struct.unpack("H", buffer.read(2))
    #     len_vm, = struct.unpack("H", buffer.read(2))
    #     valid_mask = [struct.unpack("?", buffer.read(1))[0] for _ in range(len_vm)]
    #     value_label, = struct.unpack("d", buffer.read(8))
    #     len_yl, = struct.unpack("H", buffer.read(2))
    #     yaku_label = [struct.unpack("?", buffer.read(1))[0] for _ in range(len_yl)]
    #     policy_label, = struct.unpack("d", buffer.read(8))
    #     len_sl, = struct.unpack("H", buffer.read(2))
    #     score_label = [struct.unpack("d", buffer.read(8))[0] for _ in range(len_sl)]
    #     return cls(board_vector, action_vector, value_inferred, discard_inferred, discard_index, valid_mask, value_label, yaku_label, policy_label, score_label)

class OptionalTrainingData:
    def __init__(
        self,
        board_vector: FeatureVector,
        action_vector: FeatureVector,
        value_inferred: float,
        value_label: float = None,
        yaku_label: list[bool] = None,
        policy_label: float = None,
        score_label: list[float] = None,
    ):
        self.data_type = DataType.OPTIONAL
        self.board_vector = board_vector
        self.action_vector = action_vector
        self.value_inferred = value_inferred
        self.value_label = value_label
        self.yaku_label = yaku_label
        self.policy_label = policy_label
        self.score_label = score_label
    
    # def to_bytes(self):
    #     b = struct.pack("H", len(self.board_vector))
    #     for idx in self.board_vector.get_indexes():
    #         b += struct.pack("H", idx)
    #     b += struct.pack("H", len(self.action_vector))
    #     for idx in self.action_vector.get_indexes():
    #         b += struct.pack("H", idx)
    #     b += struct.pack("d", self.value_inferred)
    #     b += struct.pack("d", self.value_label)
    #     b += struct.pack("H", len(self.yaku_label))
    #     for lab in self.yaku_label:
    #         b += struct.pack("?", lab)
    #     b += struct.pack("d", self.policy_label)
    #     b += struct.pack("H", len(self.score_label))
    #     for val in self.score_label:
    #         b += struct.pack("d", val)
    #     return b

    # @classmethod
    # def from_bytes(
    #     cls,
    #     buffer: BytesIO
    # ):
    #     len_bv, = struct.unpack("H", buffer.read(2))
    #     board_vector = FeatureVector([struct.unpack("H", buffer.read(2))[0] for _ in range(len_bv)])
    #     len_av, = struct.unpack("H", buffer.read(2))
    #     action_vector = FeatureVector([struct.unpack("H", buffer.read(2))[0] for _ in range(len_av)])
    #     value_inferred, = struct.unpack("d", buffer.read(8))
    #     value_label, = struct.unpack("d", buffer.read(8))
    #     len_yl, = struct.unpack("H", buffer.read(2))
    #     yaku_label = [struct.unpack("?", buffer.read(1))[0] for _ in range(len_yl)]
    #     policy_label, = struct.unpack("d", buffer.read(8))
    #     len_sl, = struct.unpack("H", buffer.read(2))
    #     score_label = [struct.unpack("d", buffer.read(8))[0] for _ in range(len_sl)]
    #     return cls(board_vector, action_vector, value_inferred, value_label, yaku_label, policy_label, score_label)

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
        data: list[DiscardTrainingData] | list[OptionalTrainingData]
    ):
        self.data.extend(data)
    
    def get_data(self):
        return self.data

    # def export(
    #     self,
    #     file_path: str
    # ):
    #     b = b"".join([td.to_bytes() for td in self.data])
    #     with open(file_path, "wb") as f:
    #         f.write(b)

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
        action_indexes, action_offsets = [], []
        valid_masks = []
        for entry in self.data:
            board_offsets.append(len(board_indexes))
            action_offsets.append(len(action_indexes))
            board_indexes += entry.board_vector.get_indexes()
            action_indexes += entry.action_vector.get_indexes()
            valid_masks.append(entry.valid_mask)
        return board_indexes, board_offsets, action_indexes, action_offsets, valid_masks

    def make_labels(self, valid_masks):
        value_labels, yaku_labels, discard_idxes, policy_labels, score_labels = [], [], [], [], []
        for entry, mask in zip(self.data, valid_masks):
            valids = [idx for idx in range(len(mask)) if mask[idx]]
            value_labels.append(entry.value_label)
            yaku_labels.append(self.make_yaku_labels([1 if flag else 0 for flag in entry.yaku_label]))
            discard_idxes.append(entry.discard_index)

            policy_labels.append(entry.policy_label if entry.policy_label >= 0 else entry.policy_label * 0.3)

            score_labels.append(entry.score_label)
        return value_labels, yaku_labels, discard_idxes, policy_labels, score_labels

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
        value_labels, yaku_labels, policy_labels, score_labels = [], [], [], []
        for entry in self.data:
            value_labels.append(entry.value_label)
            yaku_labels.append(self.make_yaku_labels([1 if flag else 0 for flag in entry.yaku_label]))
            policy_labels.append(entry.policy_label)
            score_labels.append(entry.score_label)
        return value_labels, yaku_labels, policy_labels, score_labels

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
    DISCOUNT_RATE = 0.95                        # 報酬の割引率（value_inferredで埋める）
    RESULT_RATE = 0.1                           # valueにresultを反映させる割合
    POLICY_MAX = 0.4
    POLICY_MIN = -POLICY_MAX
    REWARD_ROUND_RATE = 0.3                     # 報酬全体のうち、局ごとのものの割合

    def __init__(self):
        super().__init__()
    
    def set_round_result(
        self,
        reward: float,
        yaku: list[bool],
        score_diffs: list[int],
    ):
        next_round_value = reward
        for entry in reversed(self.data):
            if entry.yaku_label is not None or entry.score_label is not None:
                break   # 前の局のデータに来たら終わり
            entry.yaku_label = yaku
            entry.score_label = [diff / 24000.0 for diff in score_diffs]
            round_value_label = next_round_value
            entry.value_label = round_value_label
            next_round_value = next_round_value * self.DISCOUNT_RATE + entry.value_inferred * (1.0 - self.DISCOUNT_RATE)

    def set_game_result(
        self,
        reward: float,
    ):
        next_game_value = reward
        for entry in reversed(self.data):
            if entry.yaku_label is None or entry.score_label is None:
                Exception("yaku/score label is not setted")
            if entry.value_label is None:
                Exception("value_label is None")
            round_value_label = entry.value_label
            game_value_label = next_game_value
            next_full_value = round_value_label * self.REWARD_ROUND_RATE + game_value_label * (1.0 - self.REWARD_ROUND_RATE)
            entry.value_label = next_full_value * (1.0 - self.RESULT_RATE) + reward * self.RESULT_RATE
            policy = max(self.POLICY_MIN, min(next_full_value - entry.value_inferred, self.POLICY_MAX))
            if entry.data_type == DataType.DISCARD:     # DISCARD
                entry.policy_label = policy
                # entry.policy_label = reward
            elif entry.data_type == DataType.OPTIONAL:  # OPTIONAL
                entry.policy_label = policy
                # entry.policy_label = reward
            else:
                Exception("unexpected data type")
            next_game_value = next_game_value * self.DISCOUNT_RATE + entry.value_inferred * (1.0 - self.DISCOUNT_RATE)

    # def result(
    #     self,
    #     reward: int,
    #     yaku: list[bool],
    #     score_diffs: list[int]
    # ):
    #     next_value = reward
    #     for entry in reversed(self.data):
    #         entry.value_label = (next_value + entry.value_inferred) / 2.0 * (1.0 - self.RESULT_RATE) + reward * self.RESULT_RATE
    #         entry.yaku_label = yaku
    #         entry.score_label = [diff / 24000.0 for diff in score_diffs]
    #         if entry.data_type == DataType.DISCARD: # DISCARD
    #             policy = max(self.POLICY_MIN, min(next_value - entry.value_inferred, self.POLICY_MAX))
    #             entry.policy_label = policy
    #             # entry.policy_label = reward
    #         else:                                   # OPTIONAL
    #             policy = max(self.POLICY_MIN, min(next_value - entry.value_inferred, self.POLICY_MAX))
    #             entry.policy_label = policy
    #             # entry.policy_label = reward
    #         next_value = next_value * self.DISCOUNT_RATE + entry.value_inferred * (1.0 - self.DISCOUNT_RATE)
    #         # print(entry.value_inferred, next_value)

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
        board: Board,
        board_vector: FeatureVector,
        action_vector: FeatureVector,
        valid_mask: list[bool],
        mjx_actions: list[MjxAction],
        value: float,
        yaku: list[float],
        discard: list[float],
        probs: list[float],
    ):
        if len(yaku) != 3 or len(discard) != 34 or len(valid_mask) != 34:
            raise Exception("unexpected length of list")
        self.board = board
        self.board_vector = board_vector
        self.action_vector = action_vector
        self.valid_mask = valid_mask
        self.value = value
        self.yaku = yaku
        self.mjx_actions = mjx_actions
        self.discard = discard
        self.probs = probs
        self.selected_idx = None    # self.select()で最後に選択された牌のindex

    # def get_illegal_indexes(
    #     self
    # ):
    #     # 手牌にない牌のindexのリストを返す
    #     tiles = [Tile.from_mjx(mjx_action.tile()) for mjx_action in self.mjx_actions]
    #     legal = [tile.tile_kind for tile in tiles]
    #     return [idx for idx in range(34) if idx not in legal]

    def set_selected(
        self,
        action: Action
    ):
        # 学習用、selectedを直接セットする
        self.selected_idx = action.discard_tile.tile_kind

    # def softmax(
    #     self,
    #     out: list[float],
    #     temperature: float = 1.0
    # ):
    #     x = np.array(out)
    #     exp_x = np.exp(x / temperature)
    #     probs = exp_x / np.sum(exp_x)
    #     return probs.tolist()

    def select(
        self,
        temperature: float = None
    ):
        # 可能なアクションだけのlistを作る
        table = [None for _ in range(34)]
        # 通常と赤があるとき、通常から捨てる
        for mjx_action in self.mjx_actions:
            action = Action.from_mjx(mjx_action, self.board)
            if action.action_kind != ActionKind.DISCARD:
                raise Exception(f"unexpected action kind: {action.action_kind}")
            tile_idx = action.discard_tile.tile_kind
            if (table[tile_idx] is None) or (not action.discard_tile.red):
                table[tile_idx] = mjx_action
        
        actionlist, out, probs = [], [], []
        for idx, mjx_action in enumerate(table):
            if mjx_action is None:
                continue
            actionlist.append((idx, mjx_action))
            out.append(self.discard[idx])
            probs.append(self.probs[idx])

        # 温度がNoneなら最大値を返す
        if temperature == None:
            idx, mjx_action = actionlist[out.index(max(out))]
            self.selected_idx = idx
            return mjx_action

        # selectedに選んだindexを入れる
        idx, mjx_action = random.choices(actionlist, k=1, weights=probs)[0]
        self.selected_idx = idx
        return mjx_action

    def __repr__(self):
        def mark_val(idx, val, legal, effective):
            marked = "{:.5f}".format(val)
            if idx in effective:
                marked = f"*{marked}*"
            if idx in legal:
                marked = f"[{marked}]"
            return marked
        temperature = 1.0
        actions = [Action.from_mjx(mjx_action, self.board) for mjx_action in self.mjx_actions]
        legal = [action.discard_tile.tile_kind for action in actions]
        policies = sorted([(TileKind(idx).name, val) for idx, val in enumerate(self.probs) if val > 0.0001], reverse=True, key=lambda x:x[1])
        string = "{" + f"value: {self.value}, temp: {temperature}, probs: {policies}" + "}"
        return string

class OptionalInferred:
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
        self.discard_bin_idx = 0
        self.optional_bin_idx = 0

    def add_discard(
        self,
        inferred: DiscardInferred
    ):
        if inferred.selected_idx is None:
            raise Exception("selected_idx is None")
        training_data = DiscardTrainingData(inferred.board_vector, inferred.action_vector, inferred.value, inferred.discard, inferred.selected_idx, inferred.valid_mask)
        self.current_episode.add(training_data)
    
    def add_optional(
        self,
        inferred: OptionalInferred
    ):
        training_data = OptionalTrainingData(inferred.board_vector, inferred.action_vector, inferred.value)
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

    # def end(
    #     self,
    #     score_diffs: list[int],
    #     rank: int,
    #     yaku: list[bool]
    # ):
    #     self.current_episode.result([1.0, 0.5, -0.5, -1.0][rank - 1], yaku, score_diffs)
    #     self.episodes.append(self.current_episode)
    #     self.current_episode = Episode()

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
            # dataset.export(os.path.join(dir_path, f"discard_{self.discard_bin_idx}/data_{r}.dat"))
            with open(os.path.join(dir_path, f"discard_{self.discard_bin_idx}/data_{r}.pkl"), "wb") as f:
                pickle.dump(dataset, f)
            idx += file_size
            self.discard_bin_idx = (self.discard_bin_idx + 1) % bin_num
        self.extra_data = Dataset(discard[idx:])

        idx, size = 0, len(optional)
        while size - idx >= file_size:
            r = random.getrandbits(64)
            dataset = Dataset(optional[idx:idx+file_size])
            # dataset.export(os.path.join(dir_path, f"optional_{self.optional_bin_idx}/data_{r}.dat"))
            with open(os.path.join(dir_path, f"optional_{self.optional_bin_idx}/data_{r}.pkl"), "wb") as f:
                pickle.dump(dataset, f)
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

    # @staticmethod
    # def calc_reward(
    #     result: int
    # ):
    #     if result == 0:  # 点数移動なし
    #         return -0.1
    #     elif result > 0: # 収支プラス
    #         if result >= 48000:
    #             return 1.0
    #         elif result >= 32000:
    #             return 0.95
    #         elif result >= 24000:
    #             return 0.9
    #         elif result >= 16000:
    #             return 0.85
    #         elif result >= 12000:
    #             return 0.8
    #         elif result >= 8000:
    #             return 0.7
    #         elif result >= 4000:
    #             return 0.6
    #         elif result >= 2000:
    #             return 0.5
    #         else:
    #             return 0.4
    #     else:           # 収支マイナス
    #         if result <= -48000:
    #             return -1.0
    #         elif result <= -32000:
    #             return -0.95
    #         elif result <= -24000:
    #             return -0.9
    #         elif result <= -16000:
    #             return -0.8
    #         elif result <= -12000:
    #             return -0.7
    #         elif result <= -8000:
    #             return -0.6
    #         elif result <= -4000:
    #             return -0.5
    #         elif result <= -2000:
    #             return -0.3
    #         else:
    #             return -0.2

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
    MODE = 1
    LOG_FILE = None

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
        inferred: list[OptionalInferred]
    ):
        if self.temperature == None:
            raise Exception("temperature is None")
        else:
            x = np.array([entry.policy for entry in inferred])
            exp_x = np.exp(x / self.temperature)
            p = exp_x / np.sum(exp_x)
            return p.tolist()
    
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

        if self.MODE == 1:
            return self._inner_act(observation)
        else:
            return self._inner_act2(observation)
    
    def check(
        self,
        observation: MjxObservation
        ):

        # DUMMYはすぐ返す
        legal_actions = observation.legal_actions()
        if len(legal_actions) >= 1 and legal_actions[0].type() == mjx.ActionType.DUMMY:
            print("DUMMY")
            return legal_actions[0]
        
        if self.MODE == 1:
            return self._inner_act(observation, dump=True)
        else:
            return self._inner_act2(observation)


    def _infer_discard(
        self,
        observation: MjxObservation,
        mjx_actions: list[MjxAction]
        ):

        board_indexes, board_offsets, action_indexes, action_offsets = [], [], [], []
        board_vectors, action_vectors = [], []
        valid_masks = []

        board = Board.from_mjx(observation)
        board_vector = BoardFeature.make(board)
        action_vector = DiscardActionFeature.make(board)
        
        board_vectors.append(board_vector)
        board_offsets.append(len(board_indexes))
        board_indexes += board_vector.get_indexes()
        action_vectors.append(action_vector)
        action_offsets.append(len(action_indexes))
        action_indexes += action_vector.get_indexes()

        board_indexes = torch.LongTensor(board_indexes)
        board_offsets = torch.LongTensor(board_offsets)
        action_indexes = torch.LongTensor(action_indexes)
        action_offsets = torch.LongTensor(action_offsets)

        valid_mask = [False for _ in range(34)]
        valids = []
        for mjx_action in mjx_actions:
            action = Action.from_mjx(mjx_action, board)
            valid_mask[action.discard_tile.tile_kind] = True
            valids.append(action.discard_tile.tile_kind)  # 
        valid_masks.append(valid_mask)
        valid_masks = torch.tensor(valid_masks, dtype=torch.bool)

        value, yaku, discard, score = self.model.forward_discard(board_indexes, board_offsets, action_indexes, action_offsets, valid_masks)
        probs = torch.nn.functional.softmax(discard, dim=-1)

        inferred = DiscardInferred(board, board_vector, action_vector, valid_mask, mjx_actions, value.tolist()[0][0], yaku.tolist()[0], discard.tolist()[0], probs.tolist()[0])
        
        if self.LOG_FILE is not None:
            self.dump(self.LOG_FILE, f"{inferred}")

        return inferred

    def _infer_optional(
        self,
        observation: MjxObservation,
        mjx_actions: list[MjxAction],
        no: bool = False    # NOがないとき、NOを追加する
        ):

        add_no = no
        board_indexes, board_offsets, action_indexes, action_offsets = [], [], [], []
        board_vectors, action_vectors = [], []

        board = Board.from_mjx(observation)
        board_vector = BoardFeature.make(board)

        for mjx_action in mjx_actions:
            action = Action.from_mjx(mjx_action, board)
            action_vector = OptionalActionFeature.make(action, board)
            action_offsets.append(len(action_indexes))
            action_indexes += action_vector.get_indexes()
            action_vectors.append(action_vector)
            if action.action_kind == ActionKind.NO:
                add_no = False
        if add_no:
            mjx_actions.append(None)
            action = Action.no()
            action_vector = OptionalActionFeature.make(action, board)
            action_offsets.append(len(action_indexes))
            action_indexes += action_vector.get_indexes()
            action_vectors.append(action_vector)
            add_no = False

        for _ in zip(action_offsets, action_vectors, strict=True):
            board_offsets.append(len(board_indexes))
            board_indexes += board_vector.get_indexes()
            board_vectors.append(board_vector)

        board_indexes = torch.LongTensor(board_indexes)
        board_offsets = torch.LongTensor(board_offsets)
        action_indexes = torch.LongTensor(action_indexes)
        action_offsets = torch.LongTensor(action_offsets)

        value, yaku, policy, score = self.model.forward_optional(board_indexes, board_offsets, action_indexes, action_offsets)

        inferred = [OptionalInferred(board, board_vector, mjx_action, action_vector, v[0], y, p[0]) for mjx_action, action_vector, v, y, p in zip(mjx_actions, action_vectors, value.tolist(), yaku.tolist(), policy.tolist(), strict=True)]

        if self.LOG_FILE is not None:
            self.dump(self.LOG_FILE, f"{inferred}")

        return inferred

    def _inner_act(
        self,
        observation: MjxObservation,
        dump: bool = False
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

        # 打牌以外を推論する
        if len(optional_actions) > 0:
            optional_inferred = sorted(self._infer_optional(observation, optional_actions, no=can_discard), key=lambda x:x.policy, reverse=True)
            if dump:
                print(optional_inferred)
            
            optional = None
            if self.temperature == None:    # 最善手を選ぶ
                optional = optional_inferred[0]
            else:
                probabilities = self.softmax(optional_inferred)
                optional = random.choices(optional_inferred, k=1, weights=probabilities)[0]

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
        discard_inferred = self._infer_discard(observation, discard_actions)
        if dump:
            print(discard_inferred.board.players[0].hand)
            print(discard_inferred.board.effective_discard)
            print(discard_inferred)
        mjx_action = discard_inferred.select(self.temperature)
        self.trainer.add_discard(discard_inferred)
        return mjx_action

    def _inner_act2(
        self,
        observation: MjxObservation,
        dump: bool = False
        ):

        all_actions = [mjx_action for mjx_action in observation.legal_actions()]
        inferred = sorted(self._infer_optional(observation, all_actions, no=False), key=lambda x:x.policy, reverse=True)

        if self.temperature == None:    # 最善手を選ぶ
            selected = inferred[0]
        else:
            probabilities = self.softmax(inferred)
            selected = random.choices(inferred, k=1, weights=probabilities)[0]

        self.trainer.add_optional(selected)
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

class ShantenActor(Actor):
    def __init__(
        self,
        model: Model
        ) -> None:
        super().__init__(model)
        self.mjx_agent = MjxShantenAgent()

    def _inner_act(
        self,
        observation: MjxObservation
        ) -> MjxAction:
        # MjxShantenAgentは副露できるときは必ず副露する
        mjx_selected = self.mjx_agent.act(observation)

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
        
        # 打牌以外を推論する
        if len(optional_actions) > 0:
            optional_inferred = sorted(self._infer_optional(observation, optional_actions, no=can_discard), key=lambda x:x.policy, reverse=True)

            for entry in optional_inferred:
                if entry.mjx_action is None:
                    if selected.action_kind == ActionKind.DISCARD:
                        self.trainer.add_optional(entry)
                        break
                elif entry.mjx_action == mjx_selected:
                    self.trainer.add_optional(entry)
                    return mjx_selected

        # 打牌を推論する
        discard_inferred = self._infer_discard(observation, discard_actions)
        discard_inferred.set_selected(selected)
        self.trainer.add_discard(discard_inferred)
        return mjx_selected

    def _inner_act2(
        self,
        observation: MjxObservation,
        dump: bool = False
        ):

        selected = self.mjx_agent.act(observation)
        all_actions = [mjx_action for mjx_action in observation.legal_actions()]
        inferred = self._infer_optional(observation, all_actions, no=False)

        for entry in inferred:
            if entry.mjx_action == selected:
                self.trainer.add_optional(entry)
                return entry.mjx_action

class MenzenActor(Actor):
    def __init__(
        self,
        model: Model
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
        
        # 打牌以外を推論する
        if len(optional_actions) > 0:
            optional_inferred = sorted(self._infer_optional(observation, optional_actions, no=can_discard), key=lambda x:x.policy, reverse=True)

            for entry in optional_inferred:
                if entry.mjx_action is None:
                    if selected.action_kind == ActionKind.DISCARD:
                        self.trainer.add_optional(entry)
                        break
                elif entry.mjx_action == mjx_selected:
                    self.trainer.add_optional(entry)
                    return mjx_selected

        # 打牌を推論する
        discard_inferred = self._infer_discard(observation, discard_actions)
        discard_inferred.set_selected(selected)
        self.trainer.add_discard(discard_inferred)
        return mjx_selected

    def _inner_act2(
        self,
        observation: MjxObservation,
        dump: bool = False
        ):

        selected = self._select_mjx_action(observation)
        all_actions = [mjx_action for mjx_action in observation.legal_actions()]
        inferred = self._infer_optional(observation, all_actions, no=False)

        for entry in inferred:
            if entry.mjx_action == selected:
                self.trainer.add_optional(entry)
                return entry.mjx_action
