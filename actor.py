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

from board import Board, Action, ActionKind, TileKind, Tile, Relation, ChiKind
from feature import FeatureVector, BoardFeature, DiscardActionFeature, OptionalActionFeature
from model import Model

@unique
class OptionalActionKind(IntEnum):
    CHI_LEFT = 0
    CHI_MIDDLE = 1
    CHI_RIGHT = 2
    PON = 3
    KAN_OPEN = 4
    KAN_CLOSE = 5
    KAN_ADD = 6
    TSUMO = 7
    RON = 8
    RIICHI = 9
    DRAW = 10
    NO = 11
    SIZE = 12

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

class OptionalTrainingData:
    def __init__(
        self,
        board_vector: FeatureVector,
        action_vector: FeatureVector,
        value_inferred: float,
        optional_inferred: list[float],
        optional_index: int,
        valid_mask: list[float],
        value_label: float = None,
        yaku_label: list[bool] = None,
        policy_label: float = None,
        score_label: list[float] = None,
    ):
        self.data_type = DataType.OPTIONAL
        self.board_vector = board_vector
        self.action_vector = action_vector
        self.value_inferred = value_inferred
        self.optional_inferred = optional_inferred
        self.optional_index = optional_index
        self.valid_mask = valid_mask
        self.value_label = value_label
        self.yaku_label = yaku_label
        self.policy_label = policy_label
        self.score_label = score_label

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
        valid_masks = []
        for entry in self.data:
            board_offsets.append(len(board_indexes))
            action_offsets.append(len(action_indexes))
            board_indexes += entry.board_vector.get_indexes()
            action_indexes += entry.action_vector.get_indexes()
            valid_masks.append(entry.valid_mask)
        return board_indexes, board_offsets, action_indexes, action_offsets, valid_masks

    def make_labels(self, valid_masks):
        value_labels, yaku_labels, optional_idxes, policy_labels, score_labels = [], [], [], [], []
        for entry, mask in zip(self.data, valid_masks):
            valids = [idx for idx in range(len(mask)) if mask[idx]]
            value_labels.append(entry.value_label)
            yaku_labels.append(self.make_yaku_labels([1 if flag else 0 for flag in entry.yaku_label]))
            optional_idxes.append(entry.optional_index)

            policy_labels.append(entry.policy_label if entry.policy_label >= 0 else entry.policy_label * 0.3)

            score_labels.append(entry.score_label)
        return value_labels, yaku_labels, optional_idxes, policy_labels, score_labels

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
                # print(f"D next: {next_full_value}, label: {entry.value_label}, inferred: {entry.value_inferred}, policy: {policy}, selected: {entry.discard_index}")
            elif entry.data_type == DataType.OPTIONAL:  # OPTIONAL
                entry.policy_label = policy
                # entry.policy_label = reward
                # print(f"O next: {next_full_value}, label: {entry.value_label}, inferred: {entry.value_inferred}, policy: {policy}, selected: {OptionalActionKind(entry.optional_index).name}")
            else:
                Exception("unexpected data type")
            next_game_value = next_game_value * self.DISCOUNT_RATE + entry.value_inferred * (1.0 - self.DISCOUNT_RATE)

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
        actions = [Action.from_mjx(mjx_action, self.board) for mjx_action in self.mjx_actions]
        policies = sorted([(TileKind(idx).name, val) for idx, val in enumerate(self.probs) if val > 0.0001], reverse=True, key=lambda x:x[1])
        string = "Discard: {" + f"value: {self.value}, probs: {policies}" + "}"
        return string

class OptionalInferred:

    def __init__(
        self,
        board: Board,
        board_vector: FeatureVector,
        action_vector: FeatureVector,
        valid_mask: list[bool],
        mjx_actions: list[MjxAction],
        value: float,
        yaku: list[float],
        optional: list[float],
        probs: list[float],
    ):
        if len(yaku) != 3:
            raise Exception("unexpected length of list")
        self.board = board
        self.board_vector = board_vector
        self.action_vector = action_vector
        self.valid_mask = valid_mask
        self.value = value
        self.yaku = yaku
        self.mjx_actions = mjx_actions
        self.optional = optional
        self.probs = probs
        self.selected_idx = None    # self.select()で最後に選択されたOptionalActionKindのindex

    @classmethod
    def get_optional_actionkind(
        cls,
        action: Action
    ):
        kind = action.action_kind
        if kind is None:
            raise Exception("unexpected action kind: None")
        elif kind == ActionKind.DISCARD:
            raise Exception(f"unexpected action kind: {action.action_kind}")
        elif kind == ActionKind.CHI:
            diff = None     # stealが順子の左なら0, 中央なら1, 右なら2
            if ChiKind.M123 <= action.chi_kind <= ChiKind.M789:
                M_idx = action.chi_kind - ChiKind.M123
                diff = action.steal_tile.tile_kind - TileKind.M2 - (action.chi_kind - ChiKind.M123) + 1
            elif ChiKind.P123 <= action.chi_kind <= ChiKind.P789:
                P_idx = action.chi_kind - ChiKind.P123
                diff = action.steal_tile.tile_kind - TileKind.P2 - (action.chi_kind - ChiKind.P123) + 1
            elif ChiKind.S123 <= action.chi_kind <= ChiKind.S789:
                S_idx = action.chi_kind - ChiKind.S123
                diff = action.steal_tile.tile_kind - TileKind.S2 - (action.chi_kind - ChiKind.S123) + 1
            return OptionalActionKind.CHI_LEFT + diff
        elif kind == ActionKind.PON:
            return OptionalActionKind.PON
        elif kind == ActionKind.KAN_OPEN:
            return OptionalActionKind.KAN_OPEN
        elif kind == ActionKind.KAN_CLOSE:
            return OptionalActionKind.KAN_CLOSE
        elif kind == ActionKind.KAN_ADD:
            return OptionalActionKind.KAN_ADD
        elif kind == ActionKind.TSUMO:
            return OptionalActionKind.TSUMO
        elif kind == ActionKind.RON:
            return OptionalActionKind.RON
        elif kind == ActionKind.RIICHI:
            return OptionalActionKind.RIICHI
        elif kind == ActionKind.DRAW:
            return OptionalActionKind.DRAW
        elif kind == ActionKind.NO:
            return OptionalActionKind.NO
        else:
            raise Exception("unexpected action kind")

    def set_selected_kind(
        self,
        kind: OptionalActionKind
    ):
        self.selected_idx = kind

    def set_selected(
        self,
        action: Action
    ):
        # 学習用、selectedを直接セットする
        kind = self.get_optional_actionkind(action)
        self.set_selected_kind(kind)
    
    def select(
        self,
        temperature: float = None
    ):
        # 可能なアクションだけのlistを作る
        table = [None for _ in range(OptionalActionKind.SIZE)]
        # TODO: 同じアクション同士で優先度をつける
        for mjx_action in self.mjx_actions:
            action = Action.from_mjx(mjx_action, self.board)
            kind = self.get_optional_actionkind(action)
            table[kind] = mjx_action

        actionlist, out, probs = [], [], []
        for idx, mjx_action in enumerate(table):
            if idx != OptionalActionKind.NO and mjx_action is None:
                continue
            actionlist.append((idx, mjx_action))
            out.append(self.optional[idx])
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
        actions = [Action.from_mjx(mjx_action, self.board) for mjx_action in self.mjx_actions]
        policies = sorted([(OptionalActionKind(idx).name, val) for idx, val in enumerate(self.probs) if val > 0.0001], reverse=True, key=lambda x:x[1])
        string = "Optional: {" + f"value: {self.value}, probs: {policies}" + "}"
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
        if inferred.selected_idx is None:
            raise Exception("selected_idx is None")
        training_data = OptionalTrainingData(inferred.board_vector, inferred.action_vector, inferred.value, inferred.optional, inferred.selected_idx, inferred.valid_mask)
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
            with open(os.path.join(dir_path, f"discard_{self.discard_bin_idx}/data_{r}.pkl"), "wb") as f:
                pickle.dump(dataset, f)
            idx += file_size
            self.discard_bin_idx = (self.discard_bin_idx + 1) % bin_num
        self.extra_data = Dataset(discard[idx:])

        idx, size = 0, len(optional)
        while size - idx >= file_size:
            r = random.getrandbits(64)
            dataset = Dataset(optional[idx:idx+file_size])
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
    LOG_FILE = None

    def __init__(
        self,
        model: Model,
        temperature: float = None,  # 引数なしでランダム性なし
        no_furo: bool = False       # Trueで副露（ポン、チー、大明槓、加槓）なし
        ) -> None:
        super().__init__()
        self.evaluator = Evaluator()
        self.trainer = Trainer()
        self.temperature = temperature
        self.model = model
        self.model.eval()
        self.no_furo = no_furo

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
        for mjx_action in mjx_actions:
            action = Action.from_mjx(mjx_action, board)
            valid_mask[action.discard_tile.tile_kind] = True
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
    ):
        board_indexes, board_offsets, action_indexes, action_offsets = [], [], [], []
        board_vectors, action_vectors = [], []
        valid_masks = []

        board = Board.from_mjx(observation)
        actions = [Action.from_mjx(mjx_action, board) for mjx_action in mjx_actions]
        board_vector = BoardFeature.make(board)
        action_vector = OptionalActionFeature.make(actions, board)

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

        valid_mask = [False for _ in range(OptionalActionKind.SIZE)]
        for action, mjx_action in zip(actions, mjx_actions):
            kind = OptionalInferred.get_optional_actionkind(action)
            valid_mask[kind] = True
        valid_mask[OptionalActionKind.NO] = True   # NOは常にできる想定

        valid_masks.append(valid_mask)
        valid_masks = torch.tensor(valid_masks, dtype=torch.bool)

        value, yaku, optional, score = self.model.forward_optional(board_indexes, board_offsets, action_indexes, action_offsets, valid_masks)
        probs = torch.nn.functional.softmax(optional, dim=-1)

        inferred = OptionalInferred(board, board_vector, action_vector, valid_mask, mjx_actions, value.tolist()[0][0], yaku.tolist()[0], optional.tolist()[0], probs.tolist()[0])

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

        # 打牌以外を推論する
        if len(optional_actions) > 0:
            optional_inferred = self._infer_optional(observation, optional_actions)
            optional_mjx_action = optional_inferred.select(self.temperature)

            if self.no_furo:    # 副露禁止のとき
                selected = OptionalActionKind(optional_inferred.selected_idx)
                if selected in [OptionalActionKind.CHI_LEFT, OptionalActionKind.CHI_MIDDLE, OptionalActionKind.CHI_RIGHT, OptionalActionKind.PON, OptionalActionKind.KAN_ADD, OptionalActionKind.KAN_OPEN]:
                    optional_inferred.set_selected_kind(OptionalActionKind.NO)
                    for mjx_action in optional_actions:
                        if mjx_action.to_idx() == 179:
                            return mjx_action
                    else:
                        optional_mjx_action = None

            self.trainer.add_optional(optional_inferred)

            if optional_mjx_action is not None: # 選んだactionを返す場合(打牌に遷移しない場合)
                return optional_mjx_action

        # 打牌を推論する
        discard_inferred = self._infer_discard(observation, discard_actions)
        if dump:
            print(discard_inferred.board.players[0].hand)
            print(discard_inferred.board.effective_discard)
            print(discard_inferred)
        discard_mjx_action = discard_inferred.select(self.temperature)
        self.trainer.add_discard(discard_inferred)
        return discard_mjx_action

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
            optional_inferred = self._infer_optional(observation, optional_actions)
            if selected.action_kind == ActionKind.DISCARD:
                optional_inferred.set_selected_kind(OptionalActionKind.NO)
            else:
                optional_inferred.set_selected(selected)
                return mjx_selected
            self.trainer.add_optional(optional_inferred)

        # 打牌を推論する
        discard_inferred = self._infer_discard(observation, discard_actions)
        discard_inferred.set_selected(selected)
        self.trainer.add_discard(discard_inferred)
        return mjx_selected

class ShantenActor(MenzenActor):
    def __init__(
        self,
        model: Model
        ) -> None:
        super().__init__(model)
        self.mjx_agent = MjxShantenAgent()

    def _selected_mjx_action(
        self,
        observation: MjxObservation
    ) -> MjxAction:
        return self.mjx_agent.act(observation)
