import torch
import numpy as np
from collections import Counter

from board import Board, Tile, Hand, Player, Exposed, River, Action
from board import TileKind, ChiKind, SerialPairKind, ExposedKind, ActionKind, Wind, Relation

from feature.feature_vector import *
from feature.cnn_feature import *

class CnnMyPlayerFeature:
    SCORE = 0
    WIND = SCORE + ScoreFeature.SIZE
    RIICHI = WIND + Wind.SIZE
    SIZE = RIICHI + 1

    @classmethod
    def add(
        cls,
        v: FeatureVector,
        offset: int,
        board: Board
    ):
        player = board.players[Relation.ME]
        ScoreFeature.add(v, offset+cls.SCORE, player.score)
        v.add(offset+cls.WIND+player.wind)
        if player.riichi:
            v.add(offset+cls.RIICHI)

class CnnOpponentPlayerFeature:
    SCORE = 0
    SCORE_DIFF = SCORE + ScoreFeature.SIZE
    WIND = SCORE_DIFF + ScoreDiffFeature.SIZE
    RIICHI = WIND + Wind.SIZE
    SIZE = RIICHI + 1

    @classmethod
    def add(
        cls,
        v: FeatureVector,
        offset: int,
        relation: int,
        board: Board,
    ):
        player = board.players[relation]
        score = player.score
        ScoreFeature.add(v, offset+cls.SCORE, score)
        ScoreDiffFeature.add(v, offset+cls.SCORE_DIFF, score-board.players[Relation.ME].score)
        v.add(offset+cls.WIND+player.wind)
        if player.riichi:
            v.add(offset+cls.RIICHI)

class CnnBoardFeature:
    KYOKU = 0
    WIND = KYOKU + 8
    SHANTEN = WIND + Wind.SIZE
    PHASE = SHANTEN + ShantenFeature.SIZE
    MY_PLAYER = PHASE + PhaseFeature.SIZE
    SHIMO_PLAYER = MY_PLAYER + CnnMyPlayerFeature.SIZE
    TOIMEN_PLAYER = SHIMO_PLAYER + CnnOpponentPlayerFeature.SIZE
    KAMI_PLAYER = TOIMEN_PLAYER + CnnOpponentPlayerFeature.SIZE
    SIZE = KAMI_PLAYER + CnnOpponentPlayerFeature.SIZE

    @classmethod
    def make(
        cls,
        board: Board
    ):
        v = FeatureVector()

        kyoku = min(board.kyoku, 7)
        v.add(cls.KYOKU+kyoku)
        v.add(cls.WIND+board.wind)
        ShantenFeature.add(v, cls.SHANTEN, board.shanten)
        PhaseFeature.add(v, cls.PHASE, board.players[Relation.ME])
        CnnMyPlayerFeature.add(v, cls.MY_PLAYER, board)
        CnnOpponentPlayerFeature.add(v, cls.SHIMO_PLAYER, Relation.SHIMO, board)
        CnnOpponentPlayerFeature.add(v, cls.TOIMEN_PLAYER, Relation.TOIMEN, board)
        CnnOpponentPlayerFeature.add(v, cls.KAMI_PLAYER, Relation.KAMI, board)

        return v

class CnnBoardFeatures:
    CHANNEL = 13

    def __init__(
        self,
        pic: torch.Tensor,
        vec: FeatureVector,
    ):
        self.pic = pic
        self.vec = vec

    @classmethod
    def make_pic_np(
        cls,
        board: Board
    ):
        """
        自分の手牌  1x34x4
        副露        4x34x4
        捨て牌      4x34x4
        安全牌      4x34x1 -> 4x34x4
        """

        players = board.players

        # 自分の手牌
        my_hand_np = np.zeros((1, 34, 4), dtype=np.float32)
        for kind, cnt in enumerate(players[Relation.ME].hand.to_34_array_full()):
            my_hand_np[0, kind, range(cnt)] = 1.0
        # 副露
        exposed_np = np.zeros((4, 34, 4), dtype=np.float32)
        for relation in range(Relation.SIZE):
            for kind, cnt in enumerate(players[relation].hand.to_34_array_exposed()):
                if cnt > 4:
                    print(relation, kind, range(cnt))
                    print(players[relation].hand)
                exposed_np[relation, kind, range(cnt)] = 1.0
        # 捨て牌
        river_np = np.zeros((4, 34, 4), dtype=np.float32)
        for relation in range(Relation.SIZE):
            for kind, cnt in enumerate(players[relation].river.to_34_array()):
                river_np[relation, kind, range(cnt)] = 1.0
        # 安全牌
        safe_np = np.zeros((4, 34, 4), dtype=np.float32)
        for relation in range(Relation.SIZE):
            for kind in players[relation].safe:
                safe_np[relation, kind, range(4)] = 1.0
        
        full_np = np.concatenate([my_hand_np, exposed_np, river_np, safe_np], axis=0)
        return full_np

    @classmethod
    def make(
        cls,
        board: Board
    ):
        pic = cls.make_pic_np(board)
        vec = CnnBoardFeature.make(board)

        return cls(pic, vec)