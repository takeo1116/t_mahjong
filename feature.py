import torch
from collections import Counter

from board import Board, Tile, Hand, Player, Exposed, River, Action
from board import TileKind, ChiKind, SerialPairKind, ExposedKind, ActionKind, Wind, Relation

class FeatureVector:
    def __init__(
        self,
        v: list[int] = None
    ):
        if v == None:
            v = []
        self.v = v

    def add(
        self,
        idx: int,
        val: float = 1.0
    ):
        self.v.append(idx)

    def get_indexes(self):
        return self.v

    def __len__(self):
        return len(self.v)

    def __repr__(self):
        return self.v.__repr__()

class ExposedFeature:
    CHI = 0
    PON = CHI + ChiKind.SIZE * 4
    KAN_OPEN = PON + TileKind.SIZE
    KAN_CLOSE = KAN_OPEN + TileKind.SIZE
    SIZE = KAN_CLOSE + TileKind.SIZE
    
    @classmethod
    def add(
        cls,
        v: FeatureVector,
        offset: int,
        hand: Hand
    ):
        chis = Counter([exposed.chi_kind for exposed in hand.exposed if exposed.exposed_kind == ExposedKind.CHI])
        for kind, cnt in chis.items():
            for i in range(cnt):
                v.add(offset+cls.CHI+kind+i)
        for exposed in hand.exposed:
            match exposed.exposed_kind:
                case ExposedKind.CHI:
                    pass
                case ExposedKind.PON:
                    v.add(offset+cls.PON+exposed.pon_kind)
                case ExposedKind.KAN_OPEN:
                    v.add(offset+cls.KAN_OPEN+exposed.kan_kind)
                case ExposedKind.KAN_CLOSE:
                    v.add(offset+cls.KAN_CLOSE+exposed.kan_kind)

class ShantenFeature:
    TEMPAI = 0
    ONE = TEMPAI + 1
    TWO = ONE + 1
    THREE = TWO + 1
    FOUR = THREE + 1
    OVER_FIVE = FOUR + 1
    SIZE = OVER_FIVE + 1

    @classmethod
    def add(
        cls,
        v: FeatureVector,
        offset: int,
        shanten: int
    ):
        if shanten == 0:
            v.add(offset+cls.TEMPAI)
        elif shanten == 1:
            v.add(offset+cls.ONE)
        elif shanten == 2:
            v.add(offset+cls.TWO)
        elif shanten == 3:
            v.add(offset+cls.THREE)
        elif shanten == 4:
            v.add(offset+cls.FOUR)
        elif shanten >= 5:
            v.add(offset+cls.OVER_FIVE)

class PhaseFeature: # とりあえず自分の河の枚数で評価
    BEGINNING = 0           # 6枚まで
    MIDDLE = BEGINNING + 1  # 12枚まで
    END = MIDDLE + 1        # 13枚以上
    SIZE = END + 1

    @classmethod
    def add(
        cls,
        v: FeatureVector,
        offset: int,
        player: Player
    ):
        num = len(player.river)
        if num <= 6:
            v.add(offset+cls.BEGINNING)
        elif num <= 12:
            v.add(offset+cls.MIDDLE)
        else:
            v.add(offset+cls.END)

class HandFeature:
    MENZEN = 0
    PHASE = MENZEN + 1
    EXPOSED = PHASE + PhaseFeature.SIZE                             # 副露
    CLOSED_TILES = EXPOSED + ExposedFeature.SIZE                    # 手牌の種類と枚数
    CLOSED_CHI = CLOSED_TILES + TileKind.SIZE * 4                   # 手牌にある順子
    CLOSED_OUTSIDE_WAIT = CLOSED_CHI + ChiKind.SIZE                 # 手牌にある塔子
    CLOSED_INSIDE_WAIT = CLOSED_OUTSIDE_WAIT + SerialPairKind.SIZE  # 手牌にある嵌張
    EXIST_MAN = CLOSED_INSIDE_WAIT + ChiKind.SIZE                   # 手牌と副露に萬子があるか
    EXIST_PIN = EXIST_MAN + 1                                       # 手牌と副露に筒子があるか
    EXIST_SOU = EXIST_PIN + 1                                       # 手牌と副露に索子があるか
    EXIST_HONOR = EXIST_SOU + 1                                     # 手牌と副露に字牌があるか
    EXIST_TERMINAL = EXIST_HONOR + 1                                # 手牌と副露に么九牌があるか
    EXIST_SIMPLE = EXIST_TERMINAL + 1                               # 手牌と副露に中張牌があるか
    SIZE = EXIST_SIMPLE + 1

    @classmethod
    def add(
        cls,
        v: FeatureVector,
        offset: int,
        my_player: Player
    ):
        my_hand = my_player.hand
        exist_tiles = [False for _ in range(TileKind.SIZE)]

        menzen = True
        for exposed in my_hand.exposed:
            if exposed.exposed_kind != ExposedKind.KAN_CLOSE:
                menzen = False
        if menzen:
            v.add(offset+cls.MENZEN)
        PhaseFeature.add(v, offset+cls.PHASE, my_player)
        ExposedFeature.add(v, offset+cls.EXPOSED, my_hand)
        tiles = Counter([tile.tile_kind for tile in my_hand.closed])
        for kind, cnt in tiles.items():
            exist_tiles[kind] = True
            for i in range(cnt):
                v.add(offset+cls.CLOSED_TILES+kind*4+i)

        # 順子と嵌張
        for tile_start, tile_end, chi_start in [(TileKind.M2, TileKind.M8, ChiKind.M123), (TileKind.P2, TileKind.P8, ChiKind.P123), (TileKind.S2, TileKind.S8, ChiKind.S123)]:
            for center in range(tile_start, tile_end+1):
                if exist_tiles[center-1] and exist_tiles[center] and exist_tiles[center+1]:
                    v.add(offset+cls.CLOSED_CHI+center-tile_start+chi_start)
                if exist_tiles[center-1] and (not exist_tiles[center]) and exist_tiles[center+1]:
                    v.add(offset+cls.CLOSED_INSIDE_WAIT+center-tile_start+chi_start)

        # 塔子
        for tile_one, pair in [(TileKind.M1, SerialPairKind.M12), (TileKind.P1, SerialPairKind.P12), (TileKind.S1, SerialPairKind.S12)]:
            if exist_tiles[tile_one] and exist_tiles[tile_one+1] and (not exist_tiles[tile_one+2]):
                v.add(offset+cls.CLOSED_OUTSIDE_WAIT+pair)
        for tile_nine, pair in [(TileKind.M9, SerialPairKind.M89), (TileKind.P9, SerialPairKind.P89), (TileKind.S9, SerialPairKind.S89)]:
            if exist_tiles[tile_nine] and exist_tiles[tile_nine-1] and (not exist_tiles[tile_nine-2]):
                v.add(offset+cls.CLOSED_OUTSIDE_WAIT+pair)
        for tile_start, tile_end, pair_start in [(TileKind.M2, TileKind.M7, SerialPairKind.M23), (TileKind.P2, TileKind.P7, SerialPairKind.P23), (TileKind.S2, TileKind.S7, SerialPairKind.S23)]:
            for left in range(tile_start, tile_end+1):
                if (not exist_tiles[left-1]) and exist_tiles[left] and exist_tiles[left+1] and (not exist_tiles[left+2]):
                    v.add(offset+cls.CLOSED_OUTSIDE_WAIT+left-tile_start+pair_start)

        # 牌種
        for tile_start, tile_end, dif in [(TileKind.M1, TileKind.M9, cls.EXIST_MAN), (TileKind.P1, TileKind.P9, cls.EXIST_PIN), (TileKind.S1, TileKind.S9, cls.EXIST_SOU), (TileKind.EAST, TileKind.RED, cls.EXIST_HONOR)]:
            for tile in range(tile_start, tile_end+1):
                if exist_tiles[tile]:
                    v.add(offset+dif)
                    break
        for tile in [TileKind.M1, TileKind.M9, TileKind.P1, TileKind.P9, TileKind.S1, TileKind.S9]:
            if exist_tiles[tile]:
                v.add(offset+cls.EXIST_TERMINAL)
                break
        for tile_start, tile_end in [(TileKind.M2, TileKind.M8), (TileKind.P2, TileKind.P8), (TileKind.S2, TileKind.S8)]:
            for tile in range(tile_start, tile_end+1):
                if exist_tiles[tile]:
                    v.add(offset+cls.EXIST_SIMPLE)
                    break
            else:
                continue
            break

class RiverFeature:
    DISCARDED = 0
    SIZE = DISCARDED + TileKind.SIZE

    @classmethod
    def add(
        cls,
        v: FeatureVector,
        offset: int,
        river: River
    ):
        kinds = set([tile.tile_kind for tile in river.tiles])
        for kind in kinds:
            v.add(offset+cls.DISCARDED+kind)

class MyPlayerFeature:
    WIND = 0
    RIICHI = WIND + Wind.SIZE
    HAND = RIICHI + 1
    RIVER = HAND + HandFeature.SIZE
    SIZE = RIVER + RiverFeature.SIZE

    @classmethod
    def add(
        cls,
        v: FeatureVector,
        offset: int,
        player: Player
    ):
        v.add(offset+cls.WIND+player.wind)
        if player.riichi:
            v.add(offset+cls.RIICHI)
        HandFeature.add(v, cls.HAND, player)
        RiverFeature.add(v, offset+cls.RIVER, player.river)

class OpponentPlayerFeature:
    WIND = 0
    RIICHI = WIND + Wind.SIZE
    EXPOSED = RIICHI + 1
    RIVER = EXPOSED + ExposedFeature.SIZE
    SIZE = RIVER + RiverFeature.SIZE

    @classmethod
    def add(
        cls,
        v: FeatureVector,
        offset: int,
        player: Player
    ):
        v.add(offset+cls.WIND+player.wind)
        if player.riichi:
            v.add(offset+cls.RIICHI)
        ExposedFeature.add(v, offset+cls.EXPOSED, player.hand)
        RiverFeature.add(v, offset+cls.RIVER, player.river)

class BoardFeature:
    WIND = 0
    SHANTEN = WIND + Wind.SIZE
    PHASE = SHANTEN + ShantenFeature.SIZE
    MY_PLAYER = PHASE + PhaseFeature.SIZE
    SHIMO_PLAYER = MY_PLAYER + MyPlayerFeature.SIZE
    TOIMEN_PLAYER = SHIMO_PLAYER + OpponentPlayerFeature.SIZE
    KAMI_PLAYER = TOIMEN_PLAYER + OpponentPlayerFeature.SIZE
    SIZE = KAMI_PLAYER + OpponentPlayerFeature.SIZE

    @classmethod
    def make(
        cls,
        board: Board
    ):
        v = FeatureVector()

        v.add(cls.WIND+board.wind)
        ShantenFeature.add(v, cls.SHANTEN, board.shanten)
        PhaseFeature.add(v, cls.PHASE, board.players[Relation.ME])
        MyPlayerFeature.add(v, cls.MY_PLAYER, board.players[Relation.ME])
        OpponentPlayerFeature.add(v, cls.SHIMO_PLAYER, board.players[Relation.SHIMO])
        OpponentPlayerFeature.add(v, cls.TOIMEN_PLAYER, board.players[Relation.TOIMEN])
        OpponentPlayerFeature.add(v, cls.KAMI_PLAYER, board.players[Relation.KAMI])

        return v

class ActionFeature:
    ACTION_KIND = 0
    DISCARD_TILE = ACTION_KIND + ActionKind.SIZE
    DISCARD_TSUMOGIRI = DISCARD_TILE + TileKind.SIZE
    DISCARD_RED = DISCARD_TSUMOGIRI + 1
    DISCARD_DORA = DISCARD_RED + 1
    DISCARD_EFFECTIVE = DISCARD_DORA + 1
    DISCARD_SAFE_SHIMO = DISCARD_EFFECTIVE + 1
    DISCARD_SAFE_TOIMEN = DISCARD_SAFE_SHIMO + 1
    DISCARD_SAFE_KAMI = DISCARD_SAFE_TOIMEN + 1
    CHI_KIND = DISCARD_SAFE_KAMI + 1
    PON_KIND = CHI_KIND + ChiKind.SIZE * 3
    KAN_OPEN_KIND = PON_KIND + TileKind.SIZE
    KAN_CLOSE_KIND = KAN_OPEN_KIND + TileKind.SIZE
    KAN_ADD_KIND = KAN_CLOSE_KIND + TileKind.SIZE
    TSUMO_TILE = KAN_ADD_KIND + TileKind.SIZE
    RON_TILE = TSUMO_TILE + TileKind.SIZE
    SIZE = RON_TILE + TileKind.SIZE

    @classmethod
    def make(
        cls,
        action: Action,
        board: Board
    ):
        v = FeatureVector()

        v.add(cls.ACTION_KIND+action.action_kind)
        match action.action_kind:
            case ActionKind.DISCARD:
                tile_kind = action.discard_tile.tile_kind
                v.add(cls.DISCARD_TILE+tile_kind)
                if action.discard_tsumogiri:
                    v.add(cls.DISCARD_TSUMOGIRI)
                if action.discard_red:
                    v.add(cls.DISCARD_RED)
                if action.discard_dora:
                    v.add(cls.DISCARD_DORA)
                if action.discard_effective:
                    v.add(cls.DISCARD_EFFECTIVE)
                
                if tile_kind in board.players[1].safe:
                    v.add(cls.DISCARD_SAFE_SHIMO)
                if tile_kind in board.players[2].safe:
                    v.add(cls.DISCARD_SAFE_TOIMEN)
                if tile_kind in board.players[3].safe:
                    v.add(cls.DISCARD_SAFE_KAMI)

            case ActionKind.CHI:    # (chi, steal) = (M123,M1), (M123, M2), ..., (M789, M9), (P123, P1), ..., (S789, S9)
                if ChiKind.M123 <= action.chi_kind <= ChiKind.M789:
                    M_idx = action.chi_kind - ChiKind.M123
                    diff = action.steal_tile.tile_kind - TileKind.M2 - (action.chi_kind - ChiKind.M123) + 1 # stealが順子の左なら0, 中央なら1, 右なら2
                    v.add(cls.CHI_KIND + ChiKind.SIZE*0 + M_idx*3 + diff)
                    # print(f"chi: {action.chi_kind.name}, steal: {action.steal_tile.tile_kind.name}, M_idx: {M_idx}, diff: {diff}, idx: {cls.CHI_KIND + ChiKind.SIZE*0 + M_idx*3 + diff}")
                elif ChiKind.P123 <= action.chi_kind <= ChiKind.P789:
                    P_idx = action.chi_kind - ChiKind.P123
                    diff = action.steal_tile.tile_kind - TileKind.P2 - (action.chi_kind - ChiKind.P123) + 1
                    v.add(cls.CHI_KIND + ChiKind.SIZE*1 + P_idx*3 + diff)
                    # print(f"chi: {action.chi_kind.name}, steal: {action.steal_tile.tile_kind.name}, P_idx: {P_idx}, diff: {diff}, idx: {cls.CHI_KIND + ChiKind.SIZE*1 + P_idx*3 + diff}")
                elif ChiKind.S123 <= action.chi_kind <= ChiKind.S789:
                    S_idx = action.chi_kind - ChiKind.S123
                    diff = action.steal_tile.tile_kind - TileKind.S2 - (action.chi_kind - ChiKind.S123) + 1
                    v.add(cls.CHI_KIND + ChiKind.SIZE*2 + S_idx*3 + diff)
                    # print(f"chi: {action.chi_kind.name}, steal: {action.steal_tile.tile_kind.name}, S_idx: {S_idx}, diff: {diff}, idx: {cls.CHI_KIND + ChiKind.SIZE*2 + S_idx*3 + diff}")
            case ActionKind.PON:
                v.add(cls.PON_KIND+action.pon_kind)
            case ActionKind.KAN_OPEN:
                v.add(cls.KAN_OPEN_KIND+action.kan_kind)
            case ActionKind.KAN_CLOSE:
                v.add(cls.KAN_CLOSE_KIND+action.kan_kind)
            case ActionKind.KAN_ADD:
                v.add(cls.KAN_ADD_KIND+action.kan_kind)
            case ActionKind.TSUMO:
                v.add(cls.TSUMO_TILE+action.tsumo_tile.tile_kind)
            case ActionKind.RON:
                v.add(cls.RON_TILE+action.ron_tile.tile_kind)
            case ActionKind.RIICHI:
                pass
            case ActionKind.DRAW:
                pass
            case ActionKind.NO:
                pass

        return v