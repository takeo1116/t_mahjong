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
    # EFFECTIVE_TILES = CLOSED_TILES + TileKind.SIZE * 4            # シャンテン数が進む牌
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
        board: Board
    ):
        my_player = board.players[Relation.ME]
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
        
        # for effective_idx in board.effective_discard:
        #     v.add(offset+cls.EFFECTIVE_TILES+effective_idx)
        # print(my_hand, board.effective_discard)
        # closed = [tile.tile_kind for tile in my_hand.closed]
        # for k in board.effective_discard:
        #     if k not in closed:
        #         print(my_hand, board.effective_discard)

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

class ScoreDiffFeature:
    PLUS_OVER_48000 = 0
    PLUS_24000_47999 = 1
    PLUS_12000_23999 = 2
    PLUS_8000_11999 = 3
    PLUS_4000_7999 = 4
    PLUS_2000_3999 = 5
    PLUS_1_1999 = 6
    SAME = 7
    MINUS_1_1999 = 8
    MINUS_2000_3999 = 9
    MINUS_4000_7999 = 10
    MINUS_8000_11999 = 11
    MINUS_12000_23999 = 12
    MINUS_24000_47999 = 13
    MINUS_OVER_48000 = 14
    SIZE = 15

    @classmethod
    def add(
        cls,
        v: FeatureVector,
        offset: int,
        score_diff: int
    ):
        if 48000 <= score_diff:
            v.add(offset+cls.PLUS_OVER_48000)
        elif 24000 <= score_diff:
            v.add(offset+cls.PLUS_24000_47999)
        elif 12000 <= score_diff:
            v.add(offset+cls.PLUS_12000_23999)
        elif 8000 <= score_diff:
            v.add(offset+cls.PLUS_8000_11999)
        elif 4000 <= score_diff:
            v.add(offset+cls.PLUS_4000_7999)
        elif 2000 <= score_diff:
            v.add(offset+cls.PLUS_2000_3999)
        elif 1 <= score_diff:
            v.add(offset+cls.PLUS_1_1999)
        elif score_diff == 0:
            v.add(offset+cls.SAME)
        elif score_diff <= -1:
            v.add(offset+cls.MINUS_1_1999)
        elif score_diff <= -2000:
            v.add(offset+cls.MINUS_2000_3999)
        elif score_diff <= -4000:
            v.add(offset+cls.MINUS_4000_7999)
        elif score_diff <= -8000:
            v.add(offset+cls.MINUS_8000_11999)
        elif score_diff <= -12000:
            v.add(offset+cls.MINUS_12000_23999)
        elif score_diff <= -24000:
            v.add(offset+cls.MINUS_24000_47999)
        else:
            v.add(offset+cls.MINUS_OVER_48000)
        
class ScoreFeature:
    SCORE_OVER_69000 = 0
    SCORE_53000_68999 = 1
    SCORE_45000_52999 = 2
    SCORE_37000_44999 = 3
    SCORE_33000_36999 = 4
    SCORE_29000_32999 = 5
    SCORE_21000_28999 = 6
    SCORE_17000_20999 = 7
    SCORE_13000_16999 = 8
    SCORE_5000_12999 = 9
    SCORE_UNDER_4999 = 10
    SIZE = 11

    @classmethod
    def add(
        cls,
        v: FeatureVector,
        offset: int,
        score: int
    ):
        if 69000 <= score:
            v.add(offset+cls.SCORE_OVER_69000)
        elif 53000 <= score:
            v.add(offset+cls.SCORE_53000_68999)
        elif 45000 <= score:
            v.add(offset+cls.SCORE_45000_52999)
        elif 37000 <= score:
            v.add(offset+cls.SCORE_37000_44999)
        elif 33000 <= score:
            v.add(offset+cls.SCORE_33000_36999)
        elif 29000 <= score:
            v.add(offset+cls.SCORE_29000_32999)
        elif 21000 <= score:
            v.add(offset+cls.SCORE_21000_28999)
        elif 17000 <= score:
            v.add(offset+cls.SCORE_17000_20999)
        elif 13000 <= score:
            v.add(offset+cls.SCORE_13000_16999)
        elif 5000 <= score:
            v.add(offset+cls.SCORE_5000_12999)
        elif 0 <= score:
            v.add(offset+cls.SCORE_UNDER_4999)
        else:
            Exception("unexpected score")

class MyPlayerFeature:
    SCORE = 0
    WIND = SCORE + ScoreFeature.SIZE
    RIICHI = WIND + Wind.SIZE
    HAND = RIICHI + 1
    RIVER = HAND + HandFeature.SIZE
    SIZE = RIVER + RiverFeature.SIZE

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
        HandFeature.add(v, cls.HAND, board)
        RiverFeature.add(v, offset+cls.RIVER, player.river)

class OpponentPlayerFeature:
    SCORE = 0
    SCORE_DIFF = SCORE + ScoreFeature.SIZE
    WIND = SCORE_DIFF + ScoreDiffFeature.SIZE
    RIICHI = WIND + Wind.SIZE
    EXPOSED = RIICHI + 1
    RIVER = EXPOSED + ExposedFeature.SIZE
    SAFE = RIVER + RiverFeature.SIZE
    SIZE = SAFE + TileKind.SIZE

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
        ExposedFeature.add(v, offset+cls.EXPOSED, player.hand)
        RiverFeature.add(v, offset+cls.RIVER, player.river)
        for safe in player.safe:
            v.add(offset+cls.SAFE+safe)

class BoardFeature:
    KYOKU = 0
    WIND = KYOKU + 8
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

        kyoku = min(board.kyoku, 7)
        v.add(cls.KYOKU+kyoku)
        v.add(cls.WIND+board.wind)
        ShantenFeature.add(v, cls.SHANTEN, board.shanten)
        PhaseFeature.add(v, cls.PHASE, board.players[Relation.ME])
        MyPlayerFeature.add(v, cls.MY_PLAYER, board)
        OpponentPlayerFeature.add(v, cls.SHIMO_PLAYER, Relation.SHIMO, board)
        OpponentPlayerFeature.add(v, cls.TOIMEN_PLAYER, Relation.TOIMEN, board)
        OpponentPlayerFeature.add(v, cls.KAMI_PLAYER, Relation.KAMI, board)

        return v

class DiscardActionFeature:
    EFFECTIVE_TILES = 0
    DORA = EFFECTIVE_TILES + TileKind.SIZE
    HAVE_RED = DORA + TileKind.SIZE
    SIZE = HAVE_RED + 3

    @classmethod
    def make(
        cls,
        board: Board
    ):
        v = FeatureVector()

        for effective_idx in board.effective_discard:
            v.add(cls.EFFECTIVE_TILES+effective_idx)
        for dora in board.doras:
            v.add(cls.DORA+dora)
        for tile in board.players[Relation.ME].hand.closed:
            if tile.red:
                if tile.tile_kind == TileKind.M5:
                    v.add(cls.HAVE_RED+0)
                elif tile.tile_kind == TileKind.P5:
                    v.add(cls.HAVE_RED+1)
                elif tile.tile_kind == TileKind.S5:
                    v.add(cls.HAVE_RED+2)

        return v

class OptionalActionFeature:
    ACTION_KIND = 0
    MENZEN = ACTION_KIND + ActionKind.SIZE
    CHI_KIND = MENZEN + 1
    PON_KIND = CHI_KIND + ChiKind.SIZE * 3
    KAN_OPEN_KIND = PON_KIND + TileKind.SIZE
    KAN_CLOSE_KIND = KAN_OPEN_KIND + TileKind.SIZE
    KAN_ADD_KIND = KAN_CLOSE_KIND + TileKind.SIZE
    TSUMO_TILE = KAN_ADD_KIND + TileKind.SIZE
    RON_TILE = TSUMO_TILE + TileKind.SIZE

    # 他のアクションの情報（現状はNOに対して）
    OTHER_ACTION_KINDS = RON_TILE + TileKind.SIZE
    OTHER_CHI_KINDS = OTHER_ACTION_KINDS + ActionKind.SIZE      # 役がなくなる可能性がある鳴きに関する特徴
    OTHER_PON_KAN_KINDS = OTHER_CHI_KINDS + ChiKind.SIZE        # 役がなくなる可能性がある鳴きに関する特徴

    SIZE = OTHER_PON_KAN_KINDS + TileKind.SIZE

    @classmethod
    def chi_index(
        cls,
        chi: ChiKind,
        steal: TileKind,
    ):
        # (chi, steal) = (M123,M1), (M123, M2), ..., (M789, M9), (P123, P1), ..., (S789, S9)
        if ChiKind.M123 <= chi <= ChiKind.M789:
            M_idx = chi - ChiKind.M123
            diff = steal - TileKind.M2 - (chi - ChiKind.M123) + 1 # stealが順子の左なら0, 中央なら1, 右なら2
            return ChiKind.SIZE*0 + M_idx*3 + diff
        elif ChiKind.P123 <= chi <= ChiKind.P789:
            P_idx = chi - ChiKind.P123
            diff = steal - TileKind.P2 - (chi - ChiKind.P123) + 1
            return ChiKind.SIZE*1 + P_idx*3 + diff
        elif ChiKind.S123 <= chi <= ChiKind.S789:
            S_idx = chi - ChiKind.S123
            diff = steal - TileKind.S2 - (chi - ChiKind.S123) + 1
            return ChiKind.SIZE*2 + S_idx*3 + diff
        else:
            raise Exception(f"unexpected chi kind: {chi}")

    @classmethod
    def make(
        cls,
        action: Action,
        board: Board,
        all_actions: list[Action]
    ):
        v = FeatureVector()

        v.add(cls.ACTION_KIND+action.action_kind)

        my_hand = board.players[Relation.ME].hand
        menzen = True
        for exposed in my_hand.exposed:
            if exposed.exposed_kind != ExposedKind.KAN_CLOSE:
                menzen = False
        if menzen:
            v.add(cls.MENZEN)

        match action.action_kind:
            case ActionKind.DISCARD:
                Exception("Unexpected ActionKind(DISCARD)")

            case ActionKind.CHI:
                chi_index = cls.chi_index(action.chi_kind, action.steal_tile.tile_kind)
                v.add(cls.CHI_KIND+chi_index)
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
                # 他のアクションの情報を入れる
                for other in all_actions:
                    if other.action_kind == ActionKind.NO:
                        continue
                    v.add(cls.OTHER_ACTION_KINDS+other.action_kind)
                    match other.action_kind:
                        case ActionKind.CHI:
                            v.add(cls.OTHER_CHI_KINDS+other.chi_kind)
                        case ActionKind.PON:
                            v.add(cls.OTHER_PON_KAN_KINDS+other.pon_kind)
                        case ActionKind.KAN_OPEN | ActionKind.KAN_ADD:
                            v.add(cls.OTHER_PON_KAN_KINDS+other.kan_kind)
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