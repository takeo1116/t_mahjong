from enum import IntEnum, IntFlag, unique

# from mjx.action import Action as MjxAction
# from mjx.observation import Observation as MjxObservation
# from mjx.tile import Tile as MjxTile
# from mjx.const import TileType as MjxTileType, RelativePlayerIdx as MjxRelativePlayerIdx, EventType as MjxEventType
# from mjx.hand import Hand as MjxHand
# from mjx.open import Open as MjxOpen

@unique
class Wind(IntEnum):
    EAST = 0
    SOUTH = 1
    WEST = 2
    NORTH = 3
    SIZE = 4

    def shimo(self):
        return Wind((self + 1) % 4)

    def toimen(self):
        return Wind((self + 2) % 4)
    
    def kami(self):
        return Wind((self + 3) % 4)

@unique
class Relation(IntEnum):
    ME = 0
    SHIMO = 1
    TOIMEN = 2
    KAMI = 3
    SIZE = 4

    @classmethod
    def from_mjx(
        cls,
        relative_player_idx # MjxRelativePlayerIdx
    ):
        return cls(relative_player_idx)

@unique
class ExposedKind(IntEnum):
    CHI = 0
    PON = 1
    KAN_OPEN = 2    # 大明槓、加槓
    KAN_CLOSE = 3   # 暗槓
    SIZE = 4

@unique
class TileKind(IntEnum):
    M1 = 0          # 萬子
    M2 = 1
    M3 = 2
    M4 = 3
    M5 = 4
    M6 = 5
    M7 = 6
    M8 = 7
    M9 = 8
    P1 = 9          # 筒子
    P2 = 10
    P3 = 11
    P4 = 12
    P5 = 13
    P6 = 14
    P7 = 15
    P8 = 16
    P9 = 17
    S1 = 18         # 索子
    S2 = 19
    S3 = 20
    S4 = 21
    S5 = 22
    S6 = 23
    S7 = 24
    S8 = 25
    S9 = 26
    EAST = 27       # 東
    SOUTH = 28      # 南
    WEST = 29       # 西
    NORTH = 30      # 北
    WHITE = 31      # 白
    GREEN = 32      # 發
    RED = 33        # 中
    UNKNOWN = 34    # 裏、未ツモなど
    SIZE = 35

    @classmethod
    def from_mjx(
        cls,
        tile_type # MjxTileType
    ):
        return cls(tile_type)

@unique
class ChiKind(IntEnum):
    M123 = 0
    M234 = 1
    M345 = 2
    M456 = 3
    M567 = 4
    M678 = 5
    M789 = 6
    P123 = 7
    P234 = 8
    P345 = 9
    P456 = 10
    P567 = 11
    P678 = 12
    P789 = 13
    S123 = 14
    S234 = 15
    S345 = 16
    S456 = 17
    S567 = 18
    S678 = 19
    S789 = 20
    SIZE = 21

    @classmethod
    def from_tilekinds(
        cls,
        left: TileKind,
        center: TileKind,
        right: TileKind
    ):
        if (not (left == center-1 and center == right-1)):
            raise Exception(f"invalid chi: {left, center, right}")
        
        if TileKind.M2 <= center <= TileKind.M8:
            return cls(center - TileKind.M2 + ChiKind.M123)
        elif TileKind.P2 <= center <= TileKind.P8:
            return cls(center - TileKind.P2 + ChiKind.P123)
        elif TileKind.S2 <= center <= TileKind.S8:
            return cls(center - TileKind.S2 + ChiKind.S123)

        raise Exception(f"invalid chi: {left, center, right}")

@unique
class SerialPairKind(IntEnum):
    M12 = 0
    M23 = 1
    M34 = 2
    M45 = 3
    M56 = 4
    M67 = 5
    M78 = 6
    M89 = 7
    P12 = 8
    P23 = 9
    P34 = 10
    P45 = 11
    P56 = 12
    P67 = 13
    P78 = 14
    P89 = 15
    S12 = 16
    S23 = 17
    S34 = 18
    S45 = 19
    S56 = 20
    S67 = 21
    S78 = 22
    S89 = 23
    SIZE = 24

@unique
class ActionKind(IntEnum):
    DISCARD = 0     # 打牌
    CHI = 1         # チー
    PON = 2         # ポン
    KAN_OPEN = 3    # 大明槓
    KAN_CLOSE = 4   # 暗槓
    KAN_ADD = 5     # 加槓
    TSUMO = 6       # ツモ
    RON = 7         # ロン
    RIICHI = 8      # 立直
    DRAW = 9        # 流局
    NO = 10         # キャンセル（鳴かない、上がらない）
    SIZE = 11

class Tile():
    def __init__(
        self,
        tile_kind: TileKind,
        red: bool
    ):
        self.tile_kind = tile_kind
        self.red = red

    @classmethod
    def from_mjx(
        cls,
        tile # MjxTile
    ):
        tile_kind = TileKind.from_mjx(tile.type())
        red = tile.is_red()
        return Tile(tile_kind, red)

    def __repr__(self):
        if self.tile_kind == TileKind.UNKNOWN:
            return "X"

        name = self.tile_kind.name
        if self.red:
            name += "R"
        return name

class Exposed:  #   ポン、チー、カン
    def __init__(
        self,
        relation: Relation,
        exposed_kind: ExposedKind,
        chi_kind: ChiKind = None,
        pon_kind: TileKind = None,
        kan_kind: TileKind = None
    ):
        self.relation = relation
        self.exposed_kind = exposed_kind
        self.chi_kind = chi_kind
        self.pon_kind = pon_kind
        self.kan_kind = kan_kind
    
    @classmethod
    def from_tiles(
        cls,
        relation: Relation,
        tiles: list[Tile],
        stolen: Tile
    ):
        kind_list = [tile.tile_kind for tile in tiles]
        kind_set = set(kind_list)

        tile_num = len(kind_list)
        kind_num = len(kind_set)

        if tile_num == 4 and kind_num == 1:     # カン
            if relation == Relation.ME:             # 暗槓
                return Exposed(relation, ExposedKind.KAN_CLOSE, kan_kind=stolen.tile_kind)
            elif relation != Relation.ME:           # 大明槓、加槓
                return Exposed(relation, ExposedKind.KAN_OPEN, kan_kind=stolen.tile_kind) 
        elif tile_num == 3 and kind_num == 1:   # ポン
            return Exposed(relation, ExposedKind.PON, pon_kind=stolen.tile_kind)
        elif tile_num == 3 and kind_num == 3:   # チー
            return Exposed(relation, ExposedKind.CHI, chi_kind=ChiKind.from_tilekinds(kind_list[0], kind_list[1], kind_list[2]))

        raise Exception(f"relation: {relation.name}, tiles: {tiles}, stolen: {stolen}")

    def __repr__(self):
        if self.exposed_kind == ExposedKind.PON:
            return f"PON({self.pon_kind.name}) from {self.relation.name}"
        elif self.exposed_kind == ExposedKind.CHI:
            return f"CHI({self.chi_kind.name}) from {self.relation.name}"
        elif self.exposed_kind == ExposedKind.KAN_CLOSE:
            return f"KAN_CLOSE({self.kan_kind.name})"
        elif self.exposed_kind == ExposedKind.KAN_OPEN:
            return f"KAN_OPEN({self.kan_kind.name})"
        return "unknown"

    @classmethod
    def from_mjx(
        cls,
        open # MjxOpen
    ):
        relation = Relation.from_mjx(open.steal_from())
        tiles = [Tile.from_mjx(tile) for tile in open.tiles()]
        stolen = Tile.from_mjx(open.stolen_tile())
        return cls.from_tiles(relation, tiles, stolen)
        

class Hand:
    def __init__(
        self,
        closed: list[Tile],
        exposed: list[Exposed]
    ):
        self.closed = closed
        self.exposed = exposed

    @classmethod
    def from_exposes(   # 手牌を全部UNKNOWNで埋める（相手の手牌用）
        cls,
        exposed: list[Exposed],
        num: int = 13
    ):
        closed = [Tile(TileKind.UNKNOWN, red=False) for _ in range(num - len(exposed) * 3)]
        return cls(closed, exposed)
    
    @classmethod
    def from_mjx(
        cls,
        hand # MjxHand
    ):
        closed = [Tile.from_mjx(tile) for tile in hand.closed_tiles()]
        exposed = [Exposed.from_mjx(open) for open in hand.opens()]
        
        return cls(closed, exposed)

    def __repr__(self):
        return f"{self.closed}, {self.exposed}"

class River:
    def __init__(
        self,
        tiles: list[Tile],
    ):
        self.tiles = tiles

    def __len__(self):
        return len(self.tiles)
    
    def __repr__(self):
        return f"{self.tiles}"
    
class Player:
    def __init__(
        self,
        relation: Relation,
        wind: Wind,
        score: int,
        riichi: bool,
        hand: Hand,
        river: River,
        safe: list[TileKind]
    ):
        self.relation = relation
        self.wind = wind
        self.score = score
        self.riichi = riichi
        self.hand = hand
        self.river = river
        self.safe = safe

    def __repr__(self):
        string = self.relation.name + "(" + self.wind.name + ")"
        if self.riichi:
            string += "!"
        string += f"\nscore: {self.score}"
        string += f"\nhand: {self.hand}"
        string += f"\nriver: {self.river}"
        return string

class Board:
    def __init__(
        self,
        wind: Wind,
        shanten: int,
        players: list[Player],
        effective_discard: list[TileKind]
    ):
        self.wind = wind
        self.shanten = shanten
        self.players = players
        self.effective_discard = effective_discard

    @classmethod
    def from_mjx(
        cls,
        observation # MjxObservation
    ):
        field_wind = Wind(observation.round() % 4)

        my_wind = Wind(observation.who())

        mjx_hand = observation.curr_hand()
        my_hand = Hand.from_mjx(mjx_hand)
        shanten = mjx_hand.shanten_number()
        effective_discard = [TileKind.from_mjx(entry) for entry in mjx_hand.effective_discard_types()]

        riichi = [False, False, False, False]
        exposes = [[], [], [], []]

        discarded = [[] for _ in range(4)]  # そのプレイヤーが捨てた牌
        safe = [set() for _ in range(4)]       # そのプレイヤーが捨てた牌＋同順内フリテン＋そのプレイヤーのリーチ後に誰かが捨てた牌

        events = observation.events()
        for event in events:
            event_type = event.type()
            relation = Relation((event.who() - observation.who() + 4) % 4)

            match event_type:
                case 0 | 1: # MjxEventType.DISCARD | MjxEventType.TSUMOGIRI:
                    tile = event.tile()
                    discarded[relation].append(Tile.from_mjx(tile))
                    safe[relation].add(TileKind.from_mjx(tile.type()))
                    for r in range(4):
                        if riichi[r]:
                            safe[r].add(TileKind.from_mjx(tile.type()))
                case 2: # MjxEventType.RIICHI:
                    riichi[relation] = True
                case 7 | 8 | 3 | 9 | 4: # MjxEventType.CHI | MjxEventType.PON | MjxEventType.CLOSED_KAN | MjxEventType.OPEN_KAN | MjxEventType.ADDED_KAN:
                    exposes[relation].append(Exposed.from_mjx(event.open()))

        drawed_rev = [False, False, False, False] # 同順内フリテンを判定する用
        for event in reversed(events):  # eventの逆順
            event_type = event.type()
            relation = Relation((event.who() - observation.who() + 4) % 4)

            match event_type:
                case 0 | 1: # MjxEventType.DISCARD | MjxEventType.TSUMOGIRI:
                    tile = event.tile()
                    for r in range(4):
                        if not drawed_rev[r]:
                            safe[r].add(TileKind.from_mjx(tile.type()))
                    drawed_rev[relation] = True
            
            if drawed_rev[0] & drawed_rev[1] & drawed_rev[2] & drawed_rev[3]:
                break

        safe = [list(entry) for entry in safe]
        rivers = [River(discarded[idx]) for idx in range(4)]
        hands = [Hand.from_exposes(exposes[idx]) for idx in range(4)]
        scores = observation.tens()

        players = {
            Relation.ME: Player(relation=Relation.ME, wind=my_wind, score=scores[my_wind], riichi=riichi[Relation.ME], hand=my_hand, river=rivers[Relation.ME], safe=safe[Relation.ME]),
            Relation.SHIMO: Player(relation=Relation.SHIMO, wind=my_wind.shimo(), score=scores[my_wind.shimo()], riichi=riichi[Relation.SHIMO], hand=hands[Relation.SHIMO], river=rivers[Relation.SHIMO], safe=safe[Relation.SHIMO]),
            Relation.TOIMEN: Player(relation=Relation.TOIMEN, wind=my_wind.toimen(), score=scores[my_wind.toimen()], riichi=riichi[Relation.TOIMEN], hand=hands[Relation.TOIMEN], river=rivers[Relation.TOIMEN], safe=safe[Relation.TOIMEN]),
            Relation.KAMI: Player(relation=Relation.KAMI, wind=my_wind.kami(), score=scores[my_wind.kami()], riichi=riichi[Relation.KAMI], hand=hands[Relation.KAMI], river=rivers[Relation.KAMI], safe=safe[Relation.KAMI]),
        }
        return cls(field_wind, shanten, players, effective_discard)

class Action:
    def __init__(self):
        self.action_kind = None
        
        self.discard_tsumogiri = False
        self.discard_red = False
        self.discard_dora = False
        self.discard_effective = False
        self.discard_tile = None

        self.chi_kind = None
        self.pon_kind = None
        self.kan_kind = None
        self.steal_tile = None
        self.expose_tiles = []

        self.tsumo_tile = None
        self.ron_tile = None

    def __repr__(self):
        if self.action_kind == None:
            return "None"
        elif self.action_kind == ActionKind.DISCARD:
            return f"DISCARD_{self.discard_tile}"
        elif self.action_kind == ActionKind.CHI:
            return f"CHI_{self.chi_kind.name}({self.steal_tile})"
        elif self.action_kind == ActionKind.PON:
            return f"PON_{self.pon_kind.name}({self.steal_tile})"
        elif self.action_kind == ActionKind.KAN_OPEN:
            return f"KAN_OPEN_{self.kan_kind.name}({self.steal_tile})"
        elif self.action_kind == ActionKind.KAN_CLOSE:
            return f"KAN_CLOSE_{self.kan_kind.name}"
        elif self.action_kind == ActionKind.KAN_ADD:
            return f"KAN_ADD_{self.kan_kind.name}"
        elif self.action_kind == ActionKind.TSUMO:
            return f"TSUMO_{self.tsumo_tile}"
        elif self.action_kind == ActionKind.RON:
            return f"RON_{self.ron_tile}"
        elif self.action_kind == ActionKind.RIICHI:
            return "RIICHI"
        elif self.action_kind == ActionKind.DRAW:
            return "DRAW"
        elif self.action_kind == ActionKind.NO:
            return "NO"
        elif self.action_kind == ActionKind.SIZE:
            return "SIZE(error)"
        else:
            return "unknown action"
    
    @classmethod
    def from_mjx(
        cls,
        action, # MjxAction
        board: Board
    ):
        action_id = action.to_idx()
        instance = cls()

        if action_id <= 33:    # 打牌
            instance.action_kind = ActionKind.DISCARD
            instance.discard_tile = Tile.from_mjx(action.tile())
        elif action_id <= 36:  # 打赤5
            instance.action_kind = ActionKind.DISCARD
            if action_id == 34:
                instance.discard_tile = Tile.from_mjx(action.tile())
            elif action_id == 35:
                instance.discard_tile = Tile.from_mjx(action.tile())
            elif action_id == 36:
                instance.discard_tile = Tile.from_mjx(action.tile())
        elif action_id <= 70:  # ツモ切り
            instance.action_kind = ActionKind.DISCARD
            instance.discard_tile = Tile.from_mjx(action.tile())
            instance.discard_tsumogiri = True
        elif action_id <= 73:  # ツモ切り赤5
            instance.action_kind = ActionKind.DISCARD
            if action_id == 71:
                instance.discard_tile = Tile.from_mjx(action.tile())
            elif action_id == 72:
                instance.discard_tile = Tile.from_mjx(action.tile())
            elif action_id == 73:
                instance.discard_tile = Tile.from_mjx(action.tile())
            instance.discard_tsumogiri = True
            instance.discard_red = True
        elif action_id <= 94:  # チー
            instance.action_kind = ActionKind.CHI
            chi = action.open()
            instance.steal_tile = Tile.from_mjx(chi.stolen_tile())
            instance.expose_tiles = [Tile.from_mjx(tile) for tile in chi.tiles_from_hand()]
            instance.chi_kind = ChiKind(action_id - 74)
        elif action_id <= 103: # 赤入りのチー
            instance.action_kind = ActionKind.CHI
            chi = action.open()
            instance.steal_tile = Tile.from_mjx(chi.stolen_tile())
            instance.expose_tiles = [Tile.from_mjx(tile) for tile in chi.tiles_from_hand()]
            if action_id == 95:
                instance.chi_kind = ChiKind.M345
            elif action_id == 96:
                instance.chi_kind = ChiKind.M456
            elif action_id == 97:
                instance.chi_kind = ChiKind.M567
            elif action_id == 98:
                instance.chi_kind = ChiKind.P345
            elif action_id == 99:
                instance.chi_kind = ChiKind.P456
            elif action_id == 100:
                instance.chi_kind = ChiKind.P567
            elif action_id == 101:
                instance.chi_kind = ChiKind.S345
            elif action_id == 102:
                instance.chi_kind = ChiKind.S456
            elif action_id == 103:
                instance.chi_kind = ChiKind.S567
        elif action_id <= 140: # ポン, 赤入りのポン
            instance.action_kind = ActionKind.PON
            pon = action.open()
            steal = Tile.from_mjx(pon.stolen_tile())
            instance.steal_tile = steal
            instance.expose_tiles = [Tile.from_mjx(tile) for tile in pon.tiles_from_hand()]
            instance.pon_kind = steal.tile_kind
        elif action_id <= 174: # カン
            kan = action.open()
            steal = Tile.from_mjx(kan.stolen_tile())
            expose = [Tile.from_mjx(tile) for tile in kan.tiles_from_hand()]
            instance.steal_tile = steal
            instance.expose_tile = expose
            instance.kan_kind = steal.tile_kind
            
            exp_num = len(expose)
            if exp_num == 3:
                instance.action_kind = ActionKind.KAN_OPEN
            elif exp_num == 4:
                instance.action_kind = ActionKind.KAN_CLOSE
            elif exp_num == 1:
                instance.action_kind = ActionKind.KAN_ADD
        elif action_id == 175:  # ツモ
            instance.action_kind = ActionKind.TSUMO
            instance.tsumo_tile = Tile.from_mjx(action.tile())
        elif action_id == 176:  # ロン
            instance.action_kind = ActionKind.RON
            instance.ron_tile = Tile.from_mjx(action.tile())
        elif action_id == 177:  # 立直
            instance.action_kind = ActionKind.RIICHI
        elif action_id == 178:  # 九種九牌
            instance.action_kind = ActionKind.DRAW
        elif action_id == 179:  # キャンセル
            instance.action_kind = ActionKind.NO
        elif action_id == 180:  # ダミー
            raise Exception(f"unexpected dummy action")
        else:
            raise Exception(f"unknown action_id {action_id}")

        if instance.action_kind == ActionKind.DISCARD and instance.discard_tile.tile_kind in board.effective_discard:
            instance.discard_effective = True

        return instance