import torch

class Model(torch.nn.Module):
    MOMENTUM = 0.1
    YAKU_NUM = 3
    SCORE_NUM = 4
    DISCARD_NUM = 34
    OPTIONAL_NUM = 12

    UNIT_SIZE = 128
    MINI_UNIT_SIZE = 64

    BOARD_INNER_SIZE = UNIT_SIZE * 4
    VALUE_INNER_SIZE = UNIT_SIZE
    YAKU_INNER_SIZE = UNIT_SIZE
    SCORE_INNER_SIZE = UNIT_SIZE
    DISCARD_INNER_SIZE = UNIT_SIZE * 2
    OPTIONAL_INNER_SIZE = UNIT_SIZE * 2

    def __init__(self, board_feature_size, discard_feature_size, optional_feature_size):
        super(Model, self).__init__()
        
        self.board_feature_size = board_feature_size
        self.discard_feature_size = discard_feature_size
        self.optional_feature_size = optional_feature_size

        # discardのレイヤー
        self.discard_board_bag = torch.nn.EmbeddingBag(board_feature_size, self.BOARD_INNER_SIZE, mode="sum")
        self.discard_board_norm_1 = torch.nn.BatchNorm1d(self.BOARD_INNER_SIZE, momentum=self.MOMENTUM)
        self.discard_board_linear_1 = torch.nn.Linear(self.BOARD_INNER_SIZE, self.BOARD_INNER_SIZE, bias=False)
        self.discard_board_norm_2 = torch.nn.BatchNorm1d(self.BOARD_INNER_SIZE, momentum=self.MOMENTUM)
        self.discard_board_linear_2 = torch.nn.Linear(self.BOARD_INNER_SIZE, self.BOARD_INNER_SIZE, bias=False)
        self.discard_board_norm_3 = torch.nn.BatchNorm1d(self.BOARD_INNER_SIZE, momentum=self.MOMENTUM)

        self.discard_vys_linear_1 = torch.nn.Linear(self.VALUE_INNER_SIZE+self.YAKU_INNER_SIZE+self.SCORE_INNER_SIZE, self.VALUE_INNER_SIZE+self.YAKU_INNER_SIZE+self.SCORE_INNER_SIZE, bias=False)
        self.discard_vys_norm_1 = torch.nn.BatchNorm1d(self.VALUE_INNER_SIZE+self.YAKU_INNER_SIZE+self.SCORE_INNER_SIZE, momentum=self.MOMENTUM)
        self.discard_vys_linear_2 = torch.nn.Linear(self.VALUE_INNER_SIZE+self.YAKU_INNER_SIZE+self.SCORE_INNER_SIZE, self.VALUE_INNER_SIZE+self.YAKU_INNER_SIZE+self.SCORE_INNER_SIZE, bias=False)
        self.discard_vys_norm_2 = torch.nn.BatchNorm1d(self.VALUE_INNER_SIZE+self.YAKU_INNER_SIZE+self.SCORE_INNER_SIZE, momentum=self.MOMENTUM)

        self.discard_value_linear = torch.nn.Linear(self.VALUE_INNER_SIZE, self.VALUE_INNER_SIZE, bias=False)
        self.discard_value_norm = torch.nn.BatchNorm1d(self.VALUE_INNER_SIZE, momentum=self.MOMENTUM)
        self.discard_value_out = torch.nn.Linear(self.VALUE_INNER_SIZE, 1, bias=True)

        self.discard_yaku_linear = torch.nn.Linear(self.YAKU_INNER_SIZE, self.YAKU_INNER_SIZE, bias=False)
        self.discard_yaku_norm = torch.nn.BatchNorm1d(self.YAKU_INNER_SIZE, momentum=self.MOMENTUM)
        self.discard_yaku_out = torch.nn.Linear(self.YAKU_INNER_SIZE, self.YAKU_NUM, bias=True)

        self.discard_score_linear = torch.nn.Linear(self.SCORE_INNER_SIZE, self.SCORE_INNER_SIZE, bias=False)
        self.discard_score_norm = torch.nn.BatchNorm1d(self.SCORE_INNER_SIZE, momentum=self.MOMENTUM)
        self.discard_score_out = torch.nn.Linear(self.SCORE_INNER_SIZE, self.SCORE_NUM, bias=True)

        self.discard_action_bag = torch.nn.EmbeddingBag(discard_feature_size, self.MINI_UNIT_SIZE, mode="sum")
        self.discard_action_norm = torch.nn.BatchNorm1d(self.MINI_UNIT_SIZE, momentum=self.MOMENTUM)
        self.discard_linear = torch.nn.Linear(self.UNIT_SIZE+self.MINI_UNIT_SIZE, self.DISCARD_INNER_SIZE, bias=False)
        self.discard_norm = torch.nn.BatchNorm1d(self.DISCARD_INNER_SIZE, momentum=self.MOMENTUM)
        self.discard_out = torch.nn.Linear(self.DISCARD_INNER_SIZE, self.DISCARD_NUM, bias=True)

        # optionalのレイヤー
        self.optional_board_bag = torch.nn.EmbeddingBag(board_feature_size, self.BOARD_INNER_SIZE, mode="sum")
        self.optional_board_norm_1 = torch.nn.BatchNorm1d(self.BOARD_INNER_SIZE, momentum=self.MOMENTUM)
        self.optional_board_linear_1 = torch.nn.Linear(self.BOARD_INNER_SIZE, self.BOARD_INNER_SIZE, bias=False)
        self.optional_board_norm_2 = torch.nn.BatchNorm1d(self.BOARD_INNER_SIZE, momentum=self.MOMENTUM)
        self.optional_board_linear_2 = torch.nn.Linear(self.BOARD_INNER_SIZE, self.BOARD_INNER_SIZE, bias=False)
        self.optional_board_norm_3 = torch.nn.BatchNorm1d(self.BOARD_INNER_SIZE, momentum=self.MOMENTUM)

        self.optional_vys_linear_1 = torch.nn.Linear(self.VALUE_INNER_SIZE+self.YAKU_INNER_SIZE+self.SCORE_INNER_SIZE, self.VALUE_INNER_SIZE+self.YAKU_INNER_SIZE+self.SCORE_INNER_SIZE, bias=False)
        self.optional_vys_norm_1 = torch.nn.BatchNorm1d(self.VALUE_INNER_SIZE+self.YAKU_INNER_SIZE+self.SCORE_INNER_SIZE, momentum=self.MOMENTUM)
        self.optional_vys_linear_2 = torch.nn.Linear(self.VALUE_INNER_SIZE+self.YAKU_INNER_SIZE+self.SCORE_INNER_SIZE, self.VALUE_INNER_SIZE+self.YAKU_INNER_SIZE+self.SCORE_INNER_SIZE, bias=False)
        self.optional_vys_norm_2 = torch.nn.BatchNorm1d(self.VALUE_INNER_SIZE+self.YAKU_INNER_SIZE+self.SCORE_INNER_SIZE, momentum=self.MOMENTUM)
        
        self.optional_value_linear = torch.nn.Linear(self.VALUE_INNER_SIZE, self.VALUE_INNER_SIZE, bias=False)
        self.optional_value_norm = torch.nn.BatchNorm1d(self.VALUE_INNER_SIZE, momentum=self.MOMENTUM)
        self.optional_value_out = torch.nn.Linear(self.VALUE_INNER_SIZE, 1, bias=True)

        self.optional_yaku_linear = torch.nn.Linear(self.YAKU_INNER_SIZE, self.YAKU_INNER_SIZE, bias=False)
        self.optional_yaku_norm = torch.nn.BatchNorm1d(self.YAKU_INNER_SIZE, momentum=self.MOMENTUM)
        self.optional_yaku_out = torch.nn.Linear(self.YAKU_INNER_SIZE, self.YAKU_NUM, bias=True)

        self.optional_score_linear = torch.nn.Linear(self.SCORE_INNER_SIZE, self.SCORE_INNER_SIZE, bias=False)
        self.optional_score_norm = torch.nn.BatchNorm1d(self.SCORE_INNER_SIZE, momentum=self.MOMENTUM)
        self.optional_score_out = torch.nn.Linear(self.SCORE_INNER_SIZE, self.SCORE_NUM, bias=True)
        
        self.optional_action_bag = torch.nn.EmbeddingBag(optional_feature_size, self.MINI_UNIT_SIZE, mode="sum")
        self.optional_action_norm = torch.nn.BatchNorm1d(self.MINI_UNIT_SIZE, momentum=self.MOMENTUM)
        self.optional_linear = torch.nn.Linear(self.UNIT_SIZE+self.MINI_UNIT_SIZE, self.OPTIONAL_INNER_SIZE, bias=False)
        self.optional_norm = torch.nn.BatchNorm1d(self.OPTIONAL_INNER_SIZE, momentum=self.MOMENTUM)
        self.optional_out = torch.nn.Linear(self.OPTIONAL_INNER_SIZE, self.OPTIONAL_NUM, bias=True)

    def _innner_forward_discard(self, board_indexes, board_offsets, action_indexes, action_offsets):
        
        board = self.discard_board_bag(board_indexes, board_offsets)
        board = torch.nn.functional.relu(board)
        board = self.discard_board_norm_1(board)
        board = self.discard_board_linear_1(board)
        board = self.discard_board_norm_2(board)
        board = self.discard_board_linear_2(board)
        board = self.discard_board_norm_3(board)

        b_to_vys, b_to_d_with_a = torch.split(board, [self.VALUE_INNER_SIZE+self.YAKU_INNER_SIZE+self.SCORE_INNER_SIZE, self.UNIT_SIZE], dim=1)

        vys = torch.cat([b_to_vys], dim=1)

        vys = torch.nn.functional.relu(vys)
        vys = self.discard_vys_linear_1(vys)
        vys = torch.nn.functional.relu(vys)
        vys = self.discard_vys_norm_1(vys)
        vys = self.discard_vys_linear_2(vys)
        vys = torch.nn.functional.relu(vys)
        vys = self.discard_vys_norm_2(vys)

        value, yaku, score = torch.split(vys, [self.VALUE_INNER_SIZE, self.YAKU_INNER_SIZE, self.SCORE_INNER_SIZE], dim=1)
        
        action = self.discard_action_bag(action_indexes, action_offsets)
        action = torch.nn.functional.relu(action)
        action = self.discard_action_norm(action)
        discard = torch.cat([b_to_d_with_a, action], dim=1)

        discard = self.discard_linear(discard)
        discard = torch.nn.functional.relu(discard)
        discard = self.discard_norm(discard)
        discard = self.discard_out(discard)

        value = self.discard_value_linear(value)
        value = torch.nn.functional.relu(value)
        value = self.discard_value_norm(value)
        value = self.discard_value_out(value)
        value = torch.tanh(value)

        yaku = self.discard_yaku_linear(yaku)
        yaku = torch.nn.functional.relu(yaku)
        yaku = self.discard_yaku_norm(yaku)
        yaku = self.discard_yaku_out(yaku)

        score = self.discard_score_linear(score)
        score = torch.nn.functional.relu(score)
        score = self.discard_score_norm(score)
        score = self.discard_score_out(score)
        score = torch.tanh(score)

        return value, yaku, discard, score

    def forward_discard(self, board_indexes, board_offsets, action_indexes, action_offsets, valid_masks):
        # 無効な手を小さい値でマスクする
        value, yaku, discard, score = self._innner_forward_discard(board_indexes, board_offsets, action_indexes, action_offsets)
        discard[~valid_masks] = -1e18

        return value, yaku, discard, score

    def _inner_forward_optional(self, board_indexes, board_offsets, action_indexes, action_offsets):

        board = self.optional_board_bag(board_indexes, board_offsets)
        board = torch.nn.functional.relu(board)
        board = self.optional_board_norm_1(board)
        board = self.optional_board_linear_1(board)
        board = self.optional_board_norm_2(board)
        board = self.optional_board_linear_2(board)
        board = self.optional_board_norm_3(board)

        b_to_vys, b_to_o_with_a = torch.split(board, [self.VALUE_INNER_SIZE+self.YAKU_INNER_SIZE+self.SCORE_INNER_SIZE, self.UNIT_SIZE], dim=1)

        vys = torch.cat([b_to_vys], dim=1)

        vys = torch.nn.functional.relu(vys)
        vys = self.optional_vys_linear_1(vys)
        vys = torch.nn.functional.relu(vys)
        vys = self.optional_vys_norm_1(vys)
        vys = self.optional_vys_linear_2(vys)
        vys = torch.nn.functional.relu(vys)
        vys = self.optional_vys_norm_2(vys)

        value, yaku, score = torch.split(vys, [self.VALUE_INNER_SIZE, self.YAKU_INNER_SIZE, self.SCORE_INNER_SIZE], dim=1)

        action = self.optional_action_bag(action_indexes, action_offsets)
        action = torch.nn.functional.relu(action)
        action = self.optional_action_norm(action)
        optional = torch.cat([b_to_o_with_a, action], dim=1)

        optional = self.optional_linear(optional)
        optional = torch.nn.functional.relu(optional)
        optional = self.optional_norm(optional)
        optional = self.optional_out(optional)

        value = self.optional_value_linear(value)
        value = torch.nn.functional.relu(value)
        value = self.optional_value_norm(value)
        value = self.optional_value_out(value)
        value = torch.tanh(value)

        yaku = self.optional_yaku_linear(yaku)
        yaku = torch.nn.functional.relu(yaku)
        yaku = self.optional_yaku_norm(yaku)
        yaku = self.optional_yaku_out(yaku)

        score = self.optional_score_linear(score)
        score = torch.nn.functional.relu(score)
        score = self.optional_score_norm(score)
        score = self.optional_score_out(score)
        score = torch.tanh(score)

        return value, yaku, optional, score

    def forward_optional(self, board_indexes, board_offsets, action_indexes, action_offsets, valid_masks):
        # 無効な手を小さい値でマスクする
        value, yaku, optional, score = self._inner_forward_optional(board_indexes, board_offsets, action_indexes, action_offsets)
        optional[~valid_masks] = -1e18

        return value, yaku, optional, score

    def forward(self, board_indexes, board_offsets, action_indexes, action_offsets):
        raise NotImplementedError()
