import torch

class SimpleCnnModel(torch.nn.Module):
    MOMENTUM = 0.001    # changed
    YAKU_NUM = 3
    SCORE_NUM = 4

    UNIT_SIZE = 128
    BOARD_CHANNEL_SIZE = UNIT_SIZE // 2

    BOARD_INNER_SIZE = UNIT_SIZE * 4
    ACTION_INNER_SIZE = UNIT_SIZE * 2

    VALUE_INNER_SIZE = UNIT_SIZE
    YAKU_INNER_SIZE = UNIT_SIZE
    SCORE_INNER_SIZE = UNIT_SIZE
    POLICY_INNER_SIZE = UNIT_SIZE * 2

    def __init__(self, board_feature_size, board_input_channels, action_feature_size):
        super(SimpleCnnModel, self).__init__()

        self.board_feature_size = board_feature_size
        self.board_cnn_input_channels = board_input_channels
        self.action_feature_size = action_feature_size

        self.board_cnn_conv_1 = torch.nn.Conv2d(board_input_channels, self.BOARD_CHANNEL_SIZE//4, kernel_size=(3, 4), padding=(1, 0))
        self.board_cnn_norm2d_1 = torch.nn.BatchNorm2d(self.BOARD_CHANNEL_SIZE//4)
        self.board_cnn_conv_2 = torch.nn.Conv2d(self.BOARD_CHANNEL_SIZE//4, self.BOARD_CHANNEL_SIZE//2, kernel_size=(3, 1), padding=(1, 0))
        self.board_cnn_norm2d_2 = torch.nn.BatchNorm2d(self.BOARD_CHANNEL_SIZE//2)
        self.board_cnn_conv_3 = torch.nn.Conv2d(self.BOARD_CHANNEL_SIZE//2, self.BOARD_CHANNEL_SIZE//2, kernel_size=(3, 1), padding=(1, 0))
        self.board_cnn_norm2d_3 = torch.nn.BatchNorm2d(self.BOARD_CHANNEL_SIZE//2)
        self.board_cnn_conv_4 = torch.nn.Conv2d(self.BOARD_CHANNEL_SIZE//2, self.BOARD_CHANNEL_SIZE, kernel_size=(3, 1), padding=(1, 0))
        self.board_cnn_norm2d_4 = torch.nn.BatchNorm2d(self.BOARD_CHANNEL_SIZE)
        self.board_cnn_conv_5 = torch.nn.Conv2d(self.BOARD_CHANNEL_SIZE, self.BOARD_CHANNEL_SIZE, kernel_size=(3, 1), padding=(1, 0))
        self.board_cnn_norm2d_5 = torch.nn.BatchNorm2d(self.BOARD_CHANNEL_SIZE)

        self.board_cnn_pool = torch.nn.AdaptiveAvgPool2d((1, 1))

        self.board_mlp_bag = torch.nn.EmbeddingBag(board_feature_size, self.BOARD_INNER_SIZE-self.BOARD_CHANNEL_SIZE, mode="sum")
        self.board_mlp_norm_1 = torch.nn.BatchNorm1d(self.BOARD_INNER_SIZE-self.BOARD_CHANNEL_SIZE, momentum=self.MOMENTUM)
        self.board_mlp_linear_1 = torch.nn.Linear(self.BOARD_INNER_SIZE-self.BOARD_CHANNEL_SIZE, self.BOARD_INNER_SIZE-self.BOARD_CHANNEL_SIZE, bias=False)
        self.board_mlp_norm_2 = torch.nn.BatchNorm1d(self.BOARD_INNER_SIZE-self.BOARD_CHANNEL_SIZE, momentum=self.MOMENTUM)
        self.board_mlp_linear_2 = torch.nn.Linear(self.BOARD_INNER_SIZE, self.BOARD_INNER_SIZE, bias=False)
        self.board_mlp_norm_3 = torch.nn.BatchNorm1d(self.BOARD_INNER_SIZE, momentum=self.MOMENTUM)

        self.action_bag = torch.nn.EmbeddingBag(action_feature_size, self.ACTION_INNER_SIZE, mode="sum")
        self.action_norm = torch.nn.BatchNorm1d(self.ACTION_INNER_SIZE, momentum=self.MOMENTUM)

        self.vys_linear_1 = torch.nn.Linear(self.VALUE_INNER_SIZE+self.YAKU_INNER_SIZE+self.SCORE_INNER_SIZE, self.VALUE_INNER_SIZE+self.YAKU_INNER_SIZE+self.SCORE_INNER_SIZE, bias=False)
        self.vys_norm_1 = torch.nn.BatchNorm1d(self.VALUE_INNER_SIZE+self.YAKU_INNER_SIZE+self.SCORE_INNER_SIZE, momentum=self.MOMENTUM)
        self.vys_linear_2 = torch.nn.Linear(self.VALUE_INNER_SIZE+self.YAKU_INNER_SIZE+self.SCORE_INNER_SIZE, self.VALUE_INNER_SIZE+self.YAKU_INNER_SIZE+self.SCORE_INNER_SIZE, bias=False)
        self.vys_norm_2 = torch.nn.BatchNorm1d(self.VALUE_INNER_SIZE+self.YAKU_INNER_SIZE+self.SCORE_INNER_SIZE, momentum=self.MOMENTUM)
        
        self.value_linear = torch.nn.Linear(self.VALUE_INNER_SIZE, self.VALUE_INNER_SIZE, bias=False)
        self.value_norm = torch.nn.BatchNorm1d(self.VALUE_INNER_SIZE, momentum=self.MOMENTUM)
        self.value_out = torch.nn.Linear(self.VALUE_INNER_SIZE, 1, bias=True)

        self.yaku_linear = torch.nn.Linear(self.YAKU_INNER_SIZE, self.YAKU_INNER_SIZE, bias=False)
        self.yaku_norm = torch.nn.BatchNorm1d(self.YAKU_INNER_SIZE, momentum=self.MOMENTUM)
        self.yaku_out = torch.nn.Linear(self.YAKU_INNER_SIZE, self.YAKU_NUM, bias=True)
        
        self.score_linear = torch.nn.Linear(self.SCORE_INNER_SIZE, self.SCORE_INNER_SIZE, bias=False)
        self.score_norm = torch.nn.BatchNorm1d(self.SCORE_INNER_SIZE, momentum=self.MOMENTUM)
        self.score_out = torch.nn.Linear(self.SCORE_INNER_SIZE, self.SCORE_NUM, bias=True)

        self.policy_linear = torch.nn.Linear(self.POLICY_INNER_SIZE, self.POLICY_INNER_SIZE, bias=False)
        self.policy_norm = torch.nn.BatchNorm1d(self.POLICY_INNER_SIZE, momentum=self.MOMENTUM)
        self.policy_out = torch.nn.Linear(self.POLICY_INNER_SIZE, 1, bias=True)
        
    def forward(self, board_indexes, board_offsets, board_pictures, action_indexes, action_offsets):
        bp = self.board_cnn_norm2d_1(self.board_cnn_conv_1(board_pictures))
        bp = torch.nn.functional.relu(bp)
        bp = self.board_cnn_norm2d_2(self.board_cnn_conv_2(bp))
        # bp = torch.nn.functional.relu(bp)
        bp = self.board_cnn_norm2d_3(self.board_cnn_conv_3(bp))
        # bp = torch.nn.functional.relu(bp)
        bp = self.board_cnn_norm2d_4(self.board_cnn_conv_4(bp))
        # bp = torch.nn.functional.relu(bp)
        bp = self.board_cnn_norm2d_5(self.board_cnn_conv_5(bp))
        # bp = torch.nn.functional.relu(bp)
        bp = self.board_cnn_pool(bp)
        bp = bp.view(bp.size(0), -1)

        bv = self.board_mlp_bag(board_indexes, board_offsets)
        bv = torch.nn.functional.relu(bv)
        bv = self.board_mlp_norm_1(bv)
        bv = self.board_mlp_linear_1(bv)
        bv = self.board_mlp_norm_2(bv)

        board = torch.cat([bp, bv], dim=1)

        board = self.board_mlp_linear_2(board)
        board = self.board_mlp_norm_3(board)

        action = self.action_bag(action_indexes, action_offsets)
        action = torch.nn.functional.relu(action)

        b_to_vys, b_to_p_with_a = torch.split(board, [self.VALUE_INNER_SIZE+self.YAKU_INNER_SIZE+self.SCORE_INNER_SIZE, self.UNIT_SIZE], dim=1)
        a_to_p_with_b, a_to_p = torch.split(action, [self.UNIT_SIZE, self.UNIT_SIZE], dim=1)

        vys = torch.cat([b_to_vys], dim=1)

        vys = torch.nn.functional.relu(vys)
        vys = self.vys_linear_1(vys)
        vys = torch.nn.functional.relu(vys)
        vys = self.vys_norm_1(vys)
        vys = self.vys_linear_2(vys)
        vys = torch.nn.functional.relu(vys)
        vys = self.vys_norm_2(vys)

        value, yaku, score = torch.split(vys, [self.VALUE_INNER_SIZE, self.YAKU_INNER_SIZE, self.SCORE_INNER_SIZE], dim=1)

        value = self.value_linear(value)
        value = torch.nn.functional.relu(value)
        value = self.value_norm(value)
        value = self.value_out(value)
        value = torch.tanh(value)

        yaku = self.yaku_linear(yaku)
        yaku = torch.nn.functional.relu(yaku)
        yaku = self.yaku_norm(yaku)
        yaku = self.yaku_out(yaku)

        score = self.score_linear(score)
        score = torch.nn.functional.relu(score)
        score = self.score_norm(score)
        score = self.score_out(score)
        score = torch.tanh(score)

        policy = torch.cat([b_to_p_with_a*a_to_p_with_b, a_to_p], dim=1)

        policy = torch.nn.functional.relu(policy)
        policy = self.policy_linear(policy)
        policy = self.policy_norm(policy)
        policy = self.policy_out(policy)
        policy = torch.tanh(policy) * 0.4

        return value, yaku, policy, score