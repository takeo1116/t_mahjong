import os
import time
import random
import torch
from collections import deque

from model import Model
from feature import BoardFeature, HandFeature, ActionFeature
from actor import Dataset

class LearningConstants:
    BATTLE_NUM = 10
    BATCH_SIZE = 5000
    FILE_SIZE = 100
    FILE_MIN = 75
    SAVE_INTERVAL = 100000
    HOLD_MODELS = 10
    DELETE_RATE = 0.5
    YAKU_NUM = 3
    BIN_NUM = 5

def lr_schedule(learned_data):
    return 1e-4

def make_yaku_labels(yaku_raw):
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

def main():
    # ディレクトリ生成
    device = "cuda:0"
    learn_dir = "./learn"
    os.makedirs(learn_dir, exist_ok=True)

    agents = [file for file in os.listdir(learn_dir) if os.path.isfile(os.path.join(learn_dir, file)) and file.startswith("modelpath_") and file.endswith(".txt")]
    agent_idx = len(agents)
    agent_idx = 0
    print(f"agent_idx: {agent_idx}")

    os.mkdir(os.path.join(learn_dir, f"learndata_{agent_idx}"))

    for bin_idx in range(LearningConstants.BIN_NUM):
        os.makedirs(os.path.join(learn_dir, f"learndata_{agent_idx}/bin_{bin_idx}"), exist_ok=True)

    model = Model(BoardFeature.SIZE, ActionFeature.SIZE)
    paths = deque()
    learned_data, last_saved = 0, 0
    model_path = os.path.join(learn_dir, f"model_{agent_idx}_{learned_data}.pth")
    torch.save(model.state_dict(), model_path)
    paths.append(model_path)
    with open(os.path.join(learn_dir, f"modelpath_{agent_idx}.txt"), mode="w") as f:
        f.write(model_path)

    # 学習
    model.to(device)
    model.train()

    bin_idx = 0
    while True:
        data_dir = f"./learn/learndata_{0}/bin_{bin_idx}"
        lr = lr_schedule(learned_data)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, eps=1e-4)
        files = [os.path.join(data_dir, file) for file in os.listdir(data_dir)] # TODO: 端数が捨てられる問題をどうする？
        if len(files) >= LearningConstants.FILE_MIN:
            b = b""
            for file in files:
                with open(file, mode="rb") as f:
                    b += f.read()

            fulldataset = Dataset.from_bytes(b)
            datasets = fulldataset.split(LearningConstants.BATCH_SIZE)

            value_loss_sum, yaku_loss_sum, policy_loss_sum = 0.0, 0.0, 0.0
            loss_sum, loss_count = 0.0, 0

            for data in datasets:
                optimizer.zero_grad()

                board_indexes, board_offsets, action_indexes, action_offsets = data.make_inputs()
                board_indexes = torch.LongTensor(board_indexes).to(device)
                board_offsets = torch.LongTensor(board_offsets).to(device)
                action_indexes = torch.LongTensor(action_indexes).to(device)
                action_offsets = torch.LongTensor(action_offsets).to(device)

                value_labels, yaku_raw, policy_labels = data.make_labels()
                yaku_labels = [make_yaku_labels(yaku) for yaku in yaku_raw]

                value_labels = torch.FloatTensor([[v] for v in value_labels]).to(device)
                yaku_labels = torch.FloatTensor([ys for ys in yaku_labels]).to(device)
                policy_labels = torch.FloatTensor([[p] for p in policy_labels]).to(device)

                out = model(board_indexes, board_offsets, action_indexes, action_offsets)
                value, yaku, policy = torch.split(out, [1, LearningConstants.YAKU_NUM, 1], dim=1)

                value_loss = torch.nn.functional.huber_loss(value, value_labels, reduction="sum", delta=0.2)
                yaku_loss = torch.nn.functional.cross_entropy(yaku, yaku_labels, reduction="sum")
                policy_loss = torch.nn.functional.huber_loss(policy, policy_labels, reduction="sum", delta=0.2)

                value_loss /= 1.0
                yaku_loss /= 30.0
                policy_loss /= 0.4

                value_loss_sum += value_loss
                yaku_loss_sum += yaku_loss
                policy_loss_sum += policy_loss

                loss = value_loss + yaku_loss + policy_loss

                loss_count += len(data)
                learned_data += len(data)

                loss.backward()

                optimizer.step()

            print(f"bin {bin_idx}: learn files: {len(files)}, value_loss: {value_loss_sum / loss_count}, yaku_loss: {yaku_loss_sum / loss_count}, policy_loss: {policy_loss_sum / loss_count}, lr: {lr}", flush=True)

            for file in files:
                r = random.random()
                if r < LearningConstants.DELETE_RATE:
                    os.remove(file)

            if learned_data > last_saved + LearningConstants.SAVE_INTERVAL:
                model_path = os.path.join(learn_dir, f"model_{agent_idx}_{learned_data}.pth")
                torch.save(model.state_dict(), model_path)
                last_saved = learned_data
                print(f"saved: {model_path}")
                paths.append(model_path)
                if len(paths) > LearningConstants.HOLD_MODELS:
                    popped = paths.popleft()
                    os.remove(popped)
                with open(os.path.join(learn_dir, f"modelpath_{agent_idx}.txt"), mode="w") as f:
                    f.write(model_path)

            bin_idx = (bin_idx + 1) % LearningConstants.BIN_NUM  
        else:
            print(f"bin {bin_idx}: file({len(files)}) < min({LearningConstants.FILE_MIN})")
        time.sleep(20)

if __name__ == "__main__":
    main()