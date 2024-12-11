import os
import time
import random
import torch
from collections import deque

from model import Model
from feature import BoardFeature, HandFeature, ActionFeature
from actor import Dataset, DiscardDataset, OptionalDataset

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
        os.makedirs(os.path.join(learn_dir, f"learndata_{agent_idx}/discard_{bin_idx}"), exist_ok=True)
        os.makedirs(os.path.join(learn_dir, f"learndata_{agent_idx}/optional_{bin_idx}"), exist_ok=True)

    model = Model(BoardFeature.SIZE, ActionFeature.SIZE)
    paths = deque()
    learned_discard_data, learned_optional_data, last_saved = 0, 0, 0
    model_path = os.path.join(learn_dir, f"model_{agent_idx}_{learned_discard_data}_{learned_optional_data}.pth")
    torch.save(model.state_dict(), model_path)
    paths.append(model_path)
    with open(os.path.join(learn_dir, f"modelpath_{agent_idx}.txt"), mode="w") as f:
        f.write(model_path)

    # 学習
    model.to(device)
    model.train()

    bin_idx = 0
    while True:
        # optional学習
        optional_dir = f"./learn/learndata_{0}/optional_{bin_idx}"
        lr = lr_schedule(learned_optional_data)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, eps=1e-4)
        files = [os.path.join(optional_dir, file) for file in os.listdir(optional_dir)] # TODO: 端数が捨てられる問題をどうする？

        if len(files) >= LearningConstants.FILE_MIN:
            b = b""
            for file in files:
                with open(file, mode="rb") as f:
                    b += f.read()
            
            fulldataset = OptionalDataset.from_bytes(b)
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
                value_labels, yaku_labels, policy_labels = data.make_labels()

                value_labels = torch.FloatTensor([[v] for v in value_labels]).to(device)
                yaku_labels = torch.FloatTensor([ys for ys in yaku_labels]).to(device)
                policy_labels = torch.FloatTensor([[p] for p in policy_labels]).to(device)

                value, yaku, policy = model.forward_optional(board_indexes, board_offsets, action_indexes, action_offsets)

                value_loss = torch.nn.functional.huber_loss(value, value_labels, reduction="sum", delta=0.2)
                yaku_loss = torch.nn.functional.cross_entropy(yaku, yaku_labels, reduction="sum")
                policy_loss = torch.nn.functional.huber_loss(policy, policy_labels, reduction="sum", delta=0.2)

                value_loss /= 1.0
                yaku_loss /= 3.0
                policy_loss /= 0.4
                value_loss_sum += value_loss
                yaku_loss_sum += yaku_loss
                policy_loss_sum += policy_loss

                loss = value_loss + yaku_loss + policy_loss

                loss_count += len(data)
                learned_optional_data += len(data)

                loss.backward()
                optimizer.step()

            print(f"OPTIONAL:: bin {bin_idx}: learn files: {len(files)}, value_loss: {value_loss_sum / loss_count}, yaku_loss: {yaku_loss_sum / loss_count}, policy_loss: {policy_loss_sum / loss_count}, lr: {lr}", flush=True)

            for file in files:
                r = random.random()
                if r < LearningConstants.DELETE_RATE:
                    os.remove(file)
        else:
            print(f"optional:: bin {bin_idx}: file({len(files)}) < min({LearningConstants.FILE_MIN})")

        # discard学習
        discard_dir = f"./learn/learndata_{0}/discard_{bin_idx}"
        lr = lr_schedule(learned_discard_data)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, eps=1e-4)
        files = [os.path.join(discard_dir, file) for file in os.listdir(discard_dir)]

        if len(files) >= LearningConstants.FILE_MIN:
            b = b""
            for file in files:
                with open(file, mode="rb") as f:
                    b += f.read()
            
            fulldataset = DiscardDataset.from_bytes(b)
            datasets = fulldataset.split(LearningConstants.BATCH_SIZE)

            value_loss_sum, yaku_loss_sum, discard_loss_sum, illegal_loss_sum = 0.0, 0.0, 0.0, 0.0
            loss_sum, loss_count = 0.0, 0

            for data in datasets:
                optimizer.zero_grad()

                board_indexes, board_offsets = data.make_inputs()
                board_indexes = torch.LongTensor(board_indexes).to(device)
                board_offsets = torch.LongTensor(board_offsets).to(device)
                value_labels, yaku_labels, discard_labels, illegal_labels = data.make_labels()

                value_labels = torch.FloatTensor([[v] for v in value_labels]).to(device)
                yaku_labels = torch.FloatTensor([ys for ys in yaku_labels]).to(device)
                discard_labels = torch.FloatTensor([dl for dl in discard_labels]).to(device)
                illegal_labels = torch.FloatTensor([il for il in illegal_labels]).to(device)

                value, yaku, discard = model.forward_discard(board_indexes, board_offsets)

                value_loss = torch.nn.functional.huber_loss(value, value_labels, reduction="sum", delta=0.2)
                yaku_loss = torch.nn.functional.cross_entropy(yaku, yaku_labels, reduction="sum")
                discard_loss = torch.nn.functional.cross_entropy(discard, discard_labels, reduction="sum")
                illegal_loss = torch.nn.functional.cross_entropy(discard, illegal_labels, reduction="sum")

                value_loss /= 1.0
                yaku_loss /= 3.0
                discard_loss /= 1.0
                illegal_loss /= 3000.0
                value_loss_sum += value_loss
                yaku_loss_sum += yaku_loss
                discard_loss_sum += discard_loss
                illegal_loss_sum += illegal_loss

                loss = value_loss + yaku_loss + discard_loss + illegal_loss
                loss_count += len(data)
                learned_discard_data += len(data)

                loss.backward()
                optimizer.step()

            print(f"DISCARD:: bin {bin_idx}: learn files: {len(files)}, value_loss: {value_loss_sum / loss_count}, yaku_loss: {yaku_loss_sum / loss_count}, discard_loss: {discard_loss_sum / loss_count}, illegal_loss: {illegal_loss_sum / loss_count}, lr: {lr}", flush=True)

            for file in files:
                r = random.random()
                if r < LearningConstants.DELETE_RATE:
                    os.remove(file)

            if learned_discard_data > last_saved + LearningConstants.SAVE_INTERVAL:
                model_path = os.path.join(learn_dir, f"model_{agent_idx}_{learned_discard_data}_{learned_optional_data}.pth")
                torch.save(model.state_dict(), model_path)
                last_saved = learned_discard_data
                print(f"saved: {model_path}")
                paths.append(model_path)
                if len(paths) > LearningConstants.HOLD_MODELS:
                    popped = paths.popleft()
                    os.remove(popped)
                with open(os.path.join(learn_dir, f"modelpath_{agent_idx}.txt"), mode="w") as f:
                    f.write(model_path)
            
            bin_idx = (bin_idx + 1) % LearningConstants.BIN_NUM

        else:
            print(f"discard:: bin {bin_idx}: file({len(files)}) < min({LearningConstants.FILE_MIN})")
        time.sleep(20)



if __name__ == "__main__":
    main()