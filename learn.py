import os
import shutil
import time
import random
import torch
import pickle
from collections import deque
from torch import nn

from model.simple_mlp_model import SimpleMlpModel
from model.split_mlp_model import SplitMlpModel
from model.simple_cnn_model import SimpleCnnModel
from feature.feature_vector import BoardFeature, ActionFeature, OptionalActionFeature, DiscardActionFeature
from feature.cnn_feature import CnnBoardFeatures
from actor.simple_mlp_actor import Dataset
from actor.split_mlp_actor import DiscardDataset, OptionalDataset
from actor.simple_cnn_actor import Dataset as SimpleCnnDataset

class LearnerBase:
    def __init__(
        self,
        device: str = "cpu",
        learn_dir: str = "./learn",
    ) -> None:
        self.device = device
        self.learn_dir = learn_dir
    
    @staticmethod
    def get_grad_norm(
        model: torch.nn.Module,
    ) -> float:
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm ** 2
        return total_norm ** 0.5

    def setup_result_directories(
        self,
    ) -> None:
        raise NotImplementedError()

    def learn(
        self,
    ) -> None:
        raise NotImplementedError()

class SimpleMlpLearner(LearnerBase):
    BATCH_SIZE = 5000
    FILE_SIZE = 100
    FILE_MIN = 101
    SAVE_INTERVAL = 100000
    HOLD_MODELS = 10
    DELETE_RATE = 0.5
    YAKU_NUM = 3
    BIN_NUM = 5

    def __init__(
        self,
        model_path: str = None,          # 途中から学習する場合
        device: str = "cpu",
        learn_dir: str = "./learn",
    ) -> None:
        super().__init__(device, learn_dir)
        self.model = SimpleMlpModel(BoardFeature.SIZE, ActionFeature.SIZE)
        if model_path is not None:
            pass    # TODO: モデルの読み込みを書く
        self.device = device
        self.model.to(device)
        self.model.train()

    def get_grad_norm(
        self,
        model
    ) -> float:
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm ** 2
        return total_norm ** 0.5

    def setup_result_directories(
        self,
    ) -> None:
        if os.path.isdir(self.learn_dir):
            shutil.rmtree(self.learn_dir)
        os.makedirs(self.learn_dir, exist_ok=False)
        os.mkdir(os.path.join(self.learn_dir, "learndata"))
        for bin_idx in range(self.BIN_NUM):
            os.mkdir(os.path.join(self.learn_dir, f"learndata/bin_{bin_idx}"))

    def learn(
        self,
    ) -> None:
        self.setup_result_directories()

        paths = deque()
        learned_data, last_saved = 0, 0
        model_path = os.path.join(self.learn_dir, f"model_{learned_data}.pth")
        torch.save(self.model.state_dict(), model_path)
        paths.append(model_path)
        with open(os.path.join(self.learn_dir, f"modelpath.txt"), mode="w") as f:
            f.write(model_path)

        # 学習
        bin_idx = 0
        lr = 1e-5
        while True:
            data_dir = f"./learn/learndata/bin_{bin_idx}"
            optimizer = torch.optim.RMSprop(self.model.parameters(), lr=lr, eps=1e-4)
            files = [os.path.join(data_dir, file) for file in os.listdir(data_dir)] # TODO: 端数が捨てられる問題をどうする？

            if len(files) >= self.FILE_MIN:
                fulldataset = Dataset()
                for file in files:
                    with open(file, mode="rb") as f:
                        ds = pickle.load(f)
                        fulldataset.addrange(ds.data)
                datasets = fulldataset.split(self.BATCH_SIZE)

                value_loss_sum, yaku_loss_sum, policy_loss_sum, score_loss_sum = 0.0, 0.0, 0.0, 0.0
                loss_sum, loss_count = 0.0, 0
                grad_norm = 0.0

                for data in datasets:
                    optimizer.zero_grad()

                    board_indexes, board_offsets, action_indexes, action_offsets = data.make_inputs()
                    board_indexes = torch.LongTensor(board_indexes).to(self.device)
                    board_offsets = torch.LongTensor(board_offsets).to(self.device)
                    action_indexes = torch.LongTensor(action_indexes).to(self.device)
                    action_offsets = torch.LongTensor(action_offsets).to(self.device)
                    value_labels, yaku_labels, policy_labels, score_labels = data.make_labels()

                    value_labels = torch.FloatTensor([[v] for v in value_labels]).to(self.device)
                    yaku_labels = torch.FloatTensor([ys for ys in yaku_labels]).to(self.device)
                    policy_labels = torch.FloatTensor([[p] for p in policy_labels]).to(self.device)
                    score_labels = torch.FloatTensor([ss for ss in score_labels]).to(self.device)

                    value, yaku, policy, score = self.model.forward(board_indexes, board_offsets, action_indexes, action_offsets)

                    value_loss = torch.nn.functional.huber_loss(value, value_labels, reduction="sum", delta=0.2)
                    yaku_loss = torch.nn.functional.cross_entropy(yaku, yaku_labels, reduction="sum")
                    policy_loss = torch.nn.functional.huber_loss(policy, policy_labels, reduction="sum", delta=0.2)
                    score_loss = torch.nn.functional.huber_loss(score, score_labels, reduction="sum")

                    value_loss /= 1.0
                    yaku_loss /= 30.0
                    policy_loss /= 0.4
                    score_loss /= 5.0
                    value_loss_sum += value_loss
                    yaku_loss_sum += yaku_loss
                    policy_loss_sum += policy_loss
                    score_loss_sum += score_loss

                    loss = value_loss + yaku_loss + policy_loss + score_loss

                    loss_count += len(data)
                    learned_data += len(data)

                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 3000.0)
                    grad_norm = self.get_grad_norm(self.model)
                    optimizer.step()

                if loss_count > 0:
                    print(f"bin {bin_idx}: learn files: {len(files)}, value_loss: {value_loss_sum / loss_count}, yaku_loss: {yaku_loss_sum / loss_count}, policy_loss: {policy_loss_sum / loss_count}, score_loss: {score_loss_sum / loss_count}, grad_norm: {grad_norm}, lr: {lr}", flush=True)

                for file in files:
                    r = random.random()
                    if r < self.DELETE_RATE:
                        os.remove(file)

                if learned_data > last_saved + self.SAVE_INTERVAL:
                    model_path = os.path.join(self.learn_dir, f"model_{learned_data}.pth")
                    torch.save(self.model.state_dict(), model_path)
                    last_saved = learned_data
                    print(f"saved: {model_path}")
                    paths.append(model_path)
                    if len(paths) > self.HOLD_MODELS:
                        popped = paths.popleft()
                        os.remove(popped)
                    with open(os.path.join(self.learn_dir, f"modelpath.txt"), mode="w") as f:
                        f.write(model_path)

                bin_idx = (bin_idx + 1) % self.BIN_NUM
            else:
                print(f"bin {bin_idx}: file({len(files)}) < min({self.FILE_MIN})")
            time.sleep(10)

class SplitMlpLearner(LearnerBase):
    BATCH_SIZE = 5000
    FILE_SIZE = 100
    FILE_MIN = 101
    SAVE_INTERVAL = 100000
    HOLD_MODELS = 10
    DELETE_RATE_DISCARD = 0.7
    DELETE_RATE_OPTIONAL = 0.5
    YAKU_NUM = 3
    BIN_NUM = 5

    def __init__(
        self,
        model_path: str = None,          # 途中から学習する場合
        device: str = "cpu",
        learn_dir: str = "./learn",
    ) -> None:
        super().__init__(device, learn_dir)
        self.model = SplitMlpModel(BoardFeature.SIZE, DiscardActionFeature.SIZE, OptionalActionFeature.SIZE)
        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path))
        self.device = device
        self.model.to(device)
        self.model.train()

    def get_grad_norm(
        self,
        model
    ) -> float:
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm ** 2
        return total_norm ** 0.5

    def setup_result_directories(
        self,
    ) -> None:
        if os.path.isdir(self.learn_dir):
            shutil.rmtree(self.learn_dir)
        os.makedirs(self.learn_dir, exist_ok=False)
        os.mkdir(os.path.join(self.learn_dir, "learndata"))
        for bin_idx in range(self.BIN_NUM):
            os.mkdir(os.path.join(self.learn_dir, f"learndata/discard_{bin_idx}"))
            os.mkdir(os.path.join(self.learn_dir, f"learndata/optional_{bin_idx}"))

    def learn(
        self,
    ) -> None:
        self.setup_result_directories()

        paths = deque()
        learned_discard_data, learned_optional_data, last_saved = 0, 0, 0
        model_path = os.path.join(self.learn_dir, f"model_{learned_discard_data}_{learned_optional_data}.pth")
        torch.save(self.model.state_dict(), model_path)
        paths.append(model_path)
        with open(os.path.join(self.learn_dir, f"modelpath.txt"), mode="w") as f:
            f.write(model_path)

        # 学習
        bin_idx = 0
        discard_lr, optional_lr = 1e-5, 1e-5

        while True:
            # optional学習
            optional_dir = f"./learn/learndata/optional_{bin_idx}"
            optimizer = torch.optim.RMSprop(self.model.parameters(), lr=optional_lr, eps=1e-4)
            files = [os.path.join(optional_dir, file) for file in os.listdir(optional_dir)] # TODO: 端数が捨てられる問題をどうする？

            if len(files) >= self.FILE_MIN:
                fulldataset = OptionalDataset()
                for file in files:
                    with open(file, mode="rb") as f:
                        ds = pickle.load(f)
                        fulldataset.addrange(ds.data)
                
                datasets = fulldataset.split(self.BATCH_SIZE)

                value_loss_sum, yaku_loss_sum, policy_loss_sum, score_loss_sum = 0.0, 0.0, 0.0, 0.0
                loss_sum, loss_count = 0.0, 0
                grad_norm = 0.0

                for data in datasets:
                    optimizer.zero_grad()

                    board_indexes, board_offsets, action_indexes, action_offsets = data.make_inputs()
                    board_indexes = torch.LongTensor(board_indexes).to(self.device)
                    board_offsets = torch.LongTensor(board_offsets).to(self.device)
                    action_indexes = torch.LongTensor(action_indexes).to(self.device)
                    action_offsets = torch.LongTensor(action_offsets).to(self.device)
                    value_labels, yaku_labels, policy_labels, score_labels = data.make_labels()

                    value_labels = torch.FloatTensor([[v] for v in value_labels]).to(self.device)
                    yaku_labels = torch.FloatTensor([ys for ys in yaku_labels]).to(self.device)
                    policy_labels = torch.FloatTensor([[p] for p in policy_labels]).to(self.device)
                    score_labels = torch.FloatTensor([ss for ss in score_labels]).to(self.device)

                    value, yaku, policy, score = self.model.forward_optional(board_indexes, board_offsets, action_indexes, action_offsets)

                    value_loss = torch.nn.functional.huber_loss(value, value_labels, reduction="sum", delta=0.2)
                    yaku_loss = torch.nn.functional.cross_entropy(yaku, yaku_labels, reduction="sum")
                    policy_loss = torch.nn.functional.huber_loss(policy, policy_labels, reduction="sum", delta=0.2)
                    score_loss = torch.nn.functional.huber_loss(score, score_labels, reduction="sum")

                    value_loss /= 1.0
                    yaku_loss /= 30.0
                    policy_loss /= 0.4
                    score_loss /= 5.0
                    value_loss_sum += value_loss
                    yaku_loss_sum += yaku_loss
                    policy_loss_sum += policy_loss
                    score_loss_sum += score_loss

                    loss = value_loss + yaku_loss + policy_loss + score_loss

                    loss_count += len(data)
                    learned_optional_data += len(data)

                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 3000.0)
                    grad_norm = self.get_grad_norm(self.model)
                    optimizer.step()

                if loss_count > 0:
                    print(f"OPTIONAL:: bin {bin_idx}: learn files: {len(files)}, value_loss: {value_loss_sum / loss_count}, yaku_loss: {yaku_loss_sum / loss_count}, policy_loss: {policy_loss_sum / loss_count}, score_loss: {score_loss_sum / loss_count}, grad_norm: {grad_norm}, lr: {optional_lr}", flush=True)

                for file in files:
                    r = random.random()
                    if r < self.DELETE_RATE_OPTIONAL:
                        os.remove(file)
            else:
                print(f"optional:: bin {bin_idx}: file({len(files)}) < min({self.FILE_MIN})")

            # discard学習
            discard_dir = f"./learn/learndata/discard_{bin_idx}"
            optimizer = torch.optim.RMSprop(self.model.parameters(), lr=discard_lr, eps=1e-4)
            files = [os.path.join(discard_dir, file) for file in os.listdir(discard_dir)]

            if len(files) >= self.FILE_MIN:
                dc_start = time.time()
                fulldataset = DiscardDataset()
                for file in files:
                    with open(file, mode="rb") as f:
                        ds = pickle.load(f)
                        fulldataset.addrange(ds.data)
                dc_read = time.time()
                datasets = fulldataset.split(self.BATCH_SIZE)

                value_loss_sum, yaku_loss_sum, discard_loss_sum, score_loss_sum = 0.0, 0.0, 0.0, 0.0
                loss_sum, loss_count = 0.0, 0
                grad_norm = 0.0
                dc_learn = time.time()
                for data in datasets:
                    optimizer.zero_grad()

                    board_indexes, board_offsets, action_indexes, action_offsets, valid_masks = data.make_inputs()
                    board_indexes = torch.LongTensor(board_indexes).to(self.device)
                    board_offsets = torch.LongTensor(board_offsets).to(self.device)
                    action_indexes = torch.LongTensor(action_indexes).to(self.device)
                    action_offsets = torch.LongTensor(action_offsets).to(self.device)
                    valid_tensor = torch.tensor(valid_masks, dtype=torch.bool)

                    value_labels, yaku_labels, discard_idxes, policy_labels, score_labels = data.make_labels(valid_masks)
                    value_labels = torch.FloatTensor([[v] for v in value_labels]).to(self.device)
                    yaku_labels = torch.FloatTensor([ys for ys in yaku_labels]).to(self.device)
                    discard_idxes = torch.LongTensor([d for d in discard_idxes]).to(self.device)
                    policy_labels = torch.FloatTensor([p for p in policy_labels]).to(self.device)
                    score_labels = torch.FloatTensor([ss for ss in score_labels]).to(self.device)

                    value, yaku, discard, score = self.model.forward_discard(board_indexes, board_offsets, action_indexes, action_offsets, valid_tensor)
                    value_loss = torch.nn.functional.huber_loss(value, value_labels, reduction="sum", delta=0.2)
                    yaku_loss = torch.nn.functional.cross_entropy(yaku, yaku_labels, reduction="sum")
                    score_loss = torch.nn.functional.huber_loss(score, score_labels, reduction="sum")
                    discard_loss = torch.nn.functional.cross_entropy(discard, discard_idxes, reduction="none")
                    discard_loss *= policy_labels
                    discard_loss = discard_loss.sum()

                    value_loss /= 0.5
                    yaku_loss /= 30.0
                    discard_loss /= 10.0
                    score_loss /= 5.0
                    value_loss_sum += value_loss
                    yaku_loss_sum += yaku_loss
                    discard_loss_sum += discard_loss
                    score_loss_sum += score_loss

                    loss = value_loss + yaku_loss + discard_loss + score_loss
                    loss_count += len(data)
                    learned_discard_data += len(data)

                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1000.0)
                    grad_norm = self.get_grad_norm(self.model)
                    optimizer.step()

                print(f"DISCARD:: bin {bin_idx}: learn files: {len(files)}, value_loss: {value_loss_sum / loss_count}, yaku_loss: {yaku_loss_sum / loss_count}, discard_loss: {discard_loss_sum / loss_count}, score_loss: {score_loss_sum / loss_count}, grad_norm: {grad_norm}, lr: {discard_lr}", flush=True)

                for file in files:
                    r = random.random()
                    if r < self.DELETE_RATE_DISCARD:
                        os.remove(file)

                if learned_discard_data > last_saved + self.SAVE_INTERVAL:
                    model_path = os.path.join(self.learn_dir, f"model_{learned_discard_data}_{learned_optional_data}.pth")
                    torch.save(self.model.state_dict(), model_path)
                    last_saved = learned_discard_data
                    print(f"saved: {model_path}")
                    paths.append(model_path)
                    if len(paths) > self.HOLD_MODELS:
                        popped = paths.popleft()
                        os.remove(popped)
                    with open(os.path.join(self.learn_dir, f"modelpath.txt"), mode="w") as f:
                        f.write(model_path)
                
                bin_idx = (bin_idx + 1) % self.BIN_NUM
                dc_end = time.time()
                print(f"read: {dc_read - dc_start}, learn: {dc_end - dc_learn}")
            else:
                print(f"discard:: bin {bin_idx}: file({len(files)}) < min({self.FILE_MIN})")
            time.sleep(5)
        
class SimpleCnnLearner(LearnerBase):
    BATCH_SIZE = 5000
    FILE_SIZE = 100
    FILE_MIN = 101
    SAVE_INTERVAL = 100000
    HOLD_MODELS = 10
    DELETE_RATE = 0.5
    YAKU_NUM = 3
    BIN_NUM = 5

    def __init__(
        self,
        model_path: str = None,          # 途中から学習する場合
        device: str = "cpu",
        learn_dir: str = "./learn",
    ) -> None:
        super().__init__(device, learn_dir)
        self.model = SimpleCnnModel(BoardFeature.SIZE, CnnBoardFeatures.CHANNEL, ActionFeature.SIZE)
        if model_path is not None:
            pass    # TODO: モデルの読み込みを書く
        self.device = device
        self.model.to(device)
        self.model.train()

    def get_grad_norm(
        self,
        model
    ) -> float:
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm ** 2
        return total_norm ** 0.5

    def setup_result_directories(
        self,
    ) -> None:
        if os.path.isdir(self.learn_dir):
            shutil.rmtree(self.learn_dir)
        os.makedirs(self.learn_dir, exist_ok=False)
        os.mkdir(os.path.join(self.learn_dir, "learndata"))
        for bin_idx in range(self.BIN_NUM):
            os.mkdir(os.path.join(self.learn_dir, f"learndata/bin_{bin_idx}"))

    def learn(
        self,
    ) -> None:
        self.setup_result_directories()

        paths = deque()
        learned_data, last_saved = 0, 0
        model_path = os.path.join(self.learn_dir, f"model_{learned_data}.pth")
        torch.save(self.model.state_dict(), model_path)
        paths.append(model_path)
        with open(os.path.join(self.learn_dir, f"modelpath.txt"), mode="w") as f:
            f.write(model_path)

        # 学習
        bin_idx = 0
        lr = 1e-5
        while True:
            data_dir = f"./learn/learndata/bin_{bin_idx}"
            optimizer = torch.optim.RMSprop(self.model.parameters(), lr=lr, eps=1e-4)
            files = [os.path.join(data_dir, file) for file in os.listdir(data_dir)] # TODO: 端数が捨てられる問題をどうする？

            if len(files) >= self.FILE_MIN:
                fulldataset = SimpleCnnDataset()
                for file in files:
                    with open(file, mode="rb") as f:
                        ds = pickle.load(f)
                        fulldataset.addrange(ds.data)
                datasets = fulldataset.split(self.BATCH_SIZE)

                value_loss_sum, yaku_loss_sum, policy_loss_sum, score_loss_sum = 0.0, 0.0, 0.0, 0.0
                loss_sum, loss_count = 0.0, 0
                grad_norm = 0.0

                for data in datasets:
                    optimizer.zero_grad()

                    board_indexes, board_offsets, board_pictures, action_indexes, action_offsets = data.make_inputs()
                    board_indexes = torch.LongTensor(board_indexes).to(self.device)
                    board_offsets = torch.LongTensor(board_offsets).to(self.device)
                    action_indexes = torch.LongTensor(action_indexes).to(self.device)
                    action_offsets = torch.LongTensor(action_offsets).to(self.device)
                    board_pictures = torch.from_numpy(board_pictures).to(self.device)
                    value_labels, yaku_labels, policy_labels, score_labels = data.make_labels()

                    value_labels = torch.FloatTensor([[v] for v in value_labels]).to(self.device)
                    yaku_labels = torch.FloatTensor([ys for ys in yaku_labels]).to(self.device)
                    policy_labels = torch.FloatTensor([[p] for p in policy_labels]).to(self.device)
                    score_labels = torch.FloatTensor([ss for ss in score_labels]).to(self.device)

                    value, yaku, policy, score = self.model.forward(board_indexes, board_offsets, board_pictures, action_indexes, action_offsets)

                    value_loss = torch.nn.functional.huber_loss(value, value_labels, reduction="sum", delta=0.2)
                    yaku_loss = torch.nn.functional.cross_entropy(yaku, yaku_labels, reduction="sum")
                    policy_loss = torch.nn.functional.huber_loss(policy, policy_labels, reduction="sum", delta=0.2)
                    score_loss = torch.nn.functional.huber_loss(score, score_labels, reduction="sum")

                    value_loss /= 1.0
                    yaku_loss /= 30.0
                    policy_loss /= 0.4
                    score_loss /= 5.0
                    value_loss_sum += value_loss
                    yaku_loss_sum += yaku_loss
                    policy_loss_sum += policy_loss
                    score_loss_sum += score_loss

                    loss = value_loss + yaku_loss + policy_loss + score_loss

                    loss_count += len(data)
                    learned_data += len(data)

                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 3000.0)
                    grad_norm = self.get_grad_norm(self.model)
                    optimizer.step()

                if loss_count > 0:
                    print(f"bin {bin_idx}: learn files: {len(files)}, value_loss: {value_loss_sum / loss_count}, yaku_loss: {yaku_loss_sum / loss_count}, policy_loss: {policy_loss_sum / loss_count}, score_loss: {score_loss_sum / loss_count}, grad_norm: {grad_norm}, lr: {lr}", flush=True)

                for file in files:
                    r = random.random()
                    if r < self.DELETE_RATE:
                        os.remove(file)

                if learned_data > last_saved + self.SAVE_INTERVAL:
                    model_path = os.path.join(self.learn_dir, f"model_{learned_data}.pth")
                    torch.save(self.model.state_dict(), model_path)
                    last_saved = learned_data
                    print(f"saved: {model_path}")
                    paths.append(model_path)
                    if len(paths) > self.HOLD_MODELS:
                        popped = paths.popleft()
                        os.remove(popped)
                    with open(os.path.join(self.learn_dir, f"modelpath.txt"), mode="w") as f:
                        f.write(model_path)

                bin_idx = (bin_idx + 1) % self.BIN_NUM
            else:
                print(f"bin {bin_idx}: file({len(files)}) < min({self.FILE_MIN})")
            time.sleep(20)
        
def main():
    learner = SimpleMlpLearner(model_path=None, device="cuda:0", learn_dir="./learn")
    # learner = SplitMlpLearner(model_path=None, device="cuda:0", learn_dir="./learn")
    # learner = SimpleCnnLearner(model_path=None, device="cuda:0", learn_dir="./learn")

    learner.learn()

if __name__ == "__main__":
    main()
