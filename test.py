import torch

from battle import SingleGame
from learn import LearningConstants
from actor import ShantenActor, MenzenActor, Actor
from model import Model
from feature import BoardFeature, DiscardActionFeature, OptionalActionFeature


model = Model(BoardFeature.SIZE, DiscardActionFeature.SIZE, OptionalActionFeature.SIZE)
model.load_state_dict(torch.load("./learn/model_0_13060000_3535000.pth"))

temperature = 1.0

# agents = [
#     MenzenActor(model),
#     Actor(model),
#     Actor(model),
#     Actor(model),
# ]

# agents = [
#     ShantenActor(model),
#     Actor(model),
#     Actor(model),
#     Actor(model),
# ]

# agents = [
#     MenzenActor(model),
#     MenzenActor(model),
#     MenzenActor(model),
#     MenzenActor(model),
# ]

agents = [
    Actor(model),
    Actor(model),
    Actor(model),
    Actor(model),
]

game = SingleGame(agents)
game.visualize_one_round(None)
# game.run()

# print([(data.data_type, data.value_label) for data in agents[0].trainer.episodes[0].data])