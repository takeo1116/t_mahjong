import torch

from battle import SingleGame
from learn import LearningConstants
from actor import ShantenActor, Actor
from model import Model
from feature import BoardFeature, HandFeature, ActionFeature


model = Model(BoardFeature.SIZE, ActionFeature.SIZE)
model.load_state_dict(torch.load("learn/model_0_23245000.pth"))

temperature = 0.01

agents = [
    Actor(model),
    Actor(model),
    Actor(model),
    Actor(model),
]

# agents = [
#     Actor(model, temperature),
#     Actor(model, temperature),
#     Actor(model, temperature),
#     Actor(model, temperature),
# ]

game = SingleGame(agents)
game.visualize_one_round(None)