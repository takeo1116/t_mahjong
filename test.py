import torch

from battle import SingleGame
from learn import LearningConstants
from actor import ShantenActor, Actor
from model import Model
from feature import BoardFeature, ActionFeature


model = Model(BoardFeature.SIZE, ActionFeature.SIZE)
# model.load_state_dict(torch.load("storage/model_0_861390000.pth"))

temperature = 0.01

agents = [
    ShantenActor(model),
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
# game.visualize_one_round(None)
game.run()

print([(data.data_type, data.value_label) for data in agents[0].trainer.episodes[0].data])