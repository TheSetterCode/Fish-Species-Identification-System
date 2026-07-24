import torch
import torch.nn as nn
from torchvision import models


def load_model():

    model = models.resnet50(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        14
    )

    model.load_state_dict(
        torch.load(
            "fish_model.pth",
            map_location=torch.device("cpu")
        )
    )

    model.eval()

    return model