import torch
import torch.nn as nn

STYLE_LOSS_WEIGHT = 1e5
PERCEPTION_LOSS_WEIGHT = 1e0

def get_loss(y_hat_features, y_c_features):
    mse_loss = nn.MSELoss()
    perception_loss = mse_loss(y_hat_features[2], y_c_features[2])
    return perception_loss





