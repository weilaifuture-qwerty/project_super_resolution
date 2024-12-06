import torch
import torch.nn as nn
import torchvision
import torchvision.models as models


device = torch.device('mps')
class vgg16(nn.Module):
    def __init__(self):
        super(vgg16, self).__init__()
        vgg16_model = models.vgg16(weights=models.VGG16_Weights).to(device)
        features = vgg16_model.features
        self.relu1_2 = nn.Sequential()
        self.relu2_2 = nn.Sequential()
        self.relu3_3 = nn.Sequential()
        self.relu4_3 = nn.Sequential()
        for i in range(0, 4):
            self.relu1_2.add_module(str(i), features[i])
        for i in range(4, 9):
            self.relu2_2.add_module(str(i), features[i])
        for param in self.parameters():
            param.requires_grad = False
        
    
    def forward(self, x):
        y1 = self.relu1_2(x)
        y2 = self.relu2_2(y1)
        return y2