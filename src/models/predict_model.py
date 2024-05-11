import torch
from resnet import ResNet18

model = ResNet18(10)

input = torch.rand(1, 3 ,112 , 112)
model = ResNet18(10)

x = model(input)