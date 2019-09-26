import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, models, transforms


class DoodleClassifier(nn.Module):
    def __init__(self, num_classes):
        super(DoodleClassifier, self).__init__()
        self.require_grad = True 

        self.num_classes = num_classes
        self.model = models.resnet152(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = self.require_grad
        #set_parameter_requires_grad(self.model, feature_extract)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, self.num_classes)
        self.input_size = 224
        self.criterion = nn.CrossEntropyLoss()
    def forward(self, x, y):

        y_pred = self.model(x)
        #_, preds = torch.max()

        return self.criterion(y_pred, y)

    def pred(self, x):
        outputs = self.model(x)
        _, preds = torch.max(outputs, 1)
        return preds


