import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size = 5)      # layer 1
        self.conv2 = nn.Conv2d(10, 20, kernel_size = 5)     # layer 2
        self.conv2_drop = nn.Dropout2d()                    # dropout, prevent overfitting
        self.fc1 = nn.Linear(320, 50)                       # fully connected layer 1
        self.fc2 = nn.Linear(50, 10)                        # output layer

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))          # 5x5 with 10 filters
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))      # 5x5 with 20 filters
        x = x.view(-1, 320)                                 # reshape
        x = F.relu(self.fc1(x))                             # linear layer (320 to 50)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)                                     # linear layer (50 to 10)
        return F.log_softmax(x, dim=1)                      # logits to probabilities
    