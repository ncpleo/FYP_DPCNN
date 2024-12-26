import torch
import torch.nn as nn
import torch.nn.functional as F
from .BasicModule import BasicModule
import torch.nn.init as init

class DPCNN(BasicModule):
    """
    DPCNN for sentences classification.
    """
    def __init__(self, config):
        super(DPCNN, self).__init__()
        self.config = config
        self.channel_size = 250

        # Region embedding convolution
        self.conv_region_embedding = nn.Conv2d(
            in_channels=1,
            out_channels=self.channel_size,
            kernel_size=(3, config.sentence_max_size),
            stride=1
        )
        self.batch_norm_region = nn.BatchNorm2d(self.channel_size)

        # 3x1 convolution
        self.conv3 = nn.Conv2d(self.channel_size, self.channel_size, (3, 1), stride=1)
        self.batch_norm_conv3 = nn.BatchNorm2d(self.channel_size)

        # Pooling and activation
        self.pooling = nn.MaxPool2d(kernel_size=(3, 1), stride=2)
        self.act_fun = nn.ReLU()

        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.6)

        # Fully connected output layer
        self.linear_out = nn.Linear(self.channel_size, config.label_num)

    def forward(self, x):
        #print(f"Input shape: {x.shape}")
        batch = x.size(0)

        # Region embedding
        x = self.conv_region_embedding(x)
        x = self.batch_norm_region(x)  # Apply BatchNorm
        x = self.act_fun(x)
        x = self.dropout(x)

        # convolution block
        x = self.conv3(x)
        x = self.batch_norm_conv3(x)  # Apply BatchNorm
        x = self.act_fun(x)
        x = self.dropout(x)
        
        # Downsampling block
        while x.size()[-2] > 2:
            x = self._block(x)

        # Flatten and fully connected layer
        if x.size(2) == 1 and x.size(3) == 1:
            x = x.view(batch, self.channel_size)
        else:
            x = x.view(batch, self.channel_size, -1)
            x = x.mean(dim=2)  # Average pooling over spatial dimensions

        x = self.linear_out(x)
        return x
        

    def _block(self, x):
        # Pooling operation
        px = self.pooling(x)

        x = F.pad(px, (0, 0, 1, 1))  # Padding for height dimension
        x = F.relu(self.conv3(x))
        
        x = F.pad(x, (0, 0, 1, 1))
        x = F.relu(self.conv3(x))
        
        return px + x  # Shortcut connection (residual block)

    def predict(self, x):
        self.eval()
        out = self.forward(x)
        predict_labels = torch.max(out, 1)[1]
        self.train(mode=True)
        return predict_labels
