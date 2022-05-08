#import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.vision.models.resnet import BasicBlock # ,BottleneckBlock

class KeypointNet_Deepest(nn.Layer): # nn.Module

    def __init__(self, input_channels=3, output_channels=5, channel_size=16, 
				 dropout=False, prob=0.1):
        super(KeypointNet_Deepest, self).__init__()

        # nn.initializer.set_global_initializer(nn.initializer.KaimingNormal(), nn.initializer.Normal()) 
        ## nn.initializer.Uniform(), nn.initializer.Constant(); nn.initializer.KaimingUniform(), nn.initializer.Uniform()

        self.conv1 = nn.Conv2D(input_channels, channel_size, kernel_size=9, stride=1, padding=4)
        self.bn1 = nn.BatchNorm2D(channel_size)

        self.conv2 = nn.Conv2D(channel_size, channel_size * 2, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2D(channel_size * 2)

        self.conv3 = nn.Conv2D(channel_size * 2, channel_size * 4, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2D(channel_size * 4)
        
        # Added
        self.conv3_1 = nn.Conv2D(channel_size * 4, channel_size * 8, kernel_size=4, stride=2, padding=1)
        self.bn3_1 = nn.BatchNorm2D(channel_size * 8)
        # Added
        self.conv3_2 = nn.Conv2D(channel_size * 8, channel_size * 16, kernel_size=4, stride=2, padding=1)
        self.bn3_2 = nn.BatchNorm2D(channel_size * 16)

        self.res1 = BasicBlock(channel_size * 16, channel_size * 16)  # BottleneckBlock
        self.res2 = BasicBlock(channel_size * 16, channel_size * 16)
        self.res3 = BasicBlock(channel_size * 16, channel_size * 16)
        self.res4 = BasicBlock(channel_size * 16, channel_size * 16)
        self.res5 = BasicBlock(channel_size * 16, channel_size * 16)
        # self.res5_1 = BasicBlock(channel_size * 16, channel_size * 16) # Added
        # self.res5_2 = BasicBlock(channel_size * 8, channel_size * 8) # Added
        # Added
        # Added
        self.conv3_8 = nn.Conv2DTranspose(channel_size * 16, channel_size * 8, kernel_size=4, stride=2, padding=1,
                                        output_padding=0)
        self.bn3_8 = nn.BatchNorm2D(channel_size * 8)
        # Added
        self.conv3_9 = nn.Conv2DTranspose(channel_size * 8, channel_size * 4, kernel_size=4, stride=2, padding=1,
                                        output_padding=0)
        self.bn3_9 = nn.BatchNorm2D(channel_size * 4)

        self.conv4 = nn.Conv2DTranspose(channel_size * 4, channel_size * 2, kernel_size=4, stride=2, padding=1,
                                        output_padding=1)
        self.bn4 = nn.BatchNorm2D(channel_size * 2)

        self.conv5 = nn.Conv2DTranspose(channel_size * 2, channel_size, kernel_size=4, stride=2, padding=2,
                                        output_padding=1)  #  output_padding=1
        self.bn5 = nn.BatchNorm2D(channel_size)

        self.conv6 = nn.Conv2DTranspose(channel_size, channel_size, kernel_size=9, stride=1, padding=4)
		
        self.bn6 = nn.BatchNorm2D(channel_size)

        self.conv7 = nn.Conv2D(channel_size, output_channels, kernel_size=2)
        
		
        self.dropout = dropout
        # self.dropout_pos = nn.Dropout(p=prob)    # Torch 
        self.dropout_last = nn.Dropout2D(p=prob, data_format='NCHW')

        """
        for m in self.modules():
            if isinstance(m, (nn.Conv2D, nn.Conv2DTranspose)):
                nn.init.xavier_uniform_(m.weight, gain=1)
		"""
    # @paddle.jit.to_static # For model visualization in paddle
    def forward(self, x_in):
        x = F.relu(self.bn1(self.conv1(x_in)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn3_1(self.conv3_1(x))) # Added
        x = F.relu(self.bn3_2(self.conv3_2(x))) # Added
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        # x = self.res5_1(x)  # Added
        # x = self.res5_2(x)  # Added
        x = F.relu(self.bn3_8(self.conv3_8(x)))  # Added
        x = F.relu(self.bn3_9(self.conv3_9(x)))  # Added
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        
        if self.dropout:
            x = self.conv6(x)         # Dropout
            x = self.conv7(self.dropout_last(x))
        else:
            # x = F.relu(self.bn6(self.conv6(x)))
            x = self.conv6(x)
            x = self.conv7(x)

        return x
		