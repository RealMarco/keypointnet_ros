import paddle
import paddle.nn as nn
from paddle.vision.models import resnet34 # ,resnet50,resnet18

# dynamic graph to static graph
class Model_resnet34(nn.Layer):
    """
    simply create a 2-branch network, and concat global pooled feature vector.
    each branch = single resnet34
    """
    def __init__(self):
        super(Model_resnet34, self).__init__()
        self.branch = resnet34(pretrained=False, num_classes=0) # remove final fc; # pretrained=True
        self.decision_branch = nn.Linear(512*1 , 3) # 512* 1 * 2 ResNet34 use basic block, expansion = 1
        
    
    @paddle.jit.to_static
    def forward(self, img):
        b1 = self.branch(img)
        b1 = paddle.flatten(b1, 1)
        logit = self.decision_branch(b1)

        return logit

