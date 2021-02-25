import torch.nn as nn
import sys
import os
sys.path.append(os.path.abspath(""))
from nets.mobilenetv3 import get_mobilev3_pose_net,MobileNet_Decoder,MobileNet_Head

class MobileCenterNet(nn.Module):
    def __init__(self,num_classes):
        super(MobileCenterNet,self).__init__()
        self.backbone = get_mobilev3_pose_net()
        self.decoder = MobileNet_Decoder(960)
        self.head = MobileNet_Head(channel=64, num_classes=num_classes)

    def forward(self,x):
        feat = self.backbone(x)
        out = self.decoder(feat)
        return self.head(out)

if __name__ == '__main__':
    import torch
    model = MobileCenterNet(10).cuda()
    ip = torch.zeros((2,3,512,512)).cuda()
    out = model(ip)
    print(ip.size())
