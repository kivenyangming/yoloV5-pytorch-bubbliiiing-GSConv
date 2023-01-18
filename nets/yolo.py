import torch
from torch import nn
from nets.CSPdarknet import CSPDarknet, SPPF, Concat, GSConv, VoVGSCSP


class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes, phi, pretrained=False, input_shape=[640, 640]):
        super(YoloBody, self).__init__()
        depth_dict = {'n': 0.33, 's': 0.33, 'm': 0.67, 'l': 1.00, 'x': 1.33, }
        width_dict = {'n': 0.25, 's': 0.50, 'm': 0.75, 'l': 1.00, 'x': 1.25, }
        dep_mul, wid_mul = depth_dict[phi], width_dict[phi]

        base_channels = int(wid_mul * 64)  # 64
        base_depth = max(round(dep_mul * 3), 1)  # 3
        # -----------------------------------------------#
        #   输入图片是640, 640, 3
        #   初始的基本通道是64
        # -----------------------------------------------#
        self.backbone = CSPDarknet(base_channels, base_depth, phi, pretrained)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.concat = Concat(dimension=1)
        self.SPPF = SPPF(base_channels * 16, base_channels * 16)  # 1024 ---> 1024
        self.P5GSConv = GSConv(base_channels * 16, base_channels * 8)  # 1,1024,20,20 ---> 1,512,20,20
        self.P4VoV = VoVGSCSP(base_channels * 16, base_channels * 8)  # 1,512,40,40 ---> 1,1024,40,40
        """
        self.P4VoV = nn.Sequential(VoVGSCSP(base_channels * 16, base_channels * 8),
                           VoVGSCSP(base_channels * 8, base_channels * 8),
                           VoVGSCSP(base_channels * 8, base_channels * 8))
        """

        self.P4GSConv = GSConv(base_channels * 8, base_channels * 4)  # 1,512,40,40 ---> 1,256,40,40
        self.Head1VoV = VoVGSCSP(base_channels * 8, base_channels * 4)  # 1,512,80,80 ---> 1,256,80,80
        """
        self.Head1VoV = nn.Sequential(VoVGSCSP(base_channels * 8, base_channels * 4),
                                      VoVGSCSP(base_channels * 4, base_channels * 4),
                                      VoVGSCSP(base_channels * 4, base_channels * 4))
        """
        self.P3GSConv = GSConv(base_channels * 4, base_channels * 4, 3, 2)  # 1,256,80,80 ---> 1,256,40,40
        self.Head2VoV = VoVGSCSP(base_channels * 8, base_channels * 8)  # 1,512,40,40 ---> 1,512,40,40
        """
        self.Head2VoV = nn.Sequential(VoVGSCSP(base_channels * 8, base_channels * 8),
                              VoVGSCSP(base_channels * 8, base_channels * 8),
                              VoVGSCSP(base_channels * 8, base_channels * 8))
        """
        self.Head2GSConv = GSConv(base_channels * 8, base_channels * 8, 3, 2)  # 1,512,40,40 ---> 1,512,20,20
        self.Head3VoV = VoVGSCSP(base_channels * 16, base_channels * 16)  # 1,1024,20,20 ---> 1,1024,20,20

        """
        self.Head3VoV = nn.Sequential(VoVGSCSP(base_channels * 16, base_channels * 16),
                                        VoVGSCSP(base_channels * 16, base_channels * 16),
                                        VoVGSCSP(base_channels * 16, base_channels * 16))
        
        """
        self.yolo_head_P3 = nn.Conv2d(base_channels * 4, len(anchors_mask[2]) * (5 + num_classes), 1)
        self.yolo_head_P4 = nn.Conv2d(base_channels * 8, len(anchors_mask[1]) * (5 + num_classes), 1)
        self.yolo_head_P5 = nn.Conv2d(base_channels * 16, len(anchors_mask[0]) * (5 + num_classes), 1)

    def forward(self, x):
        P3, P4, P5 = self.backbone(x)
        P5 = self.SPPF(P5)
        P5 = self.P5GSConv(P5)
        P5_Up = self.upsample(P5)

        P4 = self.concat([P4, P5_Up])
        P4 = self.P4VoV(P4)
        P4 = self.P4GSConv(P4)
        P4_Up = self.upsample(P4)

        P3 = self.concat([P3, P4_Up])
        head1 = self.Head1VoV(P3)
        P3 = self.P3GSConv(head1)
        P34_Cat = self.concat([P3, P4])
        head2 = self.Head2VoV(P34_Cat)
        PHG = self.Head2GSConv(head2)
        PHG_Cat = self.concat([PHG, P5])
        head3 = self.Head3VoV(PHG_Cat)

        Out1 = self.yolo_head_P3(head1)  # 1,255,80,80
        Out2 = self.yolo_head_P4(head2)  # 1,255,40,40
        Out3 = self.yolo_head_P5(head3)  # 1,255,20,20

        return Out3, Out2, Out1


# if __name__ == "__main__":
#     anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
#     num_classes = 80
#     phi = 's'
#     model = YoloBody(anchors_mask, num_classes, phi, pretrained=False)
#     x = torch.ones((1, 3, 640, 640))
#     Out3, Out2, Out1 = model(x)
#     print()
