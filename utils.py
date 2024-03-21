

from torch import nn
import torch
import torch.nn.functional as F

from MobileNet_v3 import Bottleneck as MobileNet_v3_Bottleneck


class Up_inner(nn.Module):   
    def __init__(self, ch_in, ch_out,):
        super().__init__()  
        self.conv = MobileNet_v3_Bottleneck(ch_in, ch_out, kernel_size=1, exp_channels=3*ch_out, stride=1, se='True', nl='HS')
        
    def forward(self, x) :
        x = self.conv(x)    
        return x



class Up_and_Deep_Supervision_Attention(nn.Module):
    def __init__(self, ch_in, ch_out,):
        super().__init__()        
        self.Upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.Up_inner = Up_inner(ch_in, ch_out)

    def forward(self, x, mask) :
        z = x * mask
        x = self.Up_inner(z)
        x = self.Upsample(x)
        return x




class Time_Fusion(nn.Module):
    def __init__(self, ch_in, ch_out,):
        super().__init__()   
        self.Time_Fusion = nn.Sequential(
            MobileNet_v3_Bottleneck(ch_in*2, ch_out, kernel_size=3, exp_channels=ch_out, stride=1, se='True', nl='HS'),
            )
    
    def forward(self, x, y) :
        z_mix = self.Time_Fusion(torch.cat([x, y], 1))
        return z_mix




class Multi_Scale_Aggregation_inner(nn.Module):    
    def __init__(self, ch_in, ch_out=2):
        super().__init__()
        self.conv_1x1 =  nn.Sequential(
            nn.Conv2d(ch_in, ch_in, kernel_size=1), 
            nn.BatchNorm2d(ch_in),
            nn.Hardswish()
            )
        self.conv_3x3 =  nn.Sequential(
            nn.Conv2d(ch_in, ch_in, kernel_size=3, padding=1), 
            nn.BatchNorm2d(ch_in),
            nn.Hardswish()
            )
        self.conv = nn.Conv2d(ch_in*3, ch_out, kernel_size=1)
        
    def forward(self, x) :
        y1 = self.conv_1x1(x)
        y2 = self.conv_3x3(x)
        out = self.conv(torch.cat([x, y1, y2], dim=1))
        return out

class Multi_Scale_Aggregation(nn.Module):
    def __init__(self, ch_in, ch_out=2, ):
        super().__init__()
        
        self.Upsample_list = nn.ModuleList([
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True),
            nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True),
            nn.Upsample(scale_factor=16, mode="bilinear", align_corners=True),
            ])
        
        self.conv = Multi_Scale_Aggregation_inner(ch_in, ch_out)
            
    def forward(self, out_fs) :
        out_fs_sameWH=[]
        for i in range(len(out_fs)):
            if i==0:
                out_fs_sameWH.append(out_fs[i])
            else :
                out_fs_sameWH.append(self.Upsample_list[i-1](out_fs[i]))
        out_fs_sameWH = torch.cat(out_fs_sameWH, dim=1)
        out = self.conv(out_fs_sameWH)
        return out




class Predicted_Map_for_Main_Path(nn.Module):   
    def __init__(self, ch_in, ch_out=1,):
        super().__init__()         
        self.conv = nn.Sequential(
            MobileNet_v3_Bottleneck(ch_in, ch_out, kernel_size=1, exp_channels=3*ch_in, stride=1, se='True', nl='HS'),
            nn.Sigmoid(),
            )
    
    def forward(self, x) :
        x = self.conv(x)
        return x





    

class TF_Enhance_Encoder(nn.Module):
    def __init__(self, ch_in):
        super().__init__()         
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in*2, ch_in, kernel_size=1),
            nn.BatchNorm2d(ch_in),
            nn.ReLU(),
            )
    
    def forward(self, x, z_mix) :
        x2 = self.conv(torch.cat([x, z_mix], 1))
        x = x + x2
        return x

class Time_Fusion_and_Enhance_Encoder(nn.Module):
    def __init__(self, ch_in):
        super().__init__()
        self.Time_Fusion = Time_Fusion(ch_in, ch_in)  
        self.TF_Enhance_Encoder = TF_Enhance_Encoder(ch_in)
        
    def forward(self, x, y) :
        z_mix = self.Time_Fusion(x, y)
        x = self.TF_Enhance_Encoder(x, z_mix)
        y = self.TF_Enhance_Encoder(y, z_mix)
        
        return x,y,z_mix





class Predicted_Map_for_Branch_Path(nn.Module):   
    def __init__(self, ch_input, ch_in, ch_out=1,):
        super().__init__()         
        self.conv = nn.Sequential(
            MobileNet_v3_Bottleneck(ch_input, ch_in, kernel_size=3, exp_channels=3*ch_in, stride=1, se='True', nl='HS'),
            MobileNet_v3_Bottleneck(ch_in, ch_out, kernel_size=1, exp_channels=ch_in, stride=1, se='True', nl='HS'),
            nn.Sigmoid(),
            )
    
    def forward(self, x) :
        x = self.conv(x)
        return x

class NeighborImg_Core(nn.Module):
    def __init__(self):
        super().__init__()
        self.upsamplex3 = nn.Upsample(scale_factor=3, mode='bilinear')
        
    def forward(self, img_Neighbor):
        img_Neighbor=self.upsamplex3(img_Neighbor)
        b,c,w,h=img_Neighbor.shape
        w1=int(w/3)
        w2=int(w*2/3)
        img_Neighbor_core=img_Neighbor[: , : , w1:w2, w1:w2]
        return img_Neighbor_core

class Center_Neighborhood_Fusion(nn.Module):
    def __init__(self, ch_in, fout):
        super().__init__()
        self.conv = Predicted_Map_for_Branch_Path(ch_in*4, ch_in, fout)  
        self.NeighborImg_Core = NeighborImg_Core()
        
    def forward(self, A, B, AB_mix, A_Neighbor, B_Neighbor, AB_mix_Neighbor) :
        A_B_Neighbor_core = self.NeighborImg_Core(torch.abs(A_Neighbor - B_Neighbor))
        AB_mix_Neighbor_core = self.NeighborImg_Core(AB_mix_Neighbor)        
        #########
        z = self.conv(torch.cat([torch.abs(A - B), AB_mix, A_B_Neighbor_core, AB_mix_Neighbor_core], 1))        
        return z