# The code of INSINet 


import torchvision

from utils import Up_and_Deep_Supervision_Attention, Time_Fusion, Multi_Scale_Aggregation, \
                   Predicted_Map_for_Main_Path, Time_Fusion_and_Enhance_Encoder, Center_Neighborhood_Fusion

from torch import nn
import torch.nn.functional as F
import torch
import copy

# This function constructs the backbone of INSINet using a lightweight MobileNetV3-Large architecture
def _get_backbone_list_mobilenet_v3_large_lightweight(in_dim=3, pretrained=True, freeze_backbone=False):    
    bkbn_name='mobilenet_v3_large'
    entire_model = getattr(torchvision.models, bkbn_name)(pretrained=pretrained).features[:-6]   
    
    derived_model = entire_model
    
    # Freezing the backbone weights:
    if freeze_backbone:
        for param in derived_model.parameters():
            param.requires_grad = False
    
    first_layer = nn.Sequential(
        nn.Conv2d(in_dim, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
        nn.BatchNorm2d(16, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
        nn.Hardswish(),
        derived_model[1],
        )
    
    backbone_list = nn.ModuleList([
        first_layer,           # ->[1, 16, 128, 128]
        derived_model[2:4],    # ->[1, 24, 64, 64]
        derived_model[4:7],    # ->[1, 40, 32, 32]
        derived_model[7:],     # ->[1, 80, 16, 16]            
        ])
    
    return backbone_list



# The code of INSINet 
class INSINet(nn.Module):
    def __init__(
        self,
        in_dim=3,
        out_dim=2,
    ):
        super().__init__()

        # Initialize the backbone for both center and neighborhood images using MobileNetV3.
        ##################Image
        self.backbone = _get_backbone_list_mobilenet_v3_large_lightweight(in_dim=in_dim)
        ##################NeighborImage
        self.backbone_Neighbor = copy.deepcopy(self.backbone)
        
        # Time Fusion (TF) modules are defined for both image types.
        ##################Image
        self.TF_list = nn.ModuleList([
                Time_Fusion_and_Enhance_Encoder(16),  
                Time_Fusion_and_Enhance_Encoder(24),
                Time_Fusion_and_Enhance_Encoder(40),
                Time_Fusion_and_Enhance_Encoder(80),
                Time_Fusion(80, 80),  
            ])
        ##################NeighborImage
        self.TF_list_Neighbor = nn.ModuleList([
                Time_Fusion_and_Enhance_Encoder(16),  
                Time_Fusion_and_Enhance_Encoder(24),
                Time_Fusion_and_Enhance_Encoder(40),
                Time_Fusion_and_Enhance_Encoder(80),
            ])
        # Center-Neighborhood Fusion (CNF) module.
        ##################
        self.CNF_list = nn.ModuleList([
                Center_Neighborhood_Fusion(16, 1),  
                Center_Neighborhood_Fusion(24, 1),
                Center_Neighborhood_Fusion(40, 1),
                Center_Neighborhood_Fusion(80, 1),
            ])
        
        # Multi-path deep supervised attention (MDSA) module
        # The MDSA module is responsible for enhancing the multi-scale features.
        self.MDSA = nn.ModuleList([          
                Up_and_Deep_Supervision_Attention(80, 40),
                Up_and_Deep_Supervision_Attention(40, 24),
                Up_and_Deep_Supervision_Attention(24, 16),
                Up_and_Deep_Supervision_Attention(16, 16),
            ])
        # The last layer generates its prediction.
        self.last_layer_prediction = Predicted_Map_for_Main_Path(16, 1)
        
        # Multi-scale aggregation (MSA) module 
        # The final reult is generated by the multi-scale predictions.
        self.outs_mixing = Multi_Scale_Aggregation(5, out_dim)


    # This function extracts features from both the center and neighborhood images at different time points.
    # The features are then fused using the TF and CNF modules to capture the relevant spatio-temporal information.
    def encoder(self, A, B, A_Neighbor, B_Neighbor):
        features = []
        for i, layer in enumerate(self.backbone):
            ############### A, B
            A, B = layer(A), layer(B)
            A, B, AB_mix = self.TF_list[i](A, B)
            ############### A_Neighbor, B_Neighbor
            A_Neighbor, B_Neighbor = self.backbone_Neighbor[i](A_Neighbor), self.backbone_Neighbor[i](B_Neighbor)
            A_Neighbor, B_Neighbor, AB_mix_Neighbor = self.TF_list_Neighbor[i](A_Neighbor, B_Neighbor)
            ############### A, B , A_Neighbor, B_Neighbor fusion
            AB_mix_attention = self.CNF_list[i](A, B, AB_mix, A_Neighbor, B_Neighbor, AB_mix_Neighbor)
            features.append(AB_mix_attention)
        AB_mix = self.TF_list[len(self.backbone)](A, B)
        features.append(AB_mix)
        return features

    # The decoder uses multi-scale features from the encoder and aggregates them using the MDSA module.
    def decoder(self, features):
        upping = features[-1]
        for i, j in enumerate(range(-2, -6, -1)):
            upping = self.MDSA[i](upping, features[j])
        return upping

    # The forward pass extracts features using the encoder, decodes them using MDSA, and generates the final prediction using the MSA module.
    # The output is processed through a softmax function
    def forward(self, A, B, A_Neighbor, B_Neighbor):
        features = self.encoder(A, B, A_Neighbor, B_Neighbor)
        last_layer = self.decoder(features)
        out = self.last_layer_prediction(last_layer)
        #
        out_fs = [out]
        out_fs = out_fs+features[:-1]
        output_feature = self.outs_mixing(out_fs)
        #
        output=F.softmax(output_feature, dim=1)
        return output






# Verification of Model Executability
if __name__ == "__main__":
    from thop import profile
    
    net = INSINet(in_dim=3, out_dim=2)
    img = torch.randn(1, 3, 256, 256)
    Neighbor_img= torch.randn(1, 3, 256, 256)
    flops, params = profile(net, ((img, img, Neighbor_img, Neighbor_img)))
    print(f"params = {params/1e6}M")
    print(f"MACs = {flops/1e9}G")
    out=net(img,img, img, img)
    print(out.shape)