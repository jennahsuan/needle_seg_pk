from ..SAM2.sam2.modeling.memory_encoder import *
from ..SAM2.sam2.modeling.memory_attention import *
from ..SAM2.sam2.modeling.position_encoding import *
from ..SAM2.sam2.modeling.sam.transformer import *


def build_sam2_memory_encoder(mem_dim = 64, image_size = 384):
    mask_downsampler = MaskDownSampler(kernel_size=3,  ## sam2.1/sam2.1_hiera_b+.yaml
                                    stride=2,
                                    padding=1,
                                    embed_dim=256)  ## input channel = 1
    fuser = Fuser(layer=CXBlock(
                        dim= 256,
                        kernel_size= 7,
                        padding= 3,
                        layer_scale_init_value= 1e-6,
                        use_dwconv= True,  # depth-wise convs
                        ),
                    num_layers= 2)
    position_encoding = PositionEmbeddingSine(num_pos_feats= mem_dim,
                                            normalize= True,
                                            scale= None,
                                            temperature= 10000,
                                            image_size=image_size)
    return MemoryEncoder(out_dim=mem_dim,
                        mask_downsampler=mask_downsampler,
                        fuser=fuser,
                        position_encoding=position_encoding,
                        in_dim=256, ## 192,96,48,24
                        )  

def build_sam2_memory_attention(kv_in_dim = 64, local_branch=False, spatial_local_CA=False):
    self_attention = RoPEAttention(
       rope_theta=10000.0,
        feat_sizes=[384//16, 384//16],
        embedding_dim=256,
        num_heads=1,
        downsample_rate=1,
        dropout=0.1,
    )
    cross_attention = RoPEAttention(
        rope_theta=10000.0,
        feat_sizes=[384//16, 384//16],
        rope_k_repeat=True,
        embedding_dim=256,
        num_heads=1,
        downsample_rate=1,
        dropout=0.1,
        kv_in_dim=kv_in_dim,
    )
    if spatial_local_CA:
        local_cross_attention = LocalRoPEAttention(  ## 3/28
            rope_theta=10000.0,
            feat_sizes=[384//16, 384//16],
            rope_k_repeat=True,
            embedding_dim=256,
            num_heads=1,  ## 8
            downsample_rate=1,
            dropout=0.1,
            kv_in_dim=kv_in_dim,
            max_dis=7,
            dilation=1,
            enable_corr=False,
        )
        print('[spatial_local_CA]')
    else:
        local_cross_attention = None
    layer = MemoryAttentionLayer(
        activation="relu", ### default relu 
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=False,
        self_attention=self_attention,
        d_model=256,
        pos_enc_at_cross_attn_keys=True,
        pos_enc_at_cross_attn_queries=False,
        cross_attention=cross_attention,
        local_branch = local_branch,
        local_cross_attention = local_cross_attention
    )
    memory_atttention = MemoryAttention(
        d_model=256,
        pos_enc_at_input=True,
        layer=layer,
        num_layers = 2,
    )
    return memory_atttention

def build_sam2_pos_encoding(num_pos_feats= 64, image_size = 384):
    position_encoding = PositionEmbeddingSine(num_pos_feats= num_pos_feats,
                                            normalize= True,
                                            scale= None,
                                            temperature= 10000,
                                            image_size=image_size)
    return position_encoding