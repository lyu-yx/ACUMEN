import math

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch_geometric.nn import GCNConv, global_mean_pool
# from torch_geometric.data import Data

def conv_layer(in_dim, out_dim, kernel_size=1, padding=0, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_dim), nn.ReLU(True))


def linear_layer(in_dim, out_dim, bias=False):
    return nn.Sequential(nn.Linear(in_dim, out_dim, bias),
                         nn.BatchNorm1d(out_dim), nn.ReLU(True))

def pool_visual_features(visual_features, pooling_type='max'):
    """
    Pool the 3D visual features to 2D.
    visual_features: Tensor of shape [b, 576, 768]
    pooling_type: 'max' or 'avg'
    """
    if pooling_type == 'max':
        pooled, _ = torch.max(visual_features, dim=1)
    elif pooling_type == 'avg':
        pooled = torch.mean(visual_features, dim=1)
    else:
        raise ValueError("Unsupported pooling type. Choose 'max' or 'avg'.")
    return pooled

def convert_to_logits(tensor, epsilon=1e-6):
    # Convert to logits
    tensor = torch.clamp(tensor, epsilon, 1 - epsilon)
    logits = torch.log(tensor / (1 - tensor))

    return logits


class CoordConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 padding=1,
                 stride=1):
        super().__init__()
        self.conv1 = conv_layer(in_channels + 2, out_channels, kernel_size,
                                padding, stride)

    def add_coord(self, input):
        b, _, h, w = input.size()
        x_range = torch.linspace(-1, 1, w, device=input.device)
        y_range = torch.linspace(-1, 1, h, device=input.device)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([b, 1, -1, -1])
        x = x.expand([b, 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)
        input = torch.cat([input, coord_feat], 1)
        return input

    def forward(self, x):
        x = self.add_coord(x)
        x = self.conv1(x)
        return x


class DimensionalReduction(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DimensionalReduction, self).__init__()
        self.reduce = nn.Sequential(
            conv_layer(in_channel, out_channel, 3, padding=1),
            conv_layer(out_channel, out_channel, 3, padding=1)
        )

    def forward(self, x):
        return self.reduce(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_heads, num_features=3):
        super(CrossAttentionFusion, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_features)])  # List of LayerNorms

    def forward(self, features):
        normalized_features = [norm(feature) for norm, feature in zip(self.norms, features)]
        kv = torch.cat(normalized_features, dim=1)  # Shape: [b, 1728, 768]
        query = normalized_features[2]  # Shape: [b, 576, 768]
        attn_output, _ = self.attention(query=query, key=kv, value=kv)
        output = attn_output + query  # Shape: [b, 576, 768]
        return output


class FixationEstimation(nn.Module):
    def __init__(self, embed_dim, num_heads, num_decoder_layers, dim_feedforward, output_map_size):
        super(FixationEstimation, self).__init__()
        self.pos_encoder = PositionalEncoding(embed_dim)
        self.fusion = CrossAttentionFusion(embed_dim, num_heads)
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=dim_feedforward)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.intermediate_linear = nn.Linear(embed_dim, output_map_size * output_map_size)
        self.aggregate_conv = nn.Conv2d(in_channels=576, out_channels=1, kernel_size=1)
        self.tensor_out_conv = nn.Conv1d(in_channels=576, out_channels=768, kernel_size=1)
        self.reshape_size = output_map_size
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear')
        
        # Learnable memory initialization
        self.learnable_memory = nn.Parameter(torch.randn(1, embed_dim), requires_grad=True)

    def forward(self, feature_list):
        fused_features = self.fusion(feature_list)
        fused_features = fused_features.permute(1, 0, 2)  # Shape: [sequence_length, batch_size, feature_size]
        fused_features = self.pos_encoder(fused_features)  # Add positional encoding

        # self-attention mode
        out_fix = self.transformer_decoder(fused_features, fused_features)

        # Reshape and project the output to the desired fixation map size
        out_fix = self.intermediate_linear(out_fix)
        out_fix = out_fix.view(-1, 576, self.reshape_size, self.reshape_size)  # Shape: [b, 576, 24, 24]
        out_tensor = out_fix.view(-1, 576, self.reshape_size * self.reshape_size)
        out_tensor = self.tensor_out_conv(out_tensor).transpose(1, 2)  # Shape: [b, 576, 768]
        # # Adding skip connection
        out_fix = self.aggregate_conv(out_fix)  # Shape: [b, 1, 24, 24]
        out_fix = self.upsample(out_fix)  # Shape: [b, 1, 96, 96]

        return out_fix, out_tensor


class AttributePrediction(nn.Module):
    def __init__(self, num_tokens, feature_dim, num_attr, deeper_layer_weight=0.5):
        super(AttributePrediction, self).__init__()
        self.deeper_layer_weight = deeper_layer_weight
        self.middle_layer_weight = (1 - deeper_layer_weight) / 2
        self.shallower_layer_weight = (1 - deeper_layer_weight) / 2

        # Initial dimension reduction
        self.init_reduce_dim = nn.Linear(feature_dim, 256)

        # Further dimension reduction
        self.reduce_dim = nn.Linear(num_tokens * 256 * 3, 512)

        # Fully connected layers
        self.fc1 = nn.Linear(512, 256)
        self.batch_norm1 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_attr)

    def forward(self, tensors):
        # Apply weights and initial reduction to each tensor
        reduced_tensors = [self.init_reduce_dim(t * weight) for t, weight in zip(tensors, 
                        [self.shallower_layer_weight, self.middle_layer_weight, self.deeper_layer_weight])]

        # Concatenate the tensors along the feature dimension
        concatenated = torch.cat(reduced_tensors, dim=-1)  # Size: [b, 576, 256*3]
        
        # Further reduce the dimensionality of the concatenated tensor
        reduced = self.reduce_dim(concatenated.view(concatenated.size(0), -1))

        # Apply linear layers and other components
        attr_ctrb = self.fc1(reduced)
        attr_ctrb = self.batch_norm1(attr_ctrb)
        attr_ctrb = self.relu(attr_ctrb)
        attr_ctrb = self.dropout(attr_ctrb)
        attr_ctrb = self.fc2(attr_ctrb)

        return attr_ctrb



# class FeatureTransform(nn.Module):
#     def __init__(self, embed_dim, attr_dim):
#         super(FeatureTransform, self).__init__()
#         self.transform = nn.Linear(embed_dim, embed_dim)
#         self.attr_transform = nn.Linear(attr_dim, embed_dim)
#         self.gate = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.Sigmoid())

#     def forward(self, feature, image_attr):
#         # Transform feature
#         trans_feature = self.transform(feature)

#         # Transform image attribution and apply gating
#         attr_feature = self.attr_transform(image_attr).unsqueeze(1)
#         gate = self.gate(attr_feature)

#         # Apply gating with residual connection
#         gated_feature = trans_feature * gate + feature
#         return gated_feature


class FeatureTransform(nn.Module):
    def __init__(self, embed_dim, attr_dim):
        super(FeatureTransform, self).__init__()
        self.transform = nn.Linear(embed_dim, embed_dim)
        self.attr_transform = nn.Linear(attr_dim, embed_dim * 576)
        self.squeeze = nn.AdaptiveAvgPool1d(1)  
        self.excitation = nn.Sequential(
            nn.Linear(576, 576 // 16),
            nn.ReLU(),
            nn.Linear(576 // 16, 576),
            nn.Sigmoid()
        )

    def forward(self, feature, image_attr):
        # Transform feature
        trans_feature = self.transform(feature)  # b, 576, 768
        b, c, _ = trans_feature.size()
        # Transform image attribution and apply gating
        attr_feature = self.attr_transform(image_attr)  # b, 576 * 768
        attr_feature = attr_feature.view(b, c, -1)  # b, 576, 768

        y = self.squeeze(attr_feature).view(b, c)  # b, 576
        y = self.excitation(y).view(b, c, 1)  #  b, 576, 1
        gated_feature = trans_feature * y.expand_as(trans_feature) + feature

        return gated_feature

class FeatureFusionModule(nn.Module):
    def __init__(self, embed_dim, attr_dim):
        super(FeatureFusionModule, self).__init__()
        self.embed_dim = embed_dim
        self.shallow_transform = FeatureTransform(embed_dim, attr_dim)
        self.mid_transform = FeatureTransform(embed_dim, attr_dim)
        self.deep_transform = FeatureTransform(embed_dim, attr_dim)
        self.attention_gen = nn.Linear(embed_dim, 3)

    def forward(self, feature_list, fixation_pred, image_attr):
        # Apply transformations
        gated_shallow = self.shallow_transform(feature_list[0], image_attr)
        gated_mid = self.mid_transform(feature_list[1], image_attr)
        gated_deep = self.deep_transform(feature_list[2], image_attr)

        # Generate attention weights
        attention_weights = F.softmax(self.attention_gen(fixation_pred), dim=-1)

        # Apply attention weights
        weighted_shallow = gated_shallow * attention_weights[:,:,0].unsqueeze(-1) + gated_shallow
        weighted_mid = gated_mid * attention_weights[:,:,1].unsqueeze(-1) + gated_mid
        weighted_deep = gated_deep * attention_weights[:,:,2].unsqueeze(-1) +  gated_deep

        # Aggregate features
        aggregated_feature = (weighted_shallow + 2 * weighted_mid + 4 * weighted_deep) / 7
        normalized_feature = F.layer_norm(aggregated_feature, [576, self.embed_dim])

        return normalized_feature


class ProjectionNetwork(nn.Module):
    def __init__(self, input_dim, proj_dim, hidden_dim=None):
        super(ProjectionNetwork, self).__init__()
        if hidden_dim is None:
            hidden_dim = (input_dim + proj_dim) // 2  # A heuristic for hidden dimension size

        # Define the layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_dim, proj_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class ConvBR(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1):
        super(ConvBR, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class Projector(nn.Module):
    def __init__(self, word_dim=768, in_dim=768, kernel_size=3, num_attr=17):
        super().__init__()
        self.in_dim = in_dim
        self.kernel_size = kernel_size
        # visual projector
        self.vis = nn.Sequential(  # spatical resolution times 4
            nn.Upsample(scale_factor=2, mode='bilinear'),
            conv_layer(in_dim , in_dim * 2, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            conv_layer(in_dim * 2, in_dim, 3, padding=1),
            nn.Conv2d(in_dim, in_dim, 1))
        
        # textual projector
        self.attr_proj = nn.Linear(num_attr, word_dim)
        out_dim = 1 * word_dim * kernel_size * kernel_size + 1
        self.txt = nn.Linear(word_dim, out_dim)

        # output projector
        self.out_proj = nn.Sequential(
            ConvBR(in_dim, 128, 3, padding=1),
            ConvBR(128, 64, 3, padding=1),
            nn.Conv2d(64, 1, 1))

    def forward(self, x, attr, use_attr=False):
        '''
            x: b, vis_dim, 24, 24
        '''
        if use_attr:
            x = self.vis(x) # b, 768, 96, 96
            B, C, H, W = x.size()
            x = x.reshape(1, B * C, H, W) # 1, b*768, 96, 96
            # txt: b, (768*3*3 + 1) -> b, 768, 3, 3 / b 
            attr = self.attr_proj(attr)
            attr = self.txt(attr)
            weight, bias = attr[:, :-1], attr[:, -1]
            weight = weight.reshape(B, C, self.kernel_size, self.kernel_size)
            # Conv2d - 1, b*768, 96, 96 -> 1, b, 96, 96
            out = F.conv2d(x,
                        weight,
                        padding=self.kernel_size // 2,
                        groups=weight.size(0),
                        bias=bias)
            out = out.transpose(0, 1)   # b, 1, 96, 96
            return out
        else:
            x = self.vis(x) # b, 768, 96, 96
            B, C, H, W = x.size()
            # b, 768, 96, 96 -> b, 1, 96, 96
            out = self.out_proj(x)
            # b, 1, 96, 96
            return out
            


class TransformerDecoder(nn.Module):
    def __init__(self,
                 num_layers,
                 d_model,
                 nhead,
                 dim_ffn,
                 dropout,
                 return_intermediate=False):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model=d_model,
                                    nhead=nhead,
                                    dim_feedforward=dim_ffn,
                                    dropout=dropout) for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.return_intermediate = return_intermediate

    @staticmethod
    def pos1d(d_model, length):
        """
        :param d_model: dimension of the model
        :param length: length of positions
        :return: length*d_model position matrix
        """
        if d_model % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(d_model))
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                              -(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)

        return pe.unsqueeze(1)  # n, 1, 512

    @staticmethod
    def pos2d(d_model, height, width):
        """
        :param d_model: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        :return: d_model*height*width position matrix
        """
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dimension (got dim={:d})".format(d_model))
        pe = torch.zeros(d_model, height, width)
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(
            torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(
            0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(
            0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(
            0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(
            0, 1).unsqueeze(2).repeat(1, 1, width)

        return pe.reshape(-1, 1, height * width).permute(2, 1, 0)  # hw, 1, 512

    def forward(self, vis):
        '''
            vis: b, 512, h, w
            txt: b, L, 512
            
        '''
        B, C, H, W = vis.size()
        # position encoding
        vis_pos = self.pos2d(C, H, W)
        # reshape & permute
        vis = vis.reshape(B, C, -1).permute(2, 0, 1)
        # forward
        output = vis
        intermediate = []
        for layer in self.layers:
            output = layer(output, vis_pos)
            if self.return_intermediate:
                # HW, b, 512 -> b, 512, HW
                intermediate.append(self.norm(output).permute(1, 2, 0))

        if self.norm is not None:
            # HW, b, 512 -> b, 512, HW
            output = self.norm(output).permute(1, 2, 0)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
                # [output1, output2, ..., output_n]
                return intermediate
            else:
                # b, 512, HW
                return output
        return output

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        # Normalization Layer
        self.self_attn_norm = nn.LayerNorm(d_model)
        # Attention Layer
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # FFN
        self.ffn = nn.Sequential(nn.Linear(d_model, dim_feedforward),
                                 nn.ReLU(True), nn.Dropout(dropout),
                                 nn.Linear(dim_feedforward, d_model))
        # LayerNorm & Dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos.to(tensor.device)

    def forward(self, vis, vis_pos):
        '''
            vis: 24*24, b, 512
            vis_pos: 24*24, 1, 512
        '''
        # Self-Attention
        vis2 = self.norm1(vis)
        q = k = self.with_pos_embed(vis2, vis_pos)
        vis2 = self.self_attn(q, k, value=vis2)[0]
        vis2 = self.self_attn_norm(vis2)
        vis = vis + self.dropout1(vis2)
        # FFN
        vis2 = self.norm2(vis)
        vis2 = self.ffn(vis2)
        vis = vis + self.dropout2(vis2)
        return vis


def d3_to_d4(self, t):
        n, hw, c = t.size()
        if hw % 2 != 0:
            t = t[:, 1:]
        h = w = int(math.sqrt(hw))
        return t.transpose(1, 2).reshape(n, c, h, w)

class FPN(nn.Module):
    def __init__(self,
                 in_channels=[768, 768, 768],
                 out_channels=[256, 512, 1024]):
        super(FPN, self).__init__()
        # text projection [b, 768] --> [b, 1024]
        self.txt_proj = linear_layer(in_channels[2], out_channels[2]) # linear + batch norm + relu

        # fusion 1: v5 & seq -> f_5: b, 1024, 24, 24
        self.f1_v_proj = conv_layer(in_channels[2], out_channels[2], 1, 0)  # CBR
        self.norm_layer = nn.Sequential(nn.BatchNorm2d(out_channels[2]),
                                        nn.ReLU(True))
        # fusion 2: v4 & fm -> f_4: b, 512, 24, 24
        self.f2_v_proj = conv_layer(in_channels[1], out_channels[1], 3, 1)
        self.f2_cat = conv_layer(out_channels[2] + out_channels[1],
                                 out_channels[1], 1, 0)
        # fusion 3: v3 & fm_mid -> f_3: b, 512, 24, 24
        self.f3_v_proj = conv_layer(in_channels[0], out_channels[0], 3, 1)
        self.f3_cat = conv_layer(out_channels[0] + out_channels[1],
                                 out_channels[1], 1, 0)
        # fusion 4: f_3 & f_4 & f_5 -> fq: b, 256, 24, 24
        self.f4_proj5 = conv_layer(out_channels[2], out_channels[1], 3, 1)
        self.f4_proj4 = conv_layer(out_channels[1], out_channels[1], 3, 1)
        self.f4_proj3 = conv_layer(out_channels[1], out_channels[1], 3, 1)
        # aggregation
        self.aggr = conv_layer(3 * out_channels[1], out_channels[1], 1, 0)
        self.coordconv = nn.Sequential(
            CoordConv(out_channels[1], out_channels[1], 3, 1),
            conv_layer(out_channels[1], out_channels[1], 3, 1))

    def forward(self, imgs, state):
        # imgs: 3 x [b, 576, 768] state: [b, 768]
        v3, v4, v5 = imgs
        v3 = d3_to_d4(self, v3) # [b, 768, 24, 24]
        v4 = d3_to_d4(self, v4)
        v5 = d3_to_d4(self, v5) 
        # fusion 1: b, 768, 24, 24
        # text projection: b, 768 -> b, 1024
        # out: b, 1024, 24, 24
        state = self.txt_proj(state).unsqueeze(-1).unsqueeze(-1)  # b, 1024, 1, 1
        f5 = self.f1_v_proj(v5)
        f5 = self.norm_layer(f5 * state)
        # fusion 2: b, 768, 24, 24
        # out: b, 512, 24, 24
        f4 = self.f2_v_proj(v4)
        f4 = self.f2_cat(torch.cat([f4, f5], dim=1))
        # fusion 3: b, 768, 24, 24
        # out: b, 256, 24, 24
        f3 = self.f3_v_proj(v3)
        f3 = self.f3_cat(torch.cat([f3, f4], dim=1))
        # fusion 4: 3 * [b, 768, 24, 24]
        fq5 = self.f4_proj5(f5)
        fq4 = self.f4_proj4(f4)
        fq3 = self.f4_proj3(f3)
        # query
        
        fq = torch.cat([fq3, fq4, fq5], dim=1)
        fq = self.aggr(fq)
        fq = self.coordconv(fq)
        # b, out_channels[1], 24, 24
        return fq
