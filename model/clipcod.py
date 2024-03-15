import torch
import torch.nn as nn
import torch.nn.functional as F

from model.clip import build_model
from utils.losses import structure_loss, kl_div_loss, correlation_coefficient_loss, cosine_similarity_loss
from .layers import FPN, Projector, TransformerDecoder, FixationEstimation, FeatureFusionModule, ProjectionNetwork, AttributePrediction, pool_visual_features, d3_to_d4


class CLIPCODBLANK(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Vision & Text Encoder
        clip_model = torch.jit.load(cfg.clip_pretrain,
                                    map_location="cpu").eval()
        self.backbone = build_model(clip_model.state_dict(), cfg.word_len, cfg.feats_layer_num).float()
        # Multi-Modal FPN
        self.neck = FPN(in_channels=cfg.fpn_in, out_channels=cfg.fpn_out)
        # Decoder
        self.decoder = TransformerDecoder(num_layers=cfg.num_layers,
                                          d_model=cfg.vis_dim,
                                          nhead=cfg.num_head,
                                          dim_ffn=cfg.dim_ffn,
                                          dropout=cfg.dropout,
                                          return_intermediate=cfg.intermediate)
        # Projector
        self.proj = Projector(cfg.word_dim, cfg.vis_dim , 3)
        

    def forward(self, img, word, mask=None):
        '''
            img: b, 3, h, w
            word: b, words
            word_mask: b, words
            mask: b, 1, h, w
        ''' 
        # padding mask used in decoder
        pad_mask = torch.zeros_like(word).masked_fill_(word == 0, 1).bool()

        # vis: list: 3 x [b, 576, 768]
        # word: b, 77, 1024
        # state: b, 1024
        vis = self.backbone.encode_image(img)           # list: 3 x [b, 576, 768]
        word, state = self.backbone.encode_text(word)   # [b, 77, 768] [b, 768]

        # b, c, 24, 24
        fq = self.neck(vis, state)
        b, c, h, w = fq.size()
        fq = self.decoder(fq, word, pad_mask)
        fq = fq.reshape(b, c, h, w)  # [b, c, 24, 24]

        pred = self.proj(fq, state) # [b, c, 96, 96]

        if self.training:
            # resize mask
            if pred.shape[-2:] != mask.shape[-2:]:
                mask = F.interpolate(mask, pred.shape[-2:],
                                     mode='nearest').detach()
            loss = structure_loss(pred, mask)
            return pred.detach(), mask, loss
        else:
            return pred.detach()


class CLIPCOD(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # para init
        self.kl_weight = cfg.kl_weight
        self.cc_weight = cfg.cc_weight
        self.consistency_weight = cfg.consistency_weight
        self.use_attr = cfg.use_attr
        self.save_fix_attr = cfg.save_fix_attr
        
        # Vision & Text Encoder
        clip_model = torch.jit.load(cfg.clip_pretrain, map_location="cpu").eval()
        self.backbone = build_model(clip_model.state_dict(), cfg.word_len, cfg.feats_layer_num).float()
        
        # Multi-Modal FPN
        # self.neck = FPN(in_channels=cfg.fpn_in, out_channels=cfg.fpn_out)
        
        # fixation prediction
        self.fix_encoder = FixationEstimation(cfg.fix_embed_dim, 
                                              cfg.fix_num_head,
                                              cfg.fix_num_layers,
                                              cfg.fix_dim_ffn,
                                              cfg.fix_out_size)
        self.attr_pred = AttributePrediction(num_tokens=576, feature_dim=768, num_attr=cfg.num_attr)
        # visual modal fusion
        self.visual_fusion = FeatureFusionModule(embed_dim=768, attr_dim=cfg.num_attr)

        # projector for consistency loss
        self.word_proj = ProjectionNetwork(input_dim=cfg.word_dim, proj_dim=cfg.projector_dim)
        self.vis_proj = ProjectionNetwork(input_dim=cfg.vis_dim, proj_dim=cfg.projector_dim)

        # multimodal decoder
        self.decoder = TransformerDecoder(num_layers=cfg.num_layers,
                                          d_model=cfg.vis_dim,
                                          nhead=cfg.num_head,
                                          dim_ffn=cfg.dim_ffn,
                                          dropout=cfg.dropout,
                                          return_intermediate=cfg.intermediate)
        # Projector
        self.proj = Projector(cfg.word_dim, cfg.vis_dim, 3, cfg.num_attr)
         
    def forward(self, img, img_gt, overall_desc=None, camo_desc=None, attr=None, fix_gt=None):
        '''
            img: b, 3, h, w
            desc: b, worddim, words
            state: b, words
            img_gt: b, 1, h, w
            fix_gt: b, 1, h, w
        '''
        # padding mask used in decoder
        # pad_mask = torch.zeros_like(overall_desc).masked_fill_(overall_desc == 0, 1).bool()
    
        if self.training:
            # vis: list: 3 x [b, 576, 768]
            # word: b, 77, 1024
            # state: b, 1024
            vis = self.backbone.encode_image(img)           # list: 3 x [b, 576, 768]
            _, overall_state = self.backbone.encode_text(overall_desc)   # [b, 77, 768] [b, 768]
            # camo_w, camo_state = self.backbone.encode_text(camo_desc)   # [b, 77, 768] [b, 768]

            # vis branch
            # attr prediction
            attr_out = self.attr_pred(vis)
            # fix prediction
            fix_out, fix_tensor = self.fix_encoder(vis)  # [b, 1, 96, 96]  [b, 576, 768]
            vis_feats = self.visual_fusion(vis, fix_tensor, attr_out) # [b, 576, 768]
            # for consistency loss
            vis_proj = pool_visual_features(vis_feats, pooling_type='max') # [b, 576, 768] -> [b, 768]
            vis_proj = self.vis_proj(vis_proj) # [b, 768] -> [b, 512]
            word_proj = self.word_proj(overall_state)   # [b, 768] -> [b, 512]

            # multimodal branch 
            multimodal_feats = d3_to_d4(self, vis_feats)
            b, c, h, w = multimodal_feats.size() # [b, out_channels[1], 24, 24]
            multimodal_feats = self.decoder(multimodal_feats)  # desc should change to while img description
            multimodal_feats = multimodal_feats.reshape(b, c, h, w)  # [b, c, 24, 24]

            
            pred = self.proj(multimodal_feats, attr_out, self.use_attr) # [b, c, 96, 96]
            
            # resize mask
            if pred.shape[-2:] != img_gt.shape[-2:]:
                img_gt = F.interpolate(img_gt, pred.shape[-2:], mode='nearest').detach()
                fix_gt = F.interpolate(fix_gt, fix_out.shape[-2:], mode='nearest').detach()

            mask_loss = structure_loss(pred, img_gt)
            kl_loss = kl_div_loss(fix_out, fix_gt) * self.kl_weight
            cc_loss = correlation_coefficient_loss(fix_out, fix_gt) * self.cc_weight
            fix_loss = kl_loss + cc_loss
            attr_loss = nn.MSELoss()(attr_out, attr)
            consistency_loss = cosine_similarity_loss(vis_proj, word_proj) * self.consistency_weight
            total_loss = mask_loss + fix_loss + consistency_loss + attr_loss
            return pred.detach(), fix_out.detach(), total_loss, fix_loss, kl_loss, cc_loss, mask_loss, consistency_loss, attr_loss
        else:
            # vis: list: 3 x [b, 576, 768]
            # word: b, 77, 1024
            # state: b, 1024
            vis = self.backbone.encode_image(img)           # list: 3 x [b, 576, 768]
            
            # vis branch only
            # attr prediction
            attr_out = self.attr_pred(vis)
            # fix prediction
            fix_out, fix_tensor = self.fix_encoder(vis)  # [b, 1, 96, 96]  [b, 576, 768]
            vis_feats = self.visual_fusion(vis, fix_tensor, attr_out) # [b, 576, 768]

            # multimodal branch 
            multimodal_feats = d3_to_d4(self, vis_feats)
            b, c, h, w = multimodal_feats.size() # [b, out_channels[1], 24, 24]
            multimodal_feats = self.decoder(multimodal_feats)  # desc should change to while img description
            multimodal_feats = multimodal_feats.reshape(b, c, h, w)  # [b, c, 24, 24]

            
            pred = self.proj(multimodal_feats, attr_out, self.use_attr) # [b, c, 96, 96]

            if self.save_fix_attr:
                return pred.detach(), fix_out.detach(), attr_out.detach()
            else:
                return pred.detach()
