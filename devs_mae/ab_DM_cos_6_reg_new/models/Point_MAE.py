import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.layers import DropPath, trunc_normal_
import numpy as np
from .build import MODELS
from utils import misc
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *
import random
from knn_cuda import KNN
# from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from models.transformers import TransformerEncoder, TransformerDecoder, Encoder, Group
from pytorch3d.loss import chamfer_distance

# Pretrain model
class MaskTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        # define the transformer argparse
        self.mask_ratio = config.transformer_config.mask_ratio 
        self.trans_dim = config.transformer_config.trans_dim
        self.depth = config.transformer_config.depth 
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.num_heads = config.transformer_config.num_heads 
        print_log(f'[args] {config.transformer_config}', logger = 'Transformer')

        self.cls_token1 = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos1 = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_token2 = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos2 = nn.Parameter(torch.zeros(1, 1, self.trans_dim))

        # embedding
        self.encoder_dims =  config.transformer_config.encoder_dims
        self.encoder = Encoder(encoder_channel = self.encoder_dims)
        self.mask_type = config.transformer_config.mask_type

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim = self.trans_dim,
            depth = self.depth,
            drop_path_rate = dpr,
            num_heads = self.num_heads,
        )

        self.norm = nn.LayerNorm(self.trans_dim)
        self.apply(self._init_weights)
        trunc_normal_(self.cls_token1, std=.02)
        trunc_normal_(self.cls_pos1, std=.02)
        trunc_normal_(self.cls_token2, std=.02)
        trunc_normal_(self.cls_pos2, std=.02)

        self.proj_dim = 256
        self.proj_cls = nn.Sequential(
            nn.Linear(self.trans_dim, self.proj_dim),
            nn.BatchNorm1d(self.proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.proj_dim, self.proj_dim)
        )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _mask_center_block(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()
        # mask a continuous part
        mask_idx = []
        for points in center:
            # G 3
            points = points.unsqueeze(0)  # 1 G 3
            index = random.randint(0, points.size(1) - 1)
            distance_matrix = torch.norm(points[:, index].reshape(1, 1, 3) - points, p=2,
                                         dim=-1)  # 1 1 3 - 1 G 3 -> 1 G

            idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0]  # G
            ratio = self.mask_ratio
            mask_num = int(ratio * len(idx))
            mask = torch.zeros(len(idx))
            mask[idx[:mask_num]] = 1
            mask_idx.append(mask.bool())

        bool_masked_pos = torch.stack(mask_idx).to(center.device)  # B G

        return bool_masked_pos

    def _mask_center_rand(self, center, noaug = False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        B, G, _ = center.shape
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()

        self.num_mask = int(self.mask_ratio * G)

        overall_mask = np.zeros([B, G])
        for i in range(B):
            mask = np.hstack([
                np.zeros(G-self.num_mask),
                np.ones(self.num_mask),
            ])
            np.random.shuffle(mask)
            overall_mask[i, :] = mask
        overall_mask = torch.from_numpy(overall_mask).to(torch.bool)

        return overall_mask.to(center.device) # B G

    def forward(self, neighborhood, center, noaug = False):
        # generate mask
        if self.mask_type == 'rand':
            bool_masked_pos1 = self._mask_center_rand(center, noaug = noaug) # B G
            bool_masked_pos2 = self._mask_center_rand(center, noaug = noaug) # B G
        else:
            bool_masked_pos = self._mask_center_block(center, noaug = noaug)

        group_input_tokens = self.encoder(neighborhood)  #  B G C
        batch_size, seq_len, C = group_input_tokens.size()

        # Get idx for bool_mask1 & mask2
        mask_idx1, vis_idx1 = torch.where(bool_masked_pos1), torch.where(~bool_masked_pos1)
        x_vis1 = group_input_tokens[~bool_masked_pos1].reshape(batch_size, -1, C)

        mask_idx2, vis_idx2 = torch.where(bool_masked_pos2), torch.where(~bool_masked_pos2)
        x_vis2 = group_input_tokens[~bool_masked_pos2].reshape(batch_size, -1, C)

        # add pos embedding
        # mask pos center #Actually, should be visible center!!!
        masked_center1 = center[~bool_masked_pos1].reshape(batch_size, -1, 3)
        masked_center2 = center[~bool_masked_pos2].reshape(batch_size, -1, 3)

        pos1 = self.pos_embed(masked_center1)
        pos2 = self.pos_embed(masked_center2)

        cls_token1 = self.cls_token1.expand(group_input_tokens.size(0), -1, -1)
        cls_pos1 = self.cls_pos1.expand(group_input_tokens.size(0), -1, -1)
        cls_token2 = self.cls_token2.expand(group_input_tokens.size(0), -1, -1)
        cls_pos2 = self.cls_pos2.expand(group_input_tokens.size(0), -1, -1)

        x1 = torch.cat((cls_token1, x_vis1), dim=1)
        pos1 = torch.cat((cls_pos1, pos1), dim=1)

        x2 = torch.cat((cls_token2, x_vis2), dim=1)
        pos2 = torch.cat((cls_pos2, pos2), dim=1)

        # transformer for the 
        x1 = self.blocks(x1, pos1)
        x1 = self.norm(x1)

        x2 = self.blocks(x2, pos2)
        x2 = self.norm(x2)

        proj_cls_x1 = self.proj_cls(x1[:, 0]) # [bs, 384]
        proj_cls_x2 = self.proj_cls(x2[:, 0])

        return proj_cls_x1, x1[:, 1:], bool_masked_pos1, mask_idx1, vis_idx1, proj_cls_x2, x2[:, 1:], bool_masked_pos2, mask_idx2, vis_idx2


@MODELS.register_module()
class Point_MAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[Point_MAE] ', logger ='Point_MAE')
        self.config = config
        self.trans_dim = config.transformer_config.trans_dim
        self.MAE_encoder = MaskTransformer(config)

        self.group_size = config.group_size
        self.num_group = config.num_group
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.mask_token1 = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.mask_token2 = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.decoder_pos_embed1 = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )
        self.decoder_pos_embed2 = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        self.decoder_depth = config.transformer_config.decoder_depth
        self.decoder_num_heads = config.transformer_config.decoder_num_heads
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]
        self.MAE_decoder = TransformerDecoder(
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            num_heads=self.decoder_num_heads,
        )
        self.MAE_decoder_ = TransformerDecoder(
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            num_heads=self.decoder_num_heads,
        )
        print_log(f'[Point_MAE] divide point cloud into G{self.num_group} x S{self.group_size} points ...', logger ='Point_MAE')
        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)

        # prediction head
        self.increase_dim = nn.Sequential(
            nn.Conv1d(self.trans_dim, 3*self.group_size, 1)
        )
        self.increase_dim_ = nn.Sequential(
            nn.Conv1d(self.trans_dim, 3*self.group_size, 1)
        )

        trunc_normal_(self.mask_token1, std=.02)
        trunc_normal_(self.mask_token2, std=.02)
    #     self.loss = config.loss
    #     # loss
    #     self.build_loss_func(self.loss)
        
    # def build_loss_func(self, loss_type):
    #     if loss_type == "cdl1":
    #         self.loss_func = ChamferDistanceL1().cuda()
    #     elif loss_type =='cdl2':
    #         self.loss_func = ChamferDistanceL2().cuda()
    #     else:
    #         raise NotImplementedError
    #         # self.loss_func = emd().cuda()

    def forward(self, pts, vis = False, **kwargs):
        neighborhood, center = self.group_divider(pts) # [bs, G, M, 3], [bs, G, 3]

        proj_cls_x1, x_vis1, mask1, mask_idx1, vis_idx1,\
        proj_cls_x2, x_vis2, mask2, mask_idx2, vis_idx2 = self.MAE_encoder(neighborhood, center)

        # Combine Un-Masked Feats & the newly initialized Masked token for Mask1
        B, N_vis, C = x_vis1.shape  # B VIS C
        pos_emd_vis1 = self.decoder_pos_embed1(center[~mask1]).reshape(B, -1, C)
        pos_emd_mask1 = self.decoder_pos_embed1(center[mask1]).reshape(B, -1, C)
        _, N1, _ = pos_emd_mask1.shape
        mask_token1 = self.mask_token1.expand(B, N1, -1)
        full_x1 = torch.cat([x_vis1, mask_token1], dim=1)
        pos_full_x1 = torch.cat([pos_emd_vis1, pos_emd_mask1], dim=1)

        # Combine Un-Masked Feats & the newly initialized Masked token for Mask2
        pos_emd_vis2 = self.decoder_pos_embed2(center[~mask2]).reshape(B, -1, C)
        pos_emd_mask2 = self.decoder_pos_embed2(center[mask2]).reshape(B, -1, C)
        _, N2, _ = pos_emd_mask2.shape
        mask_token2 = self.mask_token2.expand(B, N1, -1)
        full_x2 = torch.cat([x_vis2, mask_token2], dim=1)
        pos_full_x2 = torch.cat([pos_emd_vis2, pos_emd_mask2], dim=1)

        # [bs, M1(38), 384], [128, 64, 384]
        x_rec1, de_feats1 = self.MAE_decoder(full_x1, pos_full_x1, N1)
        x_rec2, de_feats2 = self.MAE_decoder_(full_x2, pos_full_x2, N2)
        B, M1, C = x_rec1.shape
        _, M2, _ = x_rec2.shape

        # *** Reconstrction Loss1 ***
        rebuild_points1 = self.increase_dim(x_rec1.transpose(1, 2)).transpose(1, 2).reshape(B * M1, -1, 3)  # B M 1024
        gt_points1 = neighborhood[mask1].reshape(B * M1,-1,3) 
        # loss_recon1 = self.loss_func(rebuild_points1, gt_points1)
        loss_recon1 = chamfer_distance(rebuild_points1, gt_points1, norm=2)[0]

        # *** Reconstrction Loss2 ***
        rebuild_points2 = self.increase_dim_(x_rec2.transpose(1, 2)).transpose(1, 2).reshape(B * M2, -1, 3)  # B M 1024
        gt_points2 = neighborhood[mask2].reshape(B * M2,-1,3) 
        # loss_recon2 = self.loss_func(rebuild_points2, gt_points2)
        loss_recon2 = chamfer_distance(rebuild_points2, gt_points2, norm=2)[0]

        loss_recon = loss_recon1 + loss_recon2

        # *** Contrastive Loss (Cross) ***
        # --- 1.Put the feats as the original position for mask1 branch
        de_vis1, de_mask1 = de_feats1[:, 0:N_vis], de_feats1[:, -N1:]
        de_feats_out1 = torch.zeros_like(full_x1)
        de_feats_out1[vis_idx1[0], vis_idx1[1], :] = de_vis1.reshape(-1, 384)
        de_feats_out1[mask_idx1[0], mask_idx1[1], :] = de_mask1.reshape(-1, 384)

        # --- 2.Put the feats as the original position for mask2 branch
        de_vis2, de_mask2 = de_feats2[:, 0:N_vis], de_feats2[:, -N2:]
        de_feats_out2 = torch.zeros_like(full_x2)
        de_feats_out2[vis_idx2[0], vis_idx2[1], :] = de_vis2.reshape(-1, 384)
        de_feats_out2[mask_idx2[0], mask_idx2[1], :] = de_mask2.reshape(-1, 384)

        # --- 3. Losses ---
        de_feats_out1 = F.normalize(de_feats_out1, dim=-1)
        de_feats_out2 = F.normalize(de_feats_out2, dim=-1)
        comask = mask1&mask2 # [128, 64]
        epsilon = 1e-8
        cos_sim = torch.bmm(de_feats_out1, de_feats_out2.transpose(1, 2))
        comask = comask.unsqueeze(-1).expand(-1,-1, 64)
        loss_contras = torch.sum(
           comask*(1 - cos_sim + epsilon)
        )/comask.sum()
        reg = torch.norm(de_feats_out1, p=2) + torch.norm(de_feats_out1, p=2)
        loss_contras = loss_contras + 0.001 * reg

        if vis: #visualization
            # For rebuild_points1
            mask = mask1 # or mask2
            vis_points = neighborhood[~mask].reshape(B * (self.num_group - M1), -1, 3)
            full_vis = vis_points + center[~mask].unsqueeze(1)
            full_rebuild = rebuild_points1 + center[mask].unsqueeze(1)
            full = torch.cat([full_vis, full_rebuild], dim=0)
            # full_points = torch.cat([rebuild_points,vis_points], dim=0)
            full_center = torch.cat([center[mask], center[~mask]], dim=0)
            # full = full_points + full_center.unsqueeze(1)
            ret2 = full_vis.reshape(-1, 3).unsqueeze(0)
            ret1 = full.reshape(-1, 3).unsqueeze(0)

            # return ret1, ret2
            return ret1, ret2, full_center
        else:
            return loss_recon, loss_contras


# finetune model
@MODELS.register_module()
class PointTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        self.trans_dim = config.trans_dim
        self.depth = config.depth
        self.drop_path_rate = config.drop_path_rate
        self.cls_dim = config.cls_dim
        self.num_heads = config.num_heads

        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        self.encoder = Encoder(encoder_channel=self.encoder_dims)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim=self.trans_dim,
            depth=self.depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        self.cls_head_finetune = nn.Sequential(
                nn.Linear(self.trans_dim * 2, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, self.cls_dim)
            )

        self.build_loss_func()

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder') :
                    base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log('missing_keys', logger='Transformer')
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger='Transformer'
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='Transformer')
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger='Transformer'
                )

            print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger='Transformer')
        else:
            print_log('Training from scratch!!!', logger='Transformer')
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, pts):

        neighborhood, center = self.group_divider(pts)
        group_input_tokens = self.encoder(neighborhood)  # B G N

        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)

        pos = self.pos_embed(center)

        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        # transformer
        x = self.blocks(x, pos)
        x = self.norm(x)
        concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)
        ret = self.cls_head_finetune(concat_f)
        return ret