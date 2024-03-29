import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from extensions.pointops.functions import pointops
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *
from .build import MODELS
from .detr.build import build_encoder as build_encoder_3detr, build_preencoder as build_preencoder_3detr
from .transformer import TransformerEncoder, Group, DummyGroup, Encoder, TransformerDecoder


# For Finetuing
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
        # grouper
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        # define the encoder
        self.encoder_dims = config.encoder_dims
        self.encoder = Encoder(encoder_channel=self.encoder_dims)
        # bridge encoder and transformer
        self.reduce_dim = nn.Identity()
        if self.encoder_dims != self.trans_dim:
            self.reduce_dim = nn.Linear(self.encoder_dims, self.trans_dim)

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
            num_heads=self.num_heads
        )

        self.norm = nn.LayerNorm(self.trans_dim)

        self.cls_head_arch = config.get('cls_head_arch', '1x')
        if self.cls_head_arch == '2x':
            self.cls_head_finetune = nn.Sequential(
                nn.Linear(self.trans_dim * 2, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, self.cls_dim)
            )
        else:
            self.cls_head_finetune = nn.Sequential(
                nn.Linear(self.trans_dim * 2, 256),
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
        ckpt = torch.load(bert_ckpt_path, map_location="cpu")
        base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
        for k in list(base_ckpt.keys()):
            if k.startswith('transformer_q') and not k.startswith('transformer_q.cls_head'):
                base_ckpt[k[len('transformer_q.'):]] = base_ckpt[k]
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

    def forward(self, pts, return_feature=False):
        # divide the point cloud in the same form. This is important
        neighborhood, center = self.group_divider(pts)
        # encoder the input cloud blocks
        group_input_tokens = self.encoder(neighborhood)  # B G N
        group_input_tokens = self.reduce_dim(group_input_tokens)
        # prepare cls
        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)
        # add pos embedding
        pos = self.pos_embed(center)
        # final input
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        # transformer
        x = self.blocks(x, pos)
        x = self.norm(x)
        if return_feature: return x
        concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)
        ret = self.cls_head_finetune(concat_f)
        return ret


class MaskPointTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        # define the encoder
        self.num_group = config.transformer_config.num_group
        self.group_size = config.transformer_config.group_size
        self.encoder_dims = config.transformer_config.encoder_dims
        # define the transformer
        self.mask_ratio = config.transformer_config.mask_ratio
        self.trans_dim = config.transformer_config.trans_dim
        self.depth = config.transformer_config.depth
        self.dec_depth = config.transformer_config.dec_depth
        self.dec_query_mode = config.transformer_config.dec_query_mode
        self.dec_query_real_num = config.transformer_config.dec_query_real_num
        self.dec_query_fake_num = config.transformer_config.dec_query_fake_num
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.cls_dim = config.transformer_config.cls_dim
        self.num_heads = config.transformer_config.num_heads
        self.ambiguous_threshold = config.transformer_config.ambiguous_threshold
        self.ambiguous_dynamic_threshold = config.transformer_config.ambiguous_dynamic_threshold
        print_log(f'[Transformer args] {config.transformer_config}', logger='MaskPoint')
        # define the encoder
        self.enc_arch = config.transformer_config.get('enc_arch', 'PointViT')
        if self.enc_arch == '3detr':
            self.encoder = build_preencoder_3detr(num_group=self.num_group, group_size=self.group_size,
                                                  dim=self.encoder_dims)
        else:
            self.encoder = Encoder(encoder_channel=self.encoder_dims)
        # bridge encoder and transformer
        self.reduce_dim = nn.Identity()
        if self.encoder_dims != self.trans_dim:
            self.reduce_dim = nn.Linear(self.encoder_dims, self.trans_dim)

        # define the learnable tokens
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.trans_dim))
        self.mask_token = nn.Parameter(torch.randn(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        # pos embeddings
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        # define the transformer blocks
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        if self.enc_arch == '3detr':
            self.blocks = build_encoder_3detr(
                ndim=self.trans_dim,
                nhead=self.num_heads,
                nlayers=self.depth
            )
        else:
            self.blocks = TransformerEncoder(
                embed_dim=self.trans_dim,
                depth=self.depth,
                drop_path_rate=dpr,
                num_heads=self.num_heads
            )
        self.decoder = TransformerDecoder(
            embed_dim=self.trans_dim,
            depth=self.dec_depth,
            drop_path_rate=dpr,
            num_heads=self.num_heads
        )
        self.cls_head = nn.Sequential(
            nn.Linear(self.trans_dim, self.cls_dim),
            nn.GELU(),
            nn.Linear(self.cls_dim, self.cls_dim)
        )
        self.project = nn.Sequential(
            nn.Linear(self.trans_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, self.cls_dim)
        )
        # layer norm
        self.norm = nn.LayerNorm(self.trans_dim)
        # initialize the learnable tokens
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)
        trunc_normal_(self.mask_token, std=.02)

        self.apply(self._init_weights)
        self.access_count = 0

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

    def _generate_fake_query(self, target):
        B = target.shape[0]
        min_coords, max_coords = torch.min(target, dim=1, keepdim=True)[0], torch.max(target, dim=1, keepdim=True)[0]
        fake_target = torch.rand(B, self.dec_query_fake_num, 3, dtype=target.dtype, device=target.device) * (
                    max_coords - min_coords) + min_coords
        return fake_target

    def _generate_query_xyz(self, points, center, mode='center'):
        if mode == 'center':
            target = center
        elif mode == 'points':
            if self.dec_query_real_num == -1:
                target = points
            else:
                target = pointops.fps(points, self.dec_query_real_num)
        else:
            raise NotImplementedError
        bs, npoints, _ = target.shape
        q, fake_q = target, self._generate_fake_query(target)
        nn_dist = pointops.knn(fake_q, points, 1)[1].squeeze()
        if self.ambiguous_dynamic_threshold > 0:
            assert self.ambiguous_threshold == -1
            if self.ambiguous_dynamic_threshold == self.dec_query_real_num:
                thres_q = q
            else:
                thres_q = pointops.fps(points, self.ambiguous_dynamic_threshold)
            dist_thres = pointops.knn(thres_q, thres_q, 2)[1][..., -1].mean(-1, keepdims=True)
        else:
            assert self.ambiguous_dynamic_threshold == -1
            dist_thres = self.ambiguous_threshold
        queries = torch.cat((q, fake_q), dim=1)
        labels = torch.zeros(bs, queries.shape[1], dtype=torch.long, device=target.device)
        labels[:, :npoints] = 1
        labels[:, npoints:][nn_dist < dist_thres] = -1

        return queries, labels

    def preencoder(self, neighborhood):
        group_input_tokens = self.encoder(neighborhood)  # B G N
        group_input_tokens = self.reduce_dim(group_input_tokens)
        return group_input_tokens

    def forward(self, neighborhood, center, is_eval=False, noaug=False, points_orig=None):
        if self.enc_arch == '3detr':
            pre_enc_xyz, group_input_tokens, pre_enc_inds = self.preencoder(center)
            group_input_tokens = group_input_tokens.permute(0, 2, 1)
            center = pre_enc_xyz
        else:
            group_input_tokens = self.preencoder(neighborhood)  # [128, 64, 384]
        B, G, _ = center.shape  # [B, 64, 3]
        mask = torch.zeros(B, G, dtype=torch.bool, device=center.device)
        if not noaug:
            if type(self.mask_ratio) is list:
                assert len(self.mask_ratio) == 2
                mask_ratio = random.uniform(*self.mask_ratio)
                n_mask = int(mask_ratio * G)
            elif self.mask_ratio >= 0:
                n_mask = int(self.mask_ratio * G)
            perm = torch.randperm(G)[:n_mask]
            mask[:, perm] = True
        else:
            n_mask = 0
        n_unmask = G - n_mask

        # mask: [128, 64]
        # vis_input_tokens: [128, n_vis, 384], vis_centers: [128, n_vis, 3]
        vis_input_tokens = group_input_tokens[~mask].view(B, n_unmask, -1)
        vis_centers = center[~mask].view(B, n_unmask, -1)

        # [128, 1, 384]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        cls_pos = self.cls_pos.expand(B, -1, -1)

        pos = self.pos_embed(vis_centers)  # [128, n_vis, 384]

        if self.enc_arch == '3detr':
            x = self.blocks(vis_input_tokens.transpose(0, 1), pos=pos.transpose(0, 1))[1].transpose(0, 1)

            if is_eval:
                return self.cls_head(torch.mean(x, dim=1))
        else:
            # [128, 1 + num_vis, 384]
            x = torch.cat((cls_tokens, vis_input_tokens), dim=1)
            pos = torch.cat((cls_pos, pos), dim=1)

            # --- ViT Encoder: [128, num_vis, 384] 
            x_vis = self.blocks(x, pos)
            x_vis = self.norm(x_vis)

            if is_eval:
                return x_vis.mean(1) + x_vis.max(1)[0]  # [128, 64, 384]

            B, _, C = x_vis.shape
            if n_mask != 0:  # mask_ratio != 0
                pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)
                pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)
                pos_full = torch.cat([cls_pos, pos_emd_vis, pos_emd_mask], dim=1)

                _, N, _ = pos_emd_mask.shape  # N: num. of the mask patches
                mask_token = self.mask_token.expand(B, N, -1)
                x_full = torch.cat([x_vis, mask_token], dim=1)
            else:  # Only for mask_ratio = 0
                pos_full = self.decoder_pos_embed(center).reshape(B, -1, C)
                pos_full = torch.cat([cls_pos, pos_full], dim=1)
                x_full = x_vis

            # --- ViT Decoder: [B, 64, 384]
            x = self.decoder(x_full, pos_full)

        # [128, 512], [128, 64, 128]
        return self.cls_head(x[:, 0]), self.project(x[:, 1:])


# For PreTraining
@MODELS.register_module()
class MaskPoint(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[MaskPoint] build MaskPoint...', logger='MaskPoint')
        self.config = config
        self.m = config.m
        self.T = config.T
        self.K = config.K

        self.transformer_q = MaskPointTransformer(config)
        self.transformer_k = MaskPointTransformer(config)
        for param_q, param_k in zip(self.transformer_q.parameters(), self.transformer_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        self.use_moco_loss = config.transformer_config.use_moco_loss
        self.moco_loss_weight = config.transformer_config.moco_loss_weight
        self.selfpatch_loss_weight = config.transformer_config.selfpatch_loss_weight
        self.use_self_patch = config.transformer_config.use_self_patch
        self.use_focal_loss = config.transformer_config.use_focal_loss
        self.use_sigmoid = config.transformer_config.use_sigmoid
        if self.use_focal_loss:
            self.focal_loss_alpha = config.transformer_config.focal_loss_alpha
            self.focal_loss_gamma = config.transformer_config.focal_loss_gamma

        self.group_size = config.transformer_config.group_size
        self.num_group = config.transformer_config.num_group

        print_log(f'[MaskPoint Group] divide point cloud into G{self.num_group} x S{self.group_size} points ...',
                  logger='MaskPoint')
        self.enc_arch = config.transformer_config.get('enc_arch', 'PointViT')
        self.group_divider = (DummyGroup if self.enc_arch == '3detr' else Group)(num_group=self.num_group,
                                                                                 group_size=self.group_size)

        # create the queue
        self.register_buffer("queue", torch.randn(self.transformer_q.cls_dim, self.K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.loss_ce = nn.CrossEntropyLoss()
        self.loss_ce_batch = nn.CrossEntropyLoss(reduction='none')

        self.patch_predict = nn.Sequential(
            nn.Linear(self.transformer_q.cls_dim, self.transformer_q.cls_dim),
        )

        # loss
        self.build_loss_func()

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.transformer_q.parameters(), self.transformer_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0

        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K

        self.queue_ptr[0] = ptr

    def build_loss_func(self):
        if self.use_sigmoid:
            self.loss_bce_batch = nn.BCEWithLogitsLoss(reduction='none')
        else:
            self.loss_ce = nn.CrossEntropyLoss(ignore_index=-1)
            self.loss_ce_batch = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)

    def forward_eval(self, pts):
        with torch.no_grad():
            neighborhood, center = self.group_divider(pts)
            feats_svm = self.transformer_q(neighborhood, center, is_eval=True, noaug=True, points_orig=pts)

            return feats_svm

    def loss_focal_bce(self, pred, target):
        pred_sigmoid = pred.sigmoid()
        target = target.type_as(pred)
        pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
        focal_weight = (self.focal_loss_alpha * target + (1 - self.focal_loss_alpha) * (1 - target)) * pt.pow(
            self.focal_loss_gamma)
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none') * focal_weight
        return loss

    def loss_bce(self, preds, labels, reduction='mean'):
        loss_labels = labels.clone()
        loss_labels[labels == -1] = 0
        loss_labels_one_hot = F.one_hot(loss_labels, num_classes=2)
        preds = preds.transpose(1, 2).contiguous()

        if self.use_focal_loss:
            loss = self.loss_focal_bce(preds, loss_labels_one_hot)
        else:
            loss = self.loss_bce_batch(preds, loss_labels_one_hot.float())
        if reduction == 'mean':
            loss = loss[labels != -1].mean()
        return loss

    def forward(self, pts, noaug=False, **kwargs):  # PTS: [128(bs), 1024, 3]
        if noaug:
            return self.forward_eval(pts)

        self._momentum_update_key_encoder()
        neighborhood, center = self.group_divider(pts)  # neighborhood: [128, 64, 32, 3], center: [128, 64, 3]
        # q_cls_feature: [128, cls_dim], [128, 64, 128]
        q_cls_feats, q_patch_feats = self.transformer_q(neighborhood, center, is_eval=False, points_orig=pts)
        q_cls_feats = F.normalize(q_cls_feats, dim=1)
        q_patch_predict = self.patch_predict(q_patch_feats)

        if self.use_self_patch:
            with torch.no_grad():
                k_cls_feats, k_patch_feats = self.transformer_k(
                    neighborhood, center, is_eval=False, points_orig=pts, noaug=True)
                k_cls_feats = F.normalize(k_cls_feats, dim=1)
                k_patch_feats_norm = F.normalize(k_patch_feats, dim=-1)
            q_patch_predict = F.normalize(q_patch_predict, dim=-1)
            # SelfPatch
            cdist = torch.cdist(center, center)
            radius = torch.topk(cdist, k=2, dim=-1, largest=False)[0][:, 1].mean(dim=-1, keepdim=True)
            feats_dis = torch.cdist(k_patch_feats_norm, k_patch_feats_norm)
            mask = (cdist < radius.unsqueeze(-1) / np.sqrt(2)).to(cdist)

            global_weight = torch.exp(-cdist / 1.0) * (2 - feats_dis)
            neighbor_weight = (global_weight * mask / (
                (global_weight * mask).sum(dim=-1, keepdim=True).clip(min=1e-5))).detach()

            new_feats = torch.einsum('bmk,bmd->bkd', neighbor_weight, k_patch_feats)
            new_feats = F.normalize(new_feats, dim=-1)

            gamma_log = torch.einsum('bmd,bnd->bmn', q_patch_predict, new_feats)
            selfpatch_loss = -torch.mean(
                torch.sum(mask * global_weight.detach() * F.log_softmax(gamma_log, dim=1), dim=1))
            selfpatch_loss = self.selfpatch_loss_weight * selfpatch_loss
        else:
            selfpatch_loss = torch.tensor(0.).to(pts.device)

        if self.use_moco_loss:
            l_pos = torch.einsum('nc, nc->n', [q_cls_feats, k_cls_feats]).unsqueeze(-1)
            l_neg = torch.einsum('nc, ck->nk', [q_cls_feats, self.queue.clone().detach()])
            ce_logits = torch.cat([l_pos, l_neg], dim=1) / self.T
            labels = torch.zeros(l_pos.shape[0], dtype=torch.long).to(pts.device)
            moco_loss = self.loss_ce(ce_logits, labels)
            moco_loss = self.moco_loss_weight * moco_loss
        else:
            moco_loss = torch.tensor(0.).to(pts.device)

        self._dequeue_and_enqueue(k_cls_feats)

        return selfpatch_loss, moco_loss
