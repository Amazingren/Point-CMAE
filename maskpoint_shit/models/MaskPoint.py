import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from extensions.pointops.functions import pointops
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *
from .build import MODELS
from .cluster import gmm_params, ot_assign, GeoDecoder
from .detr.build import build_encoder as build_encoder_3detr, build_preencoder as build_preencoder_3detr
from .transformer import TransformerEncoder, TransformerDecoder, Group, DummyGroup, Encoder, TransformerDecoderMAE


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
        # divide the point clo  ud in the same form. This is important
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
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.cls_dim = config.transformer_config.cls_dim
        self.use_sigmoid = config.transformer_config.use_sigmoid
        self.num_heads = config.transformer_config.num_heads
        self.use_pts_mae_loss = config.transformer_config.use_pts_mae_loss
        self.use_cluster_loss = config.transformer_config.use_cluster_loss
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

        # pos embedding for each patch
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
        self.MAE_decoder = TransformerDecoderMAE(
            embed_dim=self.trans_dim,
            depth=self.dec_depth + 3,
            drop_path_rate=dpr,
            num_heads=self.num_heads,
        )
        self.cls_head = nn.Sequential(
            nn.Linear(self.trans_dim, self.cls_dim),
            nn.GELU(),
            nn.Linear(self.cls_dim, self.cls_dim)
        )
        self.bin_cls_head = nn.Sequential(
            nn.Linear(self.trans_dim, 64),
            nn.GELU(),
            nn.Linear(64, 2)
        )
        # layer norm
        self.norm = nn.LayerNorm(self.trans_dim)
        # initialize the learnable tokens
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)
        trunc_normal_(self.mask_token, std=.02)

        self.apply(self._init_weights)
        self.access_count = 0

        # self.linear_distri = nn.Sequential(
        #     nn.Linear(384, 256),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(256, 8)
        # )
        self.upsample = GeoDecoder(num_stages=1, de_neighbors=8)
        self.linear_distri = nn.Sequential(
            nn.Linear(384 * 2, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64)
        )

        # prediction head
        self.increase_dim = nn.Sequential(
            # nn.Conv1d(self.trans_dim, 1024, 1),
            # nn.BatchNorm1d(1024),
            # nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(self.trans_dim, 3 * self.group_size, 1)
        )

        self.loss = config.loss
        # loss
        self.build_loss_func(self.loss)

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

    def preencoder(self, neighborhood):
        group_input_tokens = self.encoder(neighborhood)  # B G N
        group_input_tokens = self.reduce_dim(group_input_tokens)
        return group_input_tokens

    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            self.loss_func = ChamferDistanceL1().cuda()
        elif loss_type == 'cdl2':
            self.loss_func = ChamferDistanceL2().cuda()
        else:
            raise NotImplementedError
            # self.loss_func = emd().cuda()

    def forward(self, neighborhood, center, only_cls_tokens=False, noaug=False, points_orig=None):
        if self.enc_arch == '3detr':
            pre_enc_xyz, group_input_tokens, pre_enc_inds = self.preencoder(center)
            group_input_tokens = group_input_tokens.permute(0, 2, 1)
            center = pre_enc_xyz
        else:
            group_input_tokens = self.preencoder(neighborhood)  # [128, 64, 384]
        B, G, _ = center.shape
        mask = torch.zeros(B, G, dtype=torch.bool, device=center.device)
        if not noaug:
            if type(self.mask_ratio) is list:
                assert len(self.mask_ratio) == 2
                mask_ratio = random.uniform(*self.mask_ratio)
                n_mask = int(mask_ratio * G)
            elif self.mask_ratio > 0:
                n_mask = int(self.mask_ratio * G)
            perm = torch.randperm(G)[:n_mask]
            mask[:, perm] = True
        else:
            n_mask = 0
        n_unmask = G - n_mask

        # mask: [128, 64]
        # masked_input_tokens: [128, 7, 384]
        # masked_centers: [128, 7, 3]
        vis_input_tokens = group_input_tokens[~mask].view(B, n_unmask, -1)
        vis_centers = center[~mask].view(B, n_unmask, -1)

        # [128, 1, 384]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        cls_pos = self.cls_pos.expand(B, -1, -1)

        pos_vis = self.pos_embed(vis_centers)  # [128, 7, 384]

        if self.enc_arch == '3detr':
            x = self.blocks(vis_input_tokens.transpose(0, 1), pos=cls_pos.transpose(0, 1))[1].transpose(0, 1)

            if only_cls_tokens:
                return self.cls_head(torch.mean(x, dim=1))
        else:
            # [128, 1+7, 384]
            x_vis = torch.cat((cls_tokens, vis_input_tokens), dim=1)
            pos_vis = torch.cat((cls_pos, pos_vis), dim=1)

            # [128, 8, 384]
            x_vis = self.blocks(x_vis, pos_vis)
            x_vis = self.norm(x_vis)

            if only_cls_tokens:
                return self.cls_head(x_vis[:, 0])  # [128, 1, 384]

        # Prepare: {Q_real & Q_fake} & The Corresponding Labels
        # [128, 512, 3] & [128, 512]
        query_points, query_labels = self._generate_query_xyz(points_orig, center, mode=self.dec_query_mode)

        # [128, 512, 384]
        query_pos = self.pos_embed(query_points)
        query_tensor = torch.zeros_like(query_pos)

        # Cross Attn [128, 512, 384]. Between:
        # query_tensor{Q_real & Q_fake} & x_vis:  
        dec_outputs = self.decoder(query_tensor, query_pos, x_vis, pos_vis)
        # Two-Layer MLP: [128, 2, 512]
        query_preds = self.bin_cls_head(dec_outputs).transpose(1, 2)

        # --- Full Decoder
        _, _, C = x_vis.shape
        pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)
        _, N, _ = pos_emd_mask.shape
        mask_token = self.mask_token.expand(B, N, -1)
        x_full = torch.cat([x_vis, mask_token], dim=1)
        pos_full = torch.cat([pos_vis, pos_emd_mask], dim=1)
        # [128, 57, 384], [128, 65, 384]
        x_rec, feats_de = self.MAE_decoder(x_full, pos_full, N)

        # --- PointMAE Recons
        if self.use_pts_mae_loss:
            B, M, C = x_rec.shape
            rebuild_points = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)  # B M 1024
            gt_points = neighborhood[mask].reshape(B * M, -1, 3)
            loss_mae = self.loss_func(rebuild_points, gt_points)
        else:
            loss_mae = torch.tensor(0.).to(dec_outputs.device)

        #  --- Cluster Loss ---
        if self.use_cluster_loss:
            cls_token = feats_de[:, 0].unsqueeze(1).expand(-1, points_orig.shape[1], -1)  # [128, 64, 384]
            new_feats = self.upsample([points_orig, center], [cls_token, feats_de])  # [128, 64, 384*2] cls_token = None
            # new_feats = torch.cat([feats_de[:, 1:], cls_token], dim=-1)  # [128, 64, 384*2]

            # x_full cluster Prob. [128, 64, cluster_dim:8]
            gamma_log = self.linear_distri(new_feats)

            gamma = F.softmax(gamma_log, dim=-1)
            # [128, 64, cluster_dim:8], [128, 64, 3] -> [128, cluster_dim:8, 3]
            _, mu = gmm_params(gamma, points_orig)
            # [128, 64, 3], [128, 8, 3] -> [128, 64, 8]
            gamma_new, dist = ot_assign(points_orig, mu)

            loss_cluster = -torch.mean(torch.sum(gamma_new.detach() * F.log_softmax(gamma_log, dim=1), dim=1))
        else:
            loss_cluster = torch.tensor(0.).to(dec_outputs.device)

        # [128, 512], [128, 2, 512], [128, 512] 
        return self.cls_head(x_vis[:, 0]), query_preds, query_labels, loss_mae, loss_cluster


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
        self.query_loss_weight = config.transformer_config.query_loss_weight
        self.use_sigmoid = config.transformer_config.use_sigmoid
        self.use_focal_loss = config.transformer_config.use_focal_loss
        if self.use_focal_loss:
            self.focal_loss_alpha = config.transformer_config.focal_loss_alpha
            self.focal_loss_gamma = config.transformer_config.focal_loss_gamma
        self.use_cluster_loss = config.transformer_config.use_cluster_loss
        self.cluster_loss_weight = config.transformer_config.cluster_loss_weight
        self.use_pts_mae_loss = config.transformer_config.use_pts_mae_loss
        self.mae_loss_weight = config.transformer_config.mae_loss_weight

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
            cls_feature = self.transformer_q(neighborhood, center, only_cls_tokens=True, noaug=True, points_orig=pts)
            return cls_feature

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
        else:
            self._momentum_update_key_encoder()

            neighborhood, center = self.group_divider(pts)  # neighborhood: [128, 64, 32, 3], center: [128, 64, 3]

            # q_cls_feature: [128, 512], query_preds: [128, 2, 512], query_labels: [128, 512]
            q_cls_feature, query_preds, query_labels, loss_mae, loss_cluster = self.transformer_q(neighborhood, center,
                                                                                                  points_orig=pts)
            q_cls_feature = F.normalize(q_cls_feature, dim=1)

            if self.use_moco_loss:
                with torch.no_grad():
                    k_cls_feature = self.transformer_k(neighborhood, center, points_orig=pts, only_cls_tokens=True)
                    k_cls_feature = F.normalize(k_cls_feature, dim=1)
                l_pos = torch.einsum('nc, nc->n', [q_cls_feature, k_cls_feature]).unsqueeze(-1)
                l_neg = torch.einsum('nc, ck->nk', [q_cls_feature, self.queue.clone().detach()])
                ce_logits = torch.cat([l_pos, l_neg], dim=1) / self.T
                labels = torch.zeros(l_pos.shape[0], dtype=torch.long).to(pts.device)
                moco_loss = self.loss_ce(ce_logits, labels)
                moco_loss = self.moco_loss_weight * moco_loss
            else:
                moco_loss = torch.tensor(0.).to(pts.device)

            if self.use_moco_loss:
                self._dequeue_and_enqueue(k_cls_feature)

            if self.use_sigmoid:
                recon_loss = self.loss_bce(query_preds, query_labels)
            else:
                # recon_loss = self.loss_ce(query_preds, query_labels)
                recon_loss = torch.tensor(0.).to(pts.device)
            recon_loss = self.query_loss_weight * recon_loss

            # --- use PointMAE
            if self.use_pts_mae_loss:
                loss_mae = self.mae_loss_weight * loss_mae
            else:
                loss_mae = torch.tensor(0.).to(pts.device)

            # --- use Cluster
            if self.use_cluster_loss:
                loss_cluster = self.cluster_loss_weight * loss_cluster
            else:
                loss_cluster = torch.tensor(0.).to(pts.device)

            return recon_loss, moco_loss, loss_mae, loss_cluster
