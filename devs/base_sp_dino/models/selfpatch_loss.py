
import numpy as np
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from helper import trunc_normal_


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Class_Attention(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to do CA 
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    
    def forward(self, x, attention=False, mask=None):
        
        B, N, C = x.shape
        q = self.q(x[:,0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) 
        if mask is not None:
            mask_temp = torch.cat([torch.ones(B,1).bool().cuda(), mask],dim=1).unsqueeze(1).unsqueeze(1).expand(-1,self.num_heads,-1,-1)
            attn = attn.masked_fill_(~mask_temp.bool(), float("-inf"))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_cls = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)
        
        if attention:
            return x_cls, attn
        else:
            return x_cls


class LayerScale_Block_CA(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # with slight modifications to add CA and LayerScale
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, Attention_block = Class_Attention,
                 Mlp_block=Mlp):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, x_cls, attention=False, mask=None):
        u = torch.cat((x_cls,x),dim=1)
        if attention:
            u_, cls_attn = self.attn(self.norm1(u), attention=True)
            return cls_attn
        else:
            u_ = self.attn(self.norm1(u), mask=mask)
        x_cls = x_cls + self.drop_path(u_)
        x_cls = x_cls + self.drop_path(self.mlp(self.norm2(x_cls)))
        return x_cls


class SelfPatchHead(nn.Module):
    def __init__(self, in_dim, num_heads):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, in_dim))
        self.cls_blocks = nn.ModuleList([
            LayerScale_Block_CA(
                dim=in_dim, num_heads=num_heads, mlp_ratio=4.0, qkv_bias=True, qk_scale=None,
                drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU, Attention_block=Class_Attention,
                Mlp_block=Mlp)
            for i in range(2)])
        trunc_normal_(self.cls_token, std=.02)
        self.norm = partial(nn.LayerNorm, eps=1e-6)(in_dim)

        self.apply(self._init_weights)
        self.embed_dim = in_dim

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, loc=False):
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        if loc: # only for teacher
            x_loc = x
            for i, blk in enumerate(self.cls_block):
                if i ==0:
                    glo_tokens = blk(x, cls_tokens)
                    loc_tokens = blk(x_loc, cls_tokens.repeat(x.shape[1],1,1))
                else:
                    glo_tokens = blk(x, glo_tokens)
                    loc_tokens = blk(x_loc, loc_tokens)
            loc_tokens = loc_tokens.view(x.shape)
            x = self.norm(torch.cat([glo_tokens, loc_tokens], dim=1))

        else: # only for student
            for i, blk in enumerate(self.cls_blocks):
                cls_tokens = blk(x, cls_tokens)
            x = self.norm(torch.cat([cls_tokens, x], dim=1))

        return x
    

class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class DINOLoss(nn.Module):
    def __init__(self, out_dim, out_dim_selfpatch, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, 1, out_dim))
        self.register_buffer("patch_center", torch.zeros(1, out_dim_selfpatch))

        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch, it):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        # teacher centering and sharpening
        student_cls = student_output[0].chunk(2)
        student_loc = student_output[1].chunk(2)

        teacher_cls = teacher_output[0].chunk(2) 
        teacher_loc = teacher_output[1].chunk(2)

        temp = self.teacher_temp_schedule[epoch]

        c_loss = 0
        p_loss = 0
        n_loss_terms = 0
        m_loss_terms = 0
        for iq in range(len(teacher_cls)):
            q_cls = F.softmax((teacher_cls[iq] - self.center)/ temp, dim=-1).detach()
            for v in range(self.ncrops): #  N groups
                if v == iq:
                    q_pat = F.softmax((teacher_loc[iq] - self.patch_center)/ temp, dim=-1).detach()


                    p_pat = student_loc[v]
                    patch_loss = torch.sum(-q_pat * F.log_softmax(p_pat / self.student_temp, dim=-1), dim=-1)
                    p_loss += patch_loss.mean()
                    m_loss_terms += 1
                else:
                    if iq > 1:
                        continue
                    cls_loss = torch.sum(-q_cls * F.log_softmax(student_cls[v] / self.student_temp, dim=-1), dim=-1)
                    c_loss += cls_loss.mean()
                    n_loss_terms += 1
        c_loss /= n_loss_terms
        p_loss /= m_loss_terms
        
        self.update_center(torch.cat(teacher_cls), it)
        self.update_patch_center(teacher_loc, it)
        return (c_loss + p_loss*0.1), c_loss.item(), p_loss.item()

    @torch.no_grad()
    def update_center(self, teacher_output, it):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

    @torch.no_grad()
    def update_patch_center(self, teacher_output, it):
        self.patch_center = self.patch_center * self.center_momentum + batch_center * (1 - self.center_momentum)



if __name__ == "__main__":
    # pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fake_inp = torch.rand(128, 64, 384).to(device)

    aggreation_head = SelfPatchHead(in_dim=384, num_heads=6).to(device)
    out_agg = aggreation_head(fake_inp)

    projection_head = DINOHead(in_dim=384, out_dim=384, 
                               use_bn=False, norm_last_layer=True, 
                               nlayers=3, hidden_dim=2048, bottleneck_dim=256).to(device) 
    
    out_proj = projection_head(out_agg)

    print(out_proj.shape)
