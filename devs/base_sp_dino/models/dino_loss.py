import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        student_cls = student_output[0][0].chunk(2) + student_output[1][0].chunk(self.ncrops-2)
        student_loc = student_output[0][1].chunk(2) + student_output[1][1].chunk(self.ncrops-2)

        teacher_cls = teacher_output[0][0].chunk(2) + teacher_output[1][0].chunk(self.ncrops-2)
        teacher_loc = teacher_output[0][1].chunk(2) + teacher_output[1][1].chunk(self.ncrops-2)

        student_cls = student_output[0][0].chunk(2) + student_output[1][0].chunk(self.ncrops-2)



        student_cls = student_output[0].chunk(2)
        student_cls

        temp = self.teacher_temp_schedule[epoch]

        c_loss = 0
        p_loss = 0
        n_loss_terms = 0
        m_loss_terms = 0
        assert len(teacher_cls) == self.ncrops
        for iq in range(len(teacher_cls)):
            q_cls = F.softmax((teacher_cls[iq] - self.center)/ temp, dim=-1).detach()
            for v in range(self.ncrops):
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