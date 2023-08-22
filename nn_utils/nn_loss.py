import torch
import torch.nn as nn

import motion_tensor.rotations


class EndEffectorLoss(nn.Module):
    def __init__(self, norm_eps=0.008):
        super(EndEffectorLoss, self).__init__()
        self.criterion = nn.MSELoss()
        self.norm_eps = norm_eps

    def forward(self, ee_ref, ee_out):
        """
        [B, ..., T]
        """
        ee_ref = ee_ref[..., 1:] - ee_ref[..., :-1]
        ee_out = ee_out[..., 1:] - ee_out[..., :-1]

        ee_ref = ee_ref * 10.0
        ee_out = ee_out * 10.0

        ref_ee_loss = self.criterion(ee_ref, ee_out)

        ref_norm = torch.norm(ee_ref, dim=-2, keepdim=True)
        contact_idx = ref_norm < self.norm_eps
        B, J, _, F = ref_norm.shape
        contact_idx = torch.broadcast_to(contact_idx, (B, J, 3, F))
        extra_ee_loss = self.criterion(ee_ref[contact_idx], ee_out[contact_idx])

        return ref_ee_loss + extra_ee_loss * 100


class FootContactLoss(nn.Module):
    def __init__(self, y_eps=0.03):
        super(FootContactLoss, self).__init__()
        self.criterion = nn.MSELoss()
        self.y_eps = y_eps

    def forward(self, feet_ref, feet_out):
        """
        [B, J, 3, T]
        """
        ref_y = feet_ref[:, :, 1, :]
        out_y = feet_out[:, :, 1, :]

        y_index = ref_y.abs() < self.y_eps  # [B, 2, F]
        fc_ref = ref_y[y_index]
        fc_out = out_y[y_index]
        if fc_ref.nelement() != 0:  # no foot contact
            fc_loss = self.criterion(fc_ref, fc_out)
        else:
            fc_loss = 0.0
        return fc_loss * 100.0


class TwistLoss(nn.Module):
    def __init__(self, alpha=100.0):
        super(TwistLoss, self).__init__()
        self.alpha = alpha / 180.0 * 3.1415926

    def forward(self, qua):
        B, J, Q, T = qua.shape
        assert Q == 4

        normed = motion_tensor.rotations.normalize_quaternion(qua)
        w = normed[..., 0:1, :]
        x = normed[..., 1:2, :]
        y = normed[..., 2:3, :]
        z = normed[..., 3:4, :]
        euler_y = torch.atan2(2 * (y * w - x * z), x * x - y * y - z * z + w * w)

        # eul = mot.rotations.quaternion_to_euler(normed, 'ZYX')
        # euler_y = eul[:, 1:, 1, :]  # (B, J, F) except for root joint

        root_ey = euler_y[:, 1:, :, :]
        diff = torch.clamp(torch.abs(root_ey) - self.alpha, min=0)  # within alpha(==100 by default) degree
        loss = torch.mean(diff ** 2)
        return loss


class QuaternionLoss(nn.Module):
    def __init__(self):
        super(QuaternionLoss, self).__init__()

    # noinspection PyMethodMayBeStatic
    def forward(self, qua):
        B, J, Q, T = qua.shape
        assert Q == 4
        qua = qua.view(B, J, -1, T)
        qua_norm = torch.norm(qua, dim=2, keepdim=True)
        l_qt = torch.mean((qua_norm - 1) ** 2)

        return l_qt
