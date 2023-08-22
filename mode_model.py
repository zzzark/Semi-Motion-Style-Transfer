import torch
import torch.nn as nn
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import motion_tensor as mot
from torch.utils.data.dataset import Dataset as TDataset
from nn_utils.nn_loss import QuaternionLoss, TwistLoss
from nn_utils.nn_blocks import ForwardKinematics
from copy import deepcopy
from nn_utils.uni_trainer import GenModel
import random

from mode_dataset import get_mixed_c_s_dataset, UniBVHDataExtractor, STSimpleProcessor, UniMotionDataDivider
# from mode_networks import EncoderContent, EncoderStyle, Decoder, AdaptiveIN
from net.model import EncoderCon, EncoderSty, Decoder
from mode_config import *
from net.blocks import StyleTripletLoss, EEVelLoss, EEPosLoss, EEDistLoss


class MoDeDataset(TDataset):
    def __init__(self, content_bvh_dataset, style_bvh_dataset, post_fix):
        super(MoDeDataset, self).__init__()
        self.c_dataset, self.s_dataset = get_mixed_c_s_dataset(content_bvh_dataset, style_bvh_dataset, post_fix)

    def __len__(self):
        return len(self.c_dataset)

    def __getitem__(self, item):
        c_idx = item
        while c_idx == item:
            c_idx = random.randint(0, len(self.c_dataset) - 1)
        x_dyn, x_pos, x_trs, _, _ = self.c_dataset[item]
        y_dyn, y_pos, y_trs, _, _ = self.c_dataset[c_idx]

        s_idx = random.randint(0, len(self.s_dataset)-1)
        m_dyn, m_pos, m_trs, _, class_id = self.s_dataset[s_idx]
        q_dyn, q_pos, q_trs, _, _ = self.s_dataset.fetch_same(class_id)
        n_dyn, n_pos, n_trs, _, _ = self.s_dataset.fetch_diff(class_id)

        return x_dyn, y_dyn, m_dyn, q_dyn, n_dyn


class MoDeGenerativeModel(GenModel):
    def __init__(self, dataset: MoDeDataset):
        super(MoDeGenerativeModel, self).__init__()

        config = {
            'enc_in_dim': 6,
            'enc_nf': 32,        # 64
            'latent_dim': 128,   # 128
            'graph': {
                'joint': {'layout': 'cmu', 'strategy': 'distance', 'max_hop': 2},
                'mid': {'layout': 'cmu', 'strategy': 'distance', 'max_hop': 1},
                'bodypart': {'layout': 'cmu', 'strategy': 'distance', 'max_hop': 1},
            }
        }
        enc_in_dim = config['enc_in_dim']
        enc_nf = config['enc_nf']
        latent_dim = config['latent_dim']
        graph_cfg = config['graph']

        self.enc_in_dim = enc_in_dim

        self.net_f = EncoderCon(enc_in_dim, enc_nf, graph_cfg=graph_cfg, norm=ENC_NORM)
        self.net_e = EncoderSty(enc_in_dim, enc_nf, graph_cfg=graph_cfg)
        self.net_d = Decoder(self.net_f.output_channels, enc_in_dim,
                             latent_dim=latent_dim, graph_cfg=graph_cfg)

        off = mot.bvh_casting.get_offsets_from_bvh(g_cmu_bvh)
        self._real_fk = ForwardKinematics(g_cmu_p_index, off)

        if EE_LOSS_TYPE == 'vel':
            ee_loss = EEVelLoss
        elif EE_LOSS_TYPE == 'pos':
            ee_loss = EEPosLoss
        elif EE_LOSS_TYPE == 'dist':
            ee_loss = EEDistLoss
        else:
            raise ValueError(f"Invalid ee loss type: {EE_LOSS_TYPE}")
        self.ee_loss = ee_loss(index=g_ee, d=EE_LOSS_D)
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

        self.stp1_loss = StyleTripletLoss(margin=g_w_margin_1)
        self.stp2_loss = StyleTripletLoss(margin=g_w_margin_2)
        self.relu = nn.ReLU()

        dd = dataset.c_dataset
        self.register_buffer("dm", dd.dyn_m[None, ...], persistent=False)  # (1, VC)
        self.register_buffer("dv", dd.dyn_v[None, ...], persistent=False)  # (1, VC)
        self.register_buffer("pm", dd.pos_m[None, ...], persistent=False)
        self.register_buffer("pv", dd.pos_v[None, ...], persistent=False)

        self.register_buffer("dm6", self.dm.view(1, -1, enc_in_dim).permute(0, 2, 1)[..., None])  # (1, C, V, T)
        self.register_buffer("dv6", self.dv.view(1, -1, enc_in_dim).permute(0, 2, 1)[..., None])  # (1, C, V, T)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Conv2d, nn.Conv1d)):
            nn.init.kaiming_normal_(module.weight.data, a=0, mode='fan_in')
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

    def dyn_norm(self, dyn):
        """
        (N, C, V, T)
        """
        assert len(dyn.shape) == 4
        dyn = (dyn - self.dm6) / self.dv6
        return dyn

    def dyn_denorm(self, dyn):
        """
        (N, C, V, T)
        """
        assert len(dyn.shape) == 4
        dyn = (dyn * self.dv6) + self.dm6
        return dyn

    def fk(self, dyn):
        """
        (N, V, 3, T)
        """
        dyn = self.dyn_denorm(dyn)
        pos = self._real_fk(dyn)
        # # DEBUG
        # from visualization.utils import quick_visualize as qv
        # qv(g_cmu_p_index, pos[..., ::2])  # down-sample to 30FPS
        return pos

    def output_to_qua(self, dyn):
        """
        (..., 4, T)
        """
        dyn = self.dyn_denorm(dyn)  # (N, C, V, T)
        r6d = dyn.permute(0, 2, 1, 3)  # (N, V, C, T)
        mtx = mot.rotations.rotation_6d_to_matrix(r6d)
        qua = mot.rotations.matrix_to_quaternion(mtx)
        return qua

    @staticmethod
    def vec_to_2d(*args):
        """
        (N, VC, T) -> (N, V, C, T) -> (N, C, V, T)
        """
        ret = []
        for x in args:
            x = x.view(x.shape[0], -1, 6, x.shape[-1]) if len(x.shape) == 3 else x  # (N, V, C, T)
            x = x.permute(0, 2, 1, 3)  # (N, C, V, T)
            ret.append(x)
        return tuple(ret) if len(ret) > 1 else ret[0]

    def forward(self, *args):
        x, y, m, q, n = args
        x, y, m, q, n = self.vec_to_2d(x, y, m, q, n)
        ret = ()

        fx = self.net_f(x)
        fm = self.net_f(m)
        pm = self.fk(m)
        e_ms = self.net_e(m)
        e_qs = self.net_e(q)
        e_nt = self.net_e(n)

        d_ms_ms = self.net_d(fm, e_ms)
        ret += (m, d_ms_ms)

        d_ms_qs = self.net_d(fm, e_qs)
        p_ms_qs = self.fk(d_ms_qs)
        ret += (m, d_ms_qs, pm, p_ms_qs)

        if g_w_t:
            ret += (e_ms, e_qs, e_nt)

        if g_w_rec:
            d_x = self.net_d(fx)
            ret += (x, d_x)

        if g_w_c or g_w_c_vel:
            d_fx_ms = self.net_d(fx, e_ms)
            ret += (x, d_fx_ms)

        if g_w_ee:
            if not g_w_c:
                d_fx_ms = self.net_d(fx, e_ms)
            p_x = self.fk(x)
            # noinspection PyUnboundLocalVariable
            p_fx_ms = self.fk(d_fx_ms)
            ret += (p_x, p_fx_ms)

        if g_w_ds:
            if not g_w_c and not g_w_ee:
                d_fx_ms = self.net_d(fx, e_ms)
            d_fx_nt = self.net_d(fx, e_nt)
            ret += (d_fx_ms, d_fx_nt)

        if g_w_s:
            if not g_w_c and not g_w_ee and not g_w_ds:
                d_fx_ms = self.net_d(fx, e_ms)
            d_fx_nt = self.net_d(fx, e_nt)
            e_xs = self.net_e(d_fx_ms)
            e_xt = self.net_e(d_fx_nt)
            ret += (e_ms, e_nt, e_xs, e_xt)

        return ret

        # if g_w_s > 0.0:
        #     d_fx_nt = self.net_d(self.net_a(fx, e_nt))
        #     d_fy_nt = self.net_d(self.net_a(fy, e_nt))
        #
        #     e_xs = self.net_e(d_fx_ms)
        #     e_xt = self.net_e(d_fx_nt)
        #     e_yt = self.net_e(d_fy_nt)
        #     ret += (e_ms, e_nt, e_xs, e_xt, e_yt)
        #
        # return ret

    def get_loss(self, *output):
        loss_dict = {}
        loss_list = []

        m, d_ms_ms, _, d_ms_qs, pm, p_ms_qs, *output = output

        l_con = (self.l1_loss(m, d_ms_ms) + self.l1_loss(m, d_ms_qs)) * g_w_con * 0.5
        loss_dict['S.con'] = l_con.item()
        loss_list.append(l_con)

        l_p_con = self.l1_loss(pm, p_ms_qs) * g_w_con * g_pos_w_decay
        loss_dict['S.p_con'] = l_p_con.item()
        loss_list.append(l_p_con)

        # >>>>>>>>
        if g_w_e_con:
            l_e_con = self.ee_loss(pm, p_ms_qs) * g_w_e_con
            loss_dict['S.e_con'] = l_e_con.item()
            loss_list.append(l_e_con)
        # <<<<<<<<

        if g_w_t:
            e_ms, e_qs, e_nt, *output = output
            l_tri = self.stp1_loss(e_ms, e_qs, e_nt) * g_w_t
            loss_dict['S.tri'] = l_tri.item()
            loss_list.append(l_tri)

        if g_w_rec:
            x, d_x, *output = output
            l_rec = self.l1_loss(x, d_x) * g_w_rec
            loss_dict['C.rec'] = l_rec.item()
            loss_list.append(l_rec)

        if g_w_c or g_w_c_vel:
            x, d_fx_ms, *output = output
            if g_w_c:
                # l_c = self.l1_loss(x, d_fx_ms) * g_w_c
                l_c = self.mse_loss(x[:, :, g_id_c, :], d_fx_ms[:, :, g_id_c, :]) * g_w_c
                loss_dict['C.c'] = l_c.item()
                loss_list.append(l_c)
            if g_w_c_vel:
                vx_ = x[..., 1:] - x[..., :-1]
                vxs = d_fx_ms[..., 1:] - d_fx_ms[..., :-1]

                l_c_vel = self.l1_loss(vx_, vxs) * g_w_c_vel
                loss_dict['C.c_vel'] = l_c_vel.item()
                loss_list.append(l_c_vel)

        if g_w_ee:
            p_x, p_fx_ms, *output = output
            l_ee = self.ee_loss(p_x, p_fx_ms) * g_w_ee
            loss_dict['C.ee'] = l_ee.item()
            loss_list.append(l_ee)

        if g_w_ds:
            d_fx_ms, d_fx_nt, *output = output
            l_ds = self.l1_loss(d_fx_ms, d_fx_nt) * g_w_ds
            loss_dict['C.ds'] = l_ds.item()
            loss_list.append(-l_ds)

        if g_w_s:
            e_ms, e_nt, e_xs, e_xt, *output = output

            stop_grad = True
            if stop_grad:
                e_ms_sg = [e.detach() for e in e_ms]
            else:
                e_ms_sg = e_ms
            l_s = self.stp2_loss(e_ms_sg, e_xs, e_xt) * g_w_s
            loss_dict['C.s'] = l_s.item()
            loss_list.append(l_s)

        loss_total = torch.sum(torch.stack(loss_list))
        return loss_total, loss_dict

        # if g_w_s > 0.0:
        #     e_ms, e_nt, e_xs, e_xt, e_yt, *output = output
        #     ls = self.tp2_loss(e_ms, e_xs, e_xt) * g_w_s
        #     loss_total = loss_total + ls
        #     loss_dict['ls'] = ls.item()
        #
        #     if g_w_d > 0.0:
        #         d1 = ((e_xt - e_yt) ** 2).sum(dim=-1)  # .sqrt
        #         d2 = ((e_nt - e_yt) ** 2).sum(dim=-1)  # .sqrt
        #         ld = self.relu(d1 - d2).mean() * g_w_d
        #         loss_total = loss_total + ld
        #         loss_dict['ld'] = ld.item()
        #
        # return loss_total, loss_dict


def revert_root_rotation(qua, yrt, hip_index):
    """
    (J, 4, T), (T, ), (1, 3, T)
    """
    f = qua.shape[-1]

    # y-rotation(euler) to quaternion
    eul = torch.zeros((1, 3, f), device=qua.device)
    eul[:, 1, :] = yrt
    yrt = mot.rotations.euler_to_quaternion(eul, to_rad=1.0)
    yrt = mot.rotations.normalize_quaternion(yrt)
    yrt[:, 1:, :].neg_()  # inverse rotation

    out = mot.rotations.mul_two_quaternions(yrt, qua[[hip_index], :, :])
    out = mot.rotations.normalize_quaternion(out)
    qua[[hip_index], :, :] = out

    return qua


def align_frames(*motion):
    f = min([e.shape[-1] for e in motion])
    ret = [e[..., :f] for e in motion]
    return tuple(ret)


@torch.no_grad()
def mode_reconstruct(x_bvh: bvh.parser.BVH, output_path,
                     device, dataset: MoDeDataset, model: MoDeGenerativeModel):

    return

    # class __Vars:
    #     extractor = UniBVHDataExtractor(g_desired_frame_time)
    #     processor = STSimpleProcessor()
    #
    # model.eval()
    #
    # x_static, x_dynamic = __Vars.extractor.extract(x_bvh)
    #
    # x_static =  __Vars.processor.f_process_static(0, *x_static)
    # x_dynamic = __Vars.processor.f_process_dynamic(0, *x_dynamic)
    # x_dyn, x_pos, x_trs, x_yrt = dataset.c_dataset.to_input(x_static, x_dynamic)
    #
    # def __pre_input(x):
    #     x = x[..., :x.shape[-1] // 4 * 4].to(device)[None, ...]
    #     x = model.vec_to_2d(x)
    #     return x
    #
    # x_dyn = __pre_input(x_dyn)
    # x_pos = __pre_input(x_pos)
    # x_trs = __pre_input(x_trs)
    #
    # """
    # perform style transfer
    # """
    # fx = model.net_f(x_dyn)
    # o_dyn = model.net_d(fx)
    # o_dyn = model.dyn_denorm(o_dyn)
    #
    # """
    # the global part of the output is ignored, however for simplicity it is still outputted
    # """
    # o_qua = model.dyn_to_qua(o_dyn)
    # o_qua, o_trs, o_yrt = o_qua[0], x_trs[0], x_yrt[0]
    # o_qua = revert_root_rotation(o_qua, o_yrt, g_hip)
    # o_trs, o_qua = align_frames(o_trs, o_qua)
    #
    # o_bvh = deepcopy(x_bvh)
    # off = mot.bvh_casting.get_offsets_from_bvh(o_bvh)
    # off *= g_output_scaling
    # mot.bvh_casting.write_offsets_to_bvh(o_bvh, off)
    # o_trs *= g_output_scaling
    # mot.bvh_casting.write_quaternion_to_bvh_object(o_trs, o_qua, o_bvh, frame_time=__Vars.extractor.desired_frame_time)
    #
    # model.train()
    # o_bvh.to_file(output_path)


@torch.no_grad()
def mode_transfer(x_bvh: bvh.parser.BVH, s_bvh: bvh.parser.BVH, output_path,
                  device, dataset: MoDeDataset, model: MoDeGenerativeModel):

    class __Vars:
        extractor = UniBVHDataExtractor(g_desired_frame_time)
        processor = STSimpleProcessor()
        # divider = UniMotionDataDivider(g_window_size, g_window_step, g_window_skip)

    model.eval()

    x_static, x_dynamic = __Vars.extractor.extract(x_bvh)
    s_static, s_dynamic = __Vars.extractor.extract(s_bvh)

    x_static =  __Vars.processor.f_process_static(0, *x_static)
    s_static =  __Vars.processor.f_process_static(0, *s_static)
    x_dynamic = __Vars.processor.f_process_dynamic(0, *x_dynamic)
    s_dynamic = __Vars.processor.f_process_dynamic(0, *s_dynamic)

    # x_dynamic = tuple([__Vars.divider.divide(e)[0] for e in x_dynamic])
    # s_dynamic = tuple([__Vars.divider.divide(e)[0] for e in s_dynamic])

    x_dyn, x_pos, x_trs, x_yrt = dataset.c_dataset.to_input(x_static, x_dynamic)
    s_dyn, s_pos, s_trs, s_yrt = dataset.c_dataset.to_input(s_static, s_dynamic)

    def __pre_input(x):
        return x[..., :x.shape[-1] // 4 * 4].to(device)[None, ...]

    x_dyn = model.vec_to_2d(__pre_input(x_dyn))
    x_pos = __pre_input(x_pos)
    x_trs = __pre_input(x_trs)
    s_dyn = model.vec_to_2d(__pre_input(s_dyn))
    s_pos = __pre_input(s_pos)
    s_trs = __pre_input(s_trs)

    """
    perform style transfer
    """
    fx = model.net_f(x_dyn)
    es = model.net_e(s_dyn)
    o_dyn = model.net_d(fx, es)

    """
    the global part of the output is ignored, however for simplicity it is still outputted
    """
    o_qua = model.output_to_qua(o_dyn)
    o_qua, o_trs, o_yrt = o_qua[0], x_trs[0], x_yrt[0]
    o_qua, o_trs, o_yrt = align_frames(o_qua, o_trs, o_yrt)
    o_qua = revert_root_rotation(o_qua, o_yrt, g_hip)

    o_bvh = deepcopy(x_bvh)
    off = mot.bvh_casting.get_offsets_from_bvh(o_bvh)
    off *= g_output_scaling
    mot.bvh_casting.write_offsets_to_bvh(o_bvh, off)
    o_trs *= g_output_scaling
    mot.bvh_casting.write_quaternion_to_bvh_object(o_trs, o_qua, o_bvh, frame_time=__Vars.extractor.desired_frame_time)

    model.train()
    o_bvh.to_file(output_path)

