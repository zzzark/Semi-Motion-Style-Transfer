import torch
from os.path import join as pj
import bvh
import os


# -------------- device -------------- #
DEVICE = torch.device("cuda:0")
g_device = torch.device(DEVICE)
PARALLEL_DEVICE = None


# -------------- dataset -------------- #
G_DATA_POSTFIX = '.6d_w128_s32'
G_ROOT_PATH = "./m21_data"

g_content_dataset = G_ROOT_PATH + "/cmu"
g_style_dataset = G_ROOT_PATH + "/bfa8"
g_cmu_bvh = bvh.parser.BVH(G_ROOT_PATH + "/t.bvh")

g_num_workers = 0


# ------------------------------------------- #
g_J = 21  # joint count
g_hip = 0

g_l_uparm = 14
g_l_elbow = 15
g_l_hand = 16

g_r_uparm = 18
g_r_elbow = 19
g_r_hand = 20

g_l_upleg = 1
g_l_knee = 2
g_l_f_base = 3
g_l_foot = 4

g_r_upleg = 5
g_r_knee = 6
g_r_f_base = 7
g_r_foot = 8

g_head = 12
g_l_hip = 1
g_r_hip = 5
g_l_sho = 14
g_r_sho = 18


g_R = 6   # rotation feature size

g_cmu_p_index = g_cmu_bvh.dfs_parent()

assert g_J == len(g_cmu_bvh.dfs_index())
assert list(g_cmu_bvh.offset_data.values())[g_l_hip].name == 'LeftUpLeg'
assert list(g_cmu_bvh.offset_data.values())[g_r_hip].name == 'RightUpLeg'
assert list(g_cmu_bvh.offset_data.values())[g_l_sho].name == 'LeftArm'
assert list(g_cmu_bvh.offset_data.values())[g_r_sho].name == 'RightArm'


# -------------- output -------------- #
g_prj_out_name = './output'
g_watch_dir = pj(g_prj_out_name, 'watch')
g_checkpoint_dir = pj(g_prj_out_name, 'checkpoints')
g_test_dir = pj(g_prj_out_name, 'test')
os.makedirs(g_prj_out_name, exist_ok=True)

g_output_scaling = 1.0 / (0.056444 * 100.0)

# -------------- logging & saving frequency -------------- #
g_epoch_per_log = 1
g_epoch_per_watch = 5
g_epoch_per_save = 5
g_epoch_total = 25


# -------- data -------- #
g_desired_frame_time = 1/60.0
g_window_size = 128
g_window_step = 32
g_window_skip = 0
g_batch_size = 32


# -------- input features -------- #
g_c_feature_size = -1
g_s_feature_size = -1


# -------- optimizer -------- #
g_optim = 'RAdam'
g_param_lr = 1e-4
g_param_wd = 1e-4

g_use_ema = True


# -------- hyper-parameters -------- #
USE_ATN = True
ENC_NORM = 'in'
STYLE_CODE_POOL = 'max'
EE_LOSS_TYPE = 'pos'  # 'vel', 'pos', 'dist'  [NOTE] 'vel' is effected by frame rate
EE_LOSS_D = 'l1'  # 'l1', 'l2', 'sl1'

if USE_ATN:
    print(" [ATN] ")

# 1.0 / 100.0; loss ratio of cm / quaternion
g_pos_w_decay = 1.0 / 10.0

# ------ SIMPLE ------ #
g_w_con = 1.0   # (simple) after ST recon.  (given a same style)
g_w_e_con = 0.0 * g_pos_w_decay

g_w_t = 1.0      # (simple) style feature triplet loss
g_w_t_cyc = 0.0  # (simple) output style feature triplet loss (redundant)

g_w_rec = 0.00   # (complex) recon.

g_w_margin_1 = 5.0  # (simple) margin of g_w_t triplet loss


# ------ COMPLEX ------ #
g_w_c = 0.01    # (complex) after ST recon.  (given a style)
g_w_c_vel = 0.0
g_id_c = [i for i in g_cmu_bvh.dfs_index()]


g_ee = [g_l_hand, g_r_hand, g_l_foot, g_r_foot]
g_w_ee = 0.2  # * g_pos_w_decay  # (complex) end-effectors (hands and feet)

g_w_ds = 0.0   # (complex) diversity sensitive

g_w_s = 0.1    # (complex) style loss (triplet)
g_w_d = 0.0    # (complex) disentanglement loss (max 0)

g_w_margin_2 = 5.0  # (complex) margin of g_w_s triplet loss
