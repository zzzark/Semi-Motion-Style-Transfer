import math

import bvh
from bvh.parser import BVH
from motion_tensor.bvh_casting import *


def fix_foot_sliding(src: bvh.parser.BVH, ref: bvh.parser.BVH,
                     src_ee_head, src_ee_lf, src_ee_rf,
                     ref_ee_head, ref_ee_lf, ref_ee_rf,
                     kernel_size=7, interp_length=13, floor_log_a=1.0,
                     algorithm='FABRIK', silence=False):
    """
    :param src: in / out
    :param ref: reference for extracting fc
    :param src_ee_head:
    :param src_ee_lf:
    :param src_ee_rf:
    :param ref_ee_head:
    :param ref_ee_lf:
    :param ref_ee_rf:
    :param kernel_size:
    :param interp_length:
    :param floor_log_a: log_e{a}
    :param algorithm: GRAD or FABRIK
    :param silence:
    :return:
    """
    from motion_tensor.motion_process import get_foot_contact
    from motion_tensor.kinematics import get_ik_target_pos, inverse_kinematics_grad, inverse_kinematics_fabrik

    if src.frames != ref.frames:
        print(f"[WARNING] src and ref frames not match: {src.frames} - {ref.frames}")

    T = min(src.frames, ref.frames)

    # ---- get from ref ---- #
    ref_hei = get_height_from_bvh(ref, ref_ee_head, ref_ee_lf)
    ref_pos = get_positions_from_bvh(ref, True)[..., :T]
    fc = get_foot_contact(ref_pos, [ref_ee_lf, ref_ee_rf], ref_hei, kernel_size=kernel_size)

    # ---- get from src ---- #
    ee_ids = [src_ee_lf, src_ee_rf]
    p_index = src.dfs_parent()
    pos = get_positions_from_bvh(src, True)[..., :T]
    tg_pos = get_ik_target_pos(fc, pos, ee_ids, interp_length=interp_length)

    off = get_offsets_from_bvh(src)
    trs, qua = get_quaternion_from_bvh(src)
    trs, qua = trs[..., :T], qua[..., :T]
    hei = get_height_from_bvh(src, src_ee_head, src_ee_lf)

    # get floor
    f_heights = torch.concat((pos[src_ee_lf, 1, fc[0]==1], pos[src_ee_rf, 1, fc[1]==1]))
    f_heights = f_heights * (torch.log(1.0 + (f_heights.abs() / hei)) / floor_log_a)
    feet_height = torch.mean(f_heights).item()
    trs[:, 1, :] -= feet_height

    if algorithm == "GRAD":
        ik_trs, ik_qua = inverse_kinematics_grad(p_index, off, trs, qua, tg_pos, ee_ids, hei, silence=silence)
    elif algorithm == "FABRIK":
        ik_trs, ik_qua = inverse_kinematics_fabrik(p_index, off, trs, qua, tg_pos, ee_ids, hei, silence=silence)
    else:
        raise NotImplementedError(algorithm)

    write_quaternion_to_bvh_object(ik_trs, ik_qua, src)
    return src


def find_joint_index(obj: BVH, name: str):
    for i, n in enumerate(obj.offset_data.keys()):
        if n == name:
            return i
    raise KeyError(f"no joint name matched to {name}")


def remove_fs(src_file, ref_file, dst_file, kernel_size=7, interp_length=13, silence=False):
    src = BVH(src_file)
    ref = BVH(ref_file)
    src_ee_head = find_joint_index(src, "Head")
    src_ee_lf = find_joint_index(src, "LeftToeBase")
    src_ee_rf = find_joint_index(src, "RightToeBase")
    ref_ee_head = find_joint_index(ref, "Head")
    ref_ee_lf = find_joint_index(ref, "LeftToeBase")
    ref_ee_rf = find_joint_index(ref, "RightToeBase")
    dst = fix_foot_sliding(src, ref,
                           src_ee_head, src_ee_lf, src_ee_rf,
                           ref_ee_head, ref_ee_lf, ref_ee_rf,
                           kernel_size=kernel_size, interp_length=interp_length, silence=silence)
    dst.to_file(dst_file)


def main():
    import os
    from os.path import join as pj

    os.makedirs("./Output_fs", exist_ok=True)
    root = r"D:\_Projects\FirstMotion\Output_MoDe-2023-7-27-GOOD-vel\watch"
    ep = "000025"
    src_file = pj(pj(root, ep), "OLD_box 3.bvh")
    ref_file = pj(pj(root, ep), "_C_box 3.bvh")
    dst_file = "./Output_fs/out.bvh"

    remove_fs(src_file, ref_file, dst_file)


if __name__ == '__main__':
    main()

