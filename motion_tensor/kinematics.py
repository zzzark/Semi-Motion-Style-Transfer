import torch
from motion_tensor.rotations import quaternion_to_matrix
from motion_tensor.rotations import quaternion_from_two_vectors
from motion_tensor.rotations import quaternion_rotate_vector_inv
from motion_tensor.rotations import mul_two_quaternions
from motion_tensor.motion_process import get_foot_contact_point


def forward_kinematics(parent_index: list, mat3x3: torch.Tensor,
                       root_pos: [torch.Tensor, None], offset: torch.Tensor,
                       world=True, is_edge=False):
    """
    implement forward kinematics in a batched manner
    :param parent_index: index of parents (-1 for no parent)
    :param mat3x3: rotation matrix, [(B), J, 3, 3, F] (batch_size x joint_num x 3 x 3 x frame_num)
    :param root_pos: root position [(B), 1, 3, F], None for Zero
    :param offset: joint offsets [(B), J, 3, F]
    :param world: world position or local position ?
    :param is_edge:
            True:  mat3x3[i] represents a rotation matrix of the parent of joint i,
                   i.e. edge rotation
            False: mat3x3[i] represents a rotation matrix of joint i,
                   i.e. joint rotation
    :return: tensor of positions in the shape of [(B), J, 3, F]
    """
    assert parent_index[0] == -1, f"the first parent index should be -1 (root), not {parent_index[0]}."

    if mat3x3.shape[-2] == 4:  # expected matrix, but got quaternion instead
        mat3x3 = quaternion_to_matrix(mat3x3)
        # warnings.warn("...")

    batch = len(mat3x3.shape) == 5
    if not batch:
        mat3x3 = mat3x3[None, ...]
        if root_pos is not None:
            root_pos = root_pos[None, ...]
        offset = offset[None, ...]

    assert len(mat3x3.shape) == 5
    assert len(offset.shape) == 4
    assert root_pos is None or len(root_pos.shape) == 4

    B, J, _, _, F = mat3x3.shape

    mat3x3 = mat3x3.permute(0, 4, 1, 2, 3)                  # mat:    [B, F, J, 3, 3]
    offset = offset.permute(0, 3, 1, 2)[..., None]          # offset: [B, F, J, 3, 1]
    if root_pos is not None:
        root_pos = root_pos.permute(0, 3, 1, 2)[..., None]  # root:   [B, F, 1, 3, 1]

    mat_mix = torch.empty_like(mat3x3, dtype=mat3x3.dtype, device=mat3x3.device)  # avoid in-place operation

    position = torch.empty((B, F, J, 3, 1), device=offset.device)  # [B, F, J, 3, 1]

    if root_pos is not None:
        position[..., 0, :, :] = root_pos[..., 0, :, :]
    else:
        position[..., 0, :, :].zero_()

    mat_mix[..., 0, :, :] = mat3x3[..., 0, :, :]
    for ci, pi in enumerate(parent_index[1:], 1):
        off_i = offset[..., ci, :, :]

        if not is_edge:
            # use .clone() to avoid using part of a tensor to compute its another part
            # otherwise we may get an inplace operation error
            mat_p = mat_mix[..., pi, :, :].clone()
            trs_i = torch.matmul(mat_p, off_i)
            position[..., ci, :, :] = trs_i
            mat_mix[..., ci, :, :] = torch.matmul(mat_p, mat3x3[..., ci, :, :])
        else:
            combo = torch.matmul(mat_mix[..., pi, :, :], mat3x3[..., ci, :, :])
            trs_i = torch.matmul(combo, off_i)
            position[..., ci, :, :] = trs_i
            mat_mix[..., ci, :, :] = combo

        if world:
            position[..., ci, :, :] += position[..., pi, :, :]

    position = position[..., 0].permute(0, 2, 3, 1)

    if not batch:
        position = position[0]

    return position


def inverse_kinematics_grad(p_index, off, trs, qua, tg_pos, ee_ids, height, silence=False,
                            iteration=50, vel_mul=0.2, qua_mul=0.01):
    """
    :param p_index:
    :param off: bone offsets
    :param trs: original root trs (update inplace)
    :param qua: original full qua (update inplace)
    :param tg_pos: full body target position
    :param ee_ids:
    :param height:
    :param silence:
    :param iteration:
    :param vel_mul: velocity constraint multiplier in loss function
    :param qua_mul: quaternion constraint multiplier in loss function
    :return:
    """
    tg_ee_pos = tg_pos[ee_ids]
    trs = trs.clone()
    qua = qua.clone()

    org_qua = qua.clone()

    trs.requires_grad = True
    qua.requires_grad = True

    # optim = torch.optim.Adam([trs, qua], lr=0.001, betas=(0.9, 0.9))
    optim = torch.optim.SGD([trs, qua], lr=0.003, momentum=0.9)
    l2 = torch.nn.MSELoss()

    for t in range(iteration):
        optim.zero_grad()

        m = quaternion_to_matrix(qua)
        ik_pos = forward_kinematics(p_index, m, trs, off, True, False)
        ik_ee_pos = ik_pos[ee_ids]

        ik_vel = ik_pos[..., :-1] - ik_pos[..., 1:]
        tg_vel = tg_pos[..., :-1] - tg_pos[..., 1:]
        ee_loss = l2(ik_ee_pos, tg_ee_pos)
        other_loss = vel_mul * l2(ik_vel, tg_vel) + qua_mul * height * l2(qua, org_qua)
        loss = ee_loss + other_loss

        loss.backward()
        optim.step()

        if not silence:
            print(f"ik loss {t+1}/{iteration}: {ee_loss.item() / height}")

    trs.requires_grad = False
    qua.requires_grad = False
    return trs, qua


def inverse_kinematics_fabrik(p_index, off, trs, qua, tg_pos, ee_ids, height, sin_lim=(0.4, 0.8),
                              silence=False, iteration=10, return_pos=False):
    """
    :param p_index:
    :param off: bone offsets
    :param trs: original root trs (update inplace)
    :param qua: original full qua (update inplace)
    :param tg_pos: full body target position
    :param ee_ids:
    :param height:
    :param sin_lim: tuple or list of tuple, rotation angle limit (sine) for each ee.
                    pos: foot base above toe; neg: foot base below toe
    :param silence:
    :param iteration:
    :param return_pos: return the target pos after ik
    :return: (trs, qua) or (trs, qua, pos)
    """
    if isinstance(sin_lim, tuple) and isinstance(sin_lim[0], float):
        sin_lim = [sin_lim for _ in range(len(ee_ids))]

    # og_pos = forward_kinematics(p_index, qua, trs, off, True, False)
    # ik_pos = og_pos.clone()

    ik_pos = forward_kinematics(p_index, qua, trs, off, True, False)
    trs = trs.clone()
    qua = qua.clone()

    def __bone_len(bn):
        return (((bn ** 2.0).sum(dim=-2)) ** 0.5)

    for t in range(iteration):
        for i, ee in enumerate(ee_ids):
            j = ee
            kin_chain = []  # kinematic chain
            while j > 0:  # TODO: take sub-base cases into consideration (e.g. hand base)
                kin_chain.append(j)
                j = p_index[j]

            # forward
            tg_j = tg_pos[ee].clone()
            for jc, jp in zip(kin_chain[:-1], kin_chain[1:]):  # (child, parent)
                if jc == ee:
                    d = ik_pos[jp] - ik_pos[jc]  # FIXME: direction is at the fixed direction
                    ik_pos[jc] = tg_j.clone()

                    # FIXME: the following code is only dedicated for handling feet
                    #        (it constraints the angle by simply compute the according y-dim value)
                    #        --
                    #        for removing foot sliding artifacts, sin_lim should be a negative value
                    #        for adapting to terrain height, sin_lim can be positive
                    #        --
                    #        we should move the code below to function `get_ik_target_pos`
                    #        and give the terrain info to fix foot sliding & terrain adaption issues
                    #        this requires our impl of FABRIK can handle multiple joint constraints
                    len_off = torch.linalg.vector_norm(off[jc], dim=0).item()
                    y_min, y_max = len_off * sin_lim[i][0], len_off * sin_lim[i][1]
                    y_d = d[1, :]
                    y_d[y_d < y_min] = y_min
                    y_d[y_d > y_max] = y_max
                    d[1, :] = y_d
                    d /= torch.linalg.vector_norm(d, dim=0)
                    d *= len_off
                else:
                    ik_pos[jc] = tg_j.clone()
                    d = ik_pos[jp] - ik_pos[jc]
                    d /= torch.linalg.vector_norm(d, dim=0)
                    len_off = torch.linalg.vector_norm(off[jc], dim=0).item()
                    d *= len_off

                tg_j += d

            # ... move root to target point && move back ...

            # backward
            kin_chain = kin_chain[::-1]
            tg_j = ik_pos[kin_chain[0]].clone()
            for jp, jc in zip(kin_chain[:-1], kin_chain[1:]):
                ik_pos[jp] = tg_j.clone()
                d = ik_pos[jc] - ik_pos[jp]

                len_off = torch.linalg.vector_norm(off[jc], dim=0).item()
                d /= torch.linalg.vector_norm(d, dim=0)
                d *= len_off
                tg_j += d

            # last joint
            ik_pos[ee] = tg_j

        if not silence:
            ee_loss = ((tg_pos[ee_ids] - ik_pos[ee_ids])**2.0).mean()  # MSELoss
            print(f"ik loss {t+1}/{iteration}: {ee_loss.item() / height}")

    # solve the joint rotations
    for ee in ee_ids:
        j = ee
        kin_chain = []  # kinematic chain
        while j >= 0:
            kin_chain.append(j)
            j = p_index[j]
        kin_chain = kin_chain[::-1]

        # id quaternion
        last_q = qua[kin_chain[0]]

        for jc, jp in zip(kin_chain[2:], kin_chain[1:]):
            v0 = off[jc]
            v1 = ik_pos[jc] - ik_pos[jp]
            v1 = quaternion_rotate_vector_inv(last_q, v1)  # last_q-1 * v * last_q
            qp = quaternion_from_two_vectors(v0, v1)
            qua[jp] = qp
            last_q = mul_two_quaternions(last_q, qp)

    return (trs, qua, ik_pos) if return_pos else (trs, qua)


def get_ik_target_pos(fc, org_pos, ee_ids, force_on_floor=True, interp_length=13):
    """
    :param fc: (target) foot contact labels
    :param org_pos: original full body position
    :param ee_ids:
    :param force_on_floor:
    :param interp_length:
    :return: full body target position
    """
    tg_ee_pos = get_foot_contact_point(org_pos[ee_ids], fc, force_on_floor, interp_length)  # ee target position
    tg_pos = org_pos.clone()
    tg_pos[ee_ids] = tg_ee_pos
    return tg_pos


def main():
    from motion_tensor.bvh_casting import get_quaternion_from_bvh
    from motion_tensor.bvh_casting import get_offsets_from_bvh
    from motion_tensor.bvh_casting import get_positions_from_bvh
    from motion_tensor.bvh_casting import get_height_from_bvh
    from motion_tensor.motion_process import get_foot_contact

    from bvh.parser import BVH

    from visualization.utils import quick_visualize as qv
    from visualization.utils import quick_visualize_fk as qv_fk

    bvh = BVH(R"D:\_datasets\Motion21\xia\0\0000.bvh")
    p_index = bvh.dfs_parent()
    ee_head = 12
    ee_lf, ee_rf = 4, 8
    ee_ids = [ee_lf, ee_rf]

    off = get_offsets_from_bvh(bvh)
    trs, qua = get_quaternion_from_bvh(bvh)
    pos = get_positions_from_bvh(bvh, True)
    hei = get_height_from_bvh(bvh, ee_head, ee_lf)
    print(f"height: {hei}")
    # qv(p_index, pos)

    fc = get_foot_contact(pos, ee_ids, hei)
    tg_pos = get_ik_target_pos(fc, pos, ee_ids)
    # qv(p_index, tg_pos)

    # print(" ========== GRAD ======== ")
    # ik_trs, ik_qua = inverse_kinematics_grad(p_index, off, trs, qua, tg_pos, ee_ids, hei)
    # ik_pos = forward_kinematics(p_index, ik_qua, ik_trs, off, True, False)
    # qv(p_index, ik_pos)

    print(" ========= FABRIK ========= ")
    # ik_pos = inverse_kinematics_fabrik(p_index, off, trs, qua, tg_pos, ee_ids, hei)
    ik_trs, ik_qua, ret_pos = inverse_kinematics_fabrik(p_index, off, trs, qua, tg_pos, ee_ids, hei, return_pos=True)
    ik_pos = forward_kinematics(p_index, ik_qua, ik_trs, off, True, False)

    # print(ret_pos.shape, ik_pos.shape)
    # ret_pos = ret_pos[..., ::6]
    # ik_pos = ik_pos[..., ::6]
    def __callback_1(*args):
        pass
    def __callback_2(*args):
        pass
    # qv(p_index, ret_pos, callback_fn=__callback_1)
    qv(p_index, ik_pos, callback_fn=__callback_2)


if __name__ == '__main__':
    main()
