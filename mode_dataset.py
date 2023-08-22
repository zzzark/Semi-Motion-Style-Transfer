from motion_tensor.dataset import *
import motion_tensor.motion_process as mop
from mode_config import *
import random
import motion_tensor as mot


class UniBVHDataExtractor(BVHDataExtractor):
    def __init__(self, desired_frame_time):
        super(UniBVHDataExtractor, self).__init__(desired_frame_time)

    @staticmethod
    def _length_to_root(off, j_idx):
        """
        sum of length of each body part from j_idx to g_hip
        """
        length = 0

        while j_idx != g_hip:
            off_j = off[j_idx]
            length += ((off_j ** 2).sum() ** 0.5).item()
            j_idx = g_cmu_p_index[j_idx]  # use cmu topology, okay with that

        return length

    @staticmethod
    def _get_ki_height(off) -> float:
        """
        get length along kinematic chain
        """
        return UniBVHDataExtractor._length_to_root(off, g_head) + UniBVHDataExtractor._length_to_root(off, g_l_foot)

    @staticmethod
    def _get_t_height(tps) -> float:
        """
        get t-pose height
        """
        return tps[g_head][1].item() - tps[g_l_foot][1].item()  # y-axis

    @staticmethod
    def _get_leg_length(tps) -> float:
        return tps[g_hip][1].item() - tps[g_l_foot][1].item()  # y-axis

    def extract(self, bvh_obj: bvh.parser.BVH) -> Tuple[tuple, tuple]:
        off = casting.get_offsets_from_bvh(bvh_obj)
        tps = casting.get_t_pose_from_bvh(bvh_obj)

        kin_hei = UniBVHDataExtractor._get_ki_height(off=off)
        leg_hei = UniBVHDataExtractor._get_leg_length(tps=tps)
        tps_hei = UniBVHDataExtractor._get_t_height(tps=tps)

        trs, qua = casting.get_quaternion_from_bvh(bvh_obj)

        # the position are not aligned to z-axis
        pos = casting.get_positions_from_bvh(bvh_obj, locomotion=False)  # DO NOT USE GLOBAL POSITION

        trs = self.scale(trs, bvh_obj.frame_time)
        qua = self.scale(qua, bvh_obj.frame_time)
        pos = self.scale(pos, bvh_obj.frame_time)

        rrt, yrt = mop.align_root_rot(pos, qua[g_hip], (g_l_hip, g_r_hip), (g_l_sho, g_r_sho))
        qua[g_hip] = rrt

        # aligned to z-axis
        mat = mot.rotations.quaternion_to_matrix(qua)
        pos = mot.kinematics.forward_kinematics(g_cmu_p_index, mat, None, off, True, False)
        r6d = mot.rotations.matrix_to_rotation_6d(mat)

        return (off, kin_hei, leg_hei, tps_hei), (trs, pos, r6d, yrt)


class UniMotionDataDivider(MotionDataDivider):
    def __init__(self, window, window_step, skip):
        super(UniMotionDataDivider, self).__init__(window, window_step, skip)

    @staticmethod
    def _pad(x):
        m = torch.flip(x, dims=(-1, ))[..., 1:]  # remove one overlapped frame
        return torch.concat([x, m], dim=-1)

    def divide(self, motion: torch.Tensor) -> List[torch.Tensor]:
        K, S, W = self.skip, self.window_step, self.window
        motion = motion[..., K:]
        clip_list = []
        total = motion.shape[-1]

        for j in range(0, total, S):
            mo_clip = motion[..., j: j + W].clone()
            # while W // 2 < mo_clip.shape[-1] < W:
            if W // 2 < mo_clip.shape[-1] < W:
                mo_clip = self._pad(mo_clip)
                mo_clip = mo_clip[..., :W]
            elif mo_clip.shape[-1] <= W // 2:
                break
            assert mo_clip.shape[-1] == W, mo_clip.shape[-1]
            clip_list.append(mo_clip)

        return clip_list


class STSimpleProcessor(MoClipProcessor):
    def __init__(self):
        super(STSimpleProcessor, self).__init__()

    def f_process_static(self, class_id, *args) -> tuple:
        return ()

    def f_process_dynamic(self, class_id, *args) -> tuple:
        (trs, pos, r6d, yrt) = args

        # New Feature
        dyn = r6d.view(-1, r6d.shape[-1])
        pos = pos.view(-1, pos.shape[-1])

        return dyn, pos, trs, yrt


class STMeanVarCollector(MoStatisticCollector):

    # noinspection PyMethodMayBeStatic
    def _mean_var_of_list(self, ls):
        m = torch.mean(torch.concat(ls, dim=-1), dim=(-1), keepdim=True)
        v = torch.var(torch.concat(ls,  dim=-1), dim=(-1), keepdim=True)
        v = v ** 0.5
        v[v < 1e-5] = 1.0
        return m, v

    def get_stat(self, class_id: int, feature: List[tuple]) -> Any:
        # dyn_ls = [e[0] for e in feature]
        # pos_ls = [e[1] for e in feature]
        #
        # dyn_m, dyn_v = self._mean_var_of_list(dyn_ls)
        # pos_m, pos_v = self._mean_var_of_list(pos_ls)
        #
        # return dyn_m, dyn_v, pos_m, pos_v

        # we do not use per-class statistics, so it just simply returns `None`
        return None

    def get_stat_all(self, feature) -> Any:
        dyn_ls = [e[0] for e in feature]
        pos_ls = [e[1] for e in feature]

        dyn_m, dyn_v = self._mean_var_of_list(dyn_ls)
        pos_m, pos_v = self._mean_var_of_list(pos_ls)

        return dyn_m, dyn_v, pos_m, pos_v


class STSimpleDataset(MoClipDataset):
    def __init__(self, cache_file_folder, meta, processor, enable_lazy_loading):
        super(STSimpleDataset, self).__init__(cache_file_folder, meta, processor, enable_lazy_loading)
        self.dyn_m = None
        self.dyn_v = None
        self.pos_m = None
        self.pos_v = None

        assert enable_lazy_loading is True  # set_mean_var first then load into cache later

        self._dif_class = {
            i: list(range(0, i)) + list(range(i+1, len(self._class_ids)))
            for i in range(len(self._class_ids))
        }

    def set_mean_var(self, dyn_m, dyn_v, pos_m, pos_v):
        self.dyn_m = dyn_m
        self.dyn_v = dyn_v
        self.pos_m = pos_m
        self.pos_v = pos_v

    def to_input(self, static: tuple, dynamic: tuple) -> tuple:
        dyn, pos, trs, yrt, _ = self._before_load_to_cache_memory(static, dynamic, 0)
        return dyn, pos, trs, yrt

    def _before_load_to_cache_memory(self, static: tuple, dynamic: tuple, class_id: int) -> tuple:
        """
        lazy loading before normalization so this should work
        """
        dyn, pos, trs, yrt = dynamic
        dyn = (dyn - self.dyn_m) / self.dyn_v
        # pos = (pos - self.pos_m) / self.pos_v

        return dyn, pos, trs, yrt, class_id

    def fetch_same(self, class_id):
        sam_index = random.choice(self._class_ids[class_id])
        return self.__getitem__(sam_index)

    def fetch_diff(self, class_id):
        diff_c_id = random.choice(self._dif_class[class_id])
        dif_index = random.choice(self._class_ids[diff_c_id])
        return self.__getitem__(dif_index)


def get_mixed_c_s_dataset(c_folder, s_folder, post_fix) -> [STSimpleDataset, STSimpleDataset]:
    extractor = UniBVHDataExtractor(g_desired_frame_time)
    divider = UniMotionDataDivider(g_window_size, g_window_step, g_window_skip)
    processor = STSimpleProcessor()
    collector = STMeanVarCollector()

    print('computing mean and var of `content` dataset ...')
    mv_dic = gather_statistic(s_folder, s_folder+post_fix, extractor, processor, collector)
    dic_per = mv_dic['per']
    lis_all = mv_dic['all']

    # # per-class to all-class
    # dyn_m = torch.mean(torch.stack([e[0] for e in dic_per.values()], dim=0), dim=0)
    # dyn_v = torch.mean(torch.stack([e[1] for e in dic_per.values()], dim=0), dim=0)  # approx
    # pos_m = torch.mean(torch.stack([e[2] for e in dic_per.values()], dim=0), dim=0)
    # pos_v = torch.mean(torch.stack([e[3] for e in dic_per.values()], dim=0), dim=0)  # approx
    dyn_m = lis_all[0]
    dyn_v = lis_all[1]
    pos_m = lis_all[2]
    pos_v = lis_all[3]

    print('caching `content` dataset ...')
    make_mo_clip_dataset(c_folder, c_folder+post_fix, divider, extractor)
    c_dataset: STSimpleDataset = load_mo_clip_dataset(c_folder+post_fix, processor, STSimpleDataset, True)
    c_dataset.set_mean_var(dyn_m, dyn_v, pos_m, pos_v)
    print(f'total clips of `content`: {len(c_dataset)}')

    print('caching `style` dataset ...')
    make_mo_clip_dataset(s_folder, s_folder+post_fix, divider, extractor)
    s_dataset: STSimpleDataset = load_mo_clip_dataset(s_folder+post_fix, processor, STSimpleDataset, True)
    s_dataset.set_mean_var(dyn_m, dyn_v, pos_m, pos_v)
    print(f'total clips of `style`: {len(s_dataset)}')

    return c_dataset, s_dataset
