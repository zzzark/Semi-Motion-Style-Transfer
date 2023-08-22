from bvh.parser import BVH
from nn_utils.uni_trainer import GenTrainer
from mode_model import MoDeDataset, MoDeGenerativeModel, mode_transfer, mode_reconstruct
from mode_config import *


def transfer(con_path: str, sty_path: str,
             dataset: MoDeDataset, trainer: GenTrainer,
             path):

    os.makedirs(path, exist_ok=True)

    c = con_path.split("/")[-1].split("\\")[-1][:-4]
    s = sty_path.split("/")[-1].split("\\")[-1][:-4]
    c_bvh = BVH(con_path)
    s_bvh = BVH(sty_path)

    mode_transfer(c_bvh, s_bvh, pj(path, f"{s}_{c}.bvh"), g_device, dataset, trainer.model)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--con", required=True)
    parser.add_argument("--sty", required=True)
    args = parser.parse_args()

    dataset = MoDeDataset(g_content_dataset, g_style_dataset, G_DATA_POSTFIX)
    trainer = GenTrainer(MoDeGenerativeModel(dataset), g_device, g_optim,
                         g_param_lr, g_param_wd, g_use_ema, PARALLEL_DEVICE)

    start_ep = max(trainer.resume(model_dir=g_checkpoint_dir, device=g_device), 0)
    print(f'{"start" if start_ep == 0 else "resume"} from epoch {start_ep}')

    transfer(args.con, args.sty, dataset, trainer,
             pj(g_prj_out_name, "original"))

    from fix_fs_folder import remove_fs_folder
    remove_fs_folder(pj(g_prj_out_name, "original"),
                     "./m21_data/content",
                     pj(g_prj_out_name, "remove_fs"))


if __name__ == '__main__':
    main()
