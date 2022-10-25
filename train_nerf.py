from config.config import Config
from argparse import ArgumentParser
from utils.nerf_utils import set_random_seed
from models.snerf2 import DVGO
from models.nerfacc import OccupancyGrid

def get_dataset_loader(cfg:Config):
    ddir = cfg.data_dir
    ddir = ddir.strip('/')
    if ddir.endswith("d_nerf"):
        print("Loading DNerf dataset")
        from datasets.dnerf_synthetic import SubjectLoader
        LoaderClass = SubjectLoader
    elif ddir.endswith("nerf_synthetic"):
        print("Loading synthetic_nerf dataset")
        from datasets.nerf_synthetic import SubjectLoader
        LoaderClass = SubjectLoader
    return LoaderClass

def get_nerf_dataset(cfg:Config, scene:str="", split:str="train"):
    LoaderClass = get_dataset_loader(cfg)
    nerf_dataset = LoaderClass(
        subject_id=scene,
        root_fp=cfg.data_dir,
        split=split,
        batch_over_images=False,
        num_rays=cfg.target_sample_batch_size // cfg.render_n_samples,
    )
    nerf_dataset.images = nerf_dataset.images.to(cfg.device)
    nerf_dataset.camtoworlds = nerf_dataset.camtoworlds.to(cfg.device)
    nerf_dataset.K = nerf_dataset.K.to(cfg.device)
    #nerf_dataset.timestamps = nerf_dataset.timestamps.to(cfg.device)
    return nerf_dataset

def get_nerf(cfg:Config):
    if cfg.nerf == "DVGO":
        nerf = DVGO(cfg).to(cfg.device)
    else:
        raise ValueError(f"Unsupported Nerf {cfg.nerf}")
    return nerf

def get_train_step(cfg:Config):
    set_random_seed(cfg.seed)
    # ==== SETUP NERF ====
    cfg_backup = cfg
    cfg = cfg.nerf_cfg
    # setup nerf
    nerf = get_nerf(cfg)
    # TODO:unify settings in OccupancyGrid
    occupancy_grid = OccupancyGrid(
        roi_aabb=cfg.scene_aabb,
        resolution=cfg.occupancy_grid_resolution,
        contraction_type=cfg.scene_contraction_type
    ).to(cfg.device)

    # ==== SETUP CYCLEGAN ===
    cfg = cfg_backup.cyclegan_cfg
    # ...

    def step():
        return 
    return step

#### MAIN-ENTRY ####
parser = ArgumentParser()
parser.add_argument("--scene", type=str, default="lego")
parser.add_argument("--cfg-file", type=str, default="config/yamls/default.yaml")
args = parser.parse_args()
cfg = Config().from_yaml_file(args.cfg_file)
cfg = cfg.convert_parameters()
print(cfg)


# data 
print("Scene", args.scene)
train_dataset = get_nerf_dataset(cfg.nerf_cfg, args.scene, split="train")
test_dataset = get_nerf_dataset(cfg.nerf_cfg, args.scene, split="test")
train_batch_num = len(train_dataset)
print("Setting up nerf model ...")
train_step = get_train_step(cfg)