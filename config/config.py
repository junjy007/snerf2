from dataclass_wizard import YAMLWizard
from dataclasses import dataclass, field
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.nerfacc import ContractionType
import torch
import math
import dotenv

@dataclass
class CycleGANConfig(YAMLWizard):
    model:str="test"
    is_train:bool=False
    gpu_ids:list=field(default_factory=lambda:[0,]) # cycle gan need iterable
    checkpoints_dir:str="checkpoints"
    name:str="cyclegan"
    preprocess:str="resize_and_crop"
    model_suffix:str=""
    input_nc:int=3
    output_nc:int=3
    ngf:int=64
    netG:str="resnet_9blocks"
    norm:str="instance"
    no_dropout:bool=True
    init_type:str="normal"
    init_gain:float=0.02
    load_iter:int=0
    epoch:str="latest"
    verbose:bool=True

@dataclass
class NerfConfig(YAMLWizard):
    name:str = "default"
    device:str = "cuda:0"

    # scene config
    scene_contraction_type:ContractionType=ContractionType.AABB
    scene_aabb:list=\
        field(default_factory=lambda:[-1.5, -1.5, -1.5, 1.5, 1.5, 1.5])

    # render
    render_n_samples:int=1024 # probably max samples per ray
    render_step_size:float=0.
    cone_angle:float=0.

    # nerf model
    nerf:str="DVGO"
    canonical_grid_size_fine:list=\
        field(default_factory=lambda:[160, 160, 160])
    density_grid_dim:int=1
    k0_grid_dim:int=12
    xyz_encoder_degrees:int=5 # x, y, z -> 3 * [sin(5 x scales) + cos(...)]
    view_encoder_degrees:int=4 
    time_encoder_degrees:int=3
    rgb_net_hidden_dim:int=128
    # nerf-occupancy
    occupancy_grid_resolution:int=128
    alpha_threshold_after_warmingup:float=0.01
    warmingup_steps:int=1000
    
    # optimisation
    learning_rate: float=0.001
    max_iter_steps:int=30000
    evaluate_every_n_steps:int=5000

    lr_grid_density:float=1e-1
    lr_grid_k0:float=1e-1
    lr_rgb_net:float=1e-3
    lr_particle_signature:float=1e-1
    lr_particle_net:float=1e-3

    target_sample_batch_size:int=1<<16

    # misc
    seed:int = 42


    # data
    data_root_dir:str=""
    data_dir:str="nerf/nerf_synthetic"

    def convert_parameters(self):
        # Some parameters are set at runtime, does not support yaml 
        # serialisation or should not (e.g. datadir). 
        # Convert them to tensor before using.
        self.scene_aabb = torch.tensor(self.scene_aabb, 
            dtype=torch.float32, device=self.device)
        # aabb-box edges take max * sqrt(3) to set up ray tracing step
        max_edge_len = (self.scene_aabb[3:] - self.scene_aabb[:3])\
            .max().item()
        self.render_step_size = \
            max_edge_len * math.sqrt(3) / self.render_n_samples 

        if self.data_root_dir == "":
            dotenv.load_dotenv()
            self.data_root_dir = os.environ["DATADIR"]
        
        if not self.data_dir.startswith('/'):
            self.data_dir = os.path.join(self.data_root_dir, self.data_dir)

        return self

    
@dataclass
class Config(YAMLWizard):
    device:str="cuda:0"
    seed:int=42
    cyclegan_cfg:CycleGANConfig=None
    nerf_cfg:NerfConfig=None

    def __post_init__(self):
        if self.cyclegan_cfg is None:
            self.cyclegan_cfg = CycleGANConfig()
        if self.nerf_cfg is None:
            self.nerf_cfg = NerfConfig(
                device=self.device,
                seed=self.seed)

    def convert_parameters(self):
        self.nerf_cfg = self.nerf_cfg.convert_parameters()
        return self
    
if __name__ == "__main__":
    cfg = Config()
    d = os.path.dirname(__file__)
    cfg.to_yaml_file(f"{d}/yamls/default.yaml")

# 
# class BaseOptions():
#     """This class defines options used during both training and test time.
# 
#     It also implements several helper functions such as parsing, printing, and saving the options.
#     It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
#     """
# 
#     def __init__(self):
#         """Reset the class; indicates the class hasn't been initailized"""
#         self.initialized = False
# 
#     def initialize(self, parser):
#         """Define the common options that are used in both training and test."""
#         # basic parameters
#         parser.add_argument('--dataroot', default='/home/hannah/win_d/data', help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
#         parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
#         parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
#         parser.add_argument('--checkpoints_dir', type=str, default='./cyclegan/checkpoints', help='models are saved here')
#         # model parameters
#         parser.add_argument('--model', type=str, default='test', help='chooses which model to use. [cycle_gan | pix2pix | test | colorization]')
#         parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels: 3 for RGB and 1 for grayscale')
#         parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels: 3 for RGB and 1 for grayscale')
#         parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
#         parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
#         parser.add_argument('--netD', type=str, default='basic', help='specify discriminator architecture [basic | n_layers | pixel]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
#         parser.add_argument('--netG', type=str, default='resnet_9blocks', help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')
#         parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
#         parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization [instance | batch | none]')
#         parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
#         parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
#         parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
#         # dataset parameters
#         parser.add_argument('--dataset_mode', type=str, default='unaligned', help='chooses how datasets are loaded. [unaligned | aligned | single | colorization]')
#         parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
#         parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
#         parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
#         parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
#         parser.add_argument('--load_size', type=int, default=286, help='scale images to this size')
#         parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
#         parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
#         parser.add_argument('--preprocess', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
#         parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
#         parser.add_argument('--display_winsize', type=int, default=256, help='display window size for both visdom and HTML')
#         # additional parameters
#         parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
#         parser.add_argument('--load_iter', type=int, default='0', help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]')
#         parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
#         parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')
#         # wandb parameters
#         parser.add_argument('--use_wandb', action='store_true', help='if specified, then init wandb logging')
#         parser.add_argument('--wandb_project_name', type=str, default='CycleGAN-and-pix2pix', help='specify wandb project name')
#         parser.add_argument(
#         "--train_split",
#         type=str,
#         default="trainval",
#         choices=["train", "trainval"],
#         help="which train split to use",
#     )
#         parser.add_argument(
#             "--scene",
#             type=str,
#             default="lego",
#             choices=[
#                 # nerf synthetic
#                 "chair",
#                 "drums",
#                 "ficus",
#                 "hotdog",
#                 "lego",
#                 "materials",
#                 "mic",
#                 "ship",
#                 # mipnerf360 unbounded
#                 "garden",
#             ],
#             help="which scene to use",
#         )
#         parser.add_argument(
#             "--aabb",
#             type=lambda s: [float(item) for item in s.split(",")],
#             default="-1.5,-1.5,-1.5,1.5,1.5,1.5",
#             help="delimited list input",
#         )
#         parser.add_argument(
#             "--test_chunk_size",
#             type=int,
#             default=1024,
#         )
#         parser.add_argument(
#             "--unbounded",
#             action="store_true",
#             help="whether to use unbounded rendering",
#         )
#         parser.add_argument("--cone_angle", type=float, default=0.0)
#         self.initialized = True
#         return parser
# 
#     def gather_options(self):
#         """Initialize our parser with basic options(only once).
#         Add additional model-specific and dataset-specific options.
#         These options are defined in the <modify_commandline_options> function
#         in model and dataset classes.
#         """
#         if not self.initialized:  # check if it has been initialized
#             parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#             parser = self.initialize(parser)
# 
#         # get the basic options
#         opt, _ = parser.parse_known_args()
# 
#         # modify model-related parser options
#         model_name = opt.model
#         model_option_setter = models.get_option_setter(model_name)
#         parser = model_option_setter(parser, self.is_train)
#         opt, _ = parser.parse_known_args()  # parse again with new defaults
# 
#         # modify dataset-related parser options
#         dataset_name = opt.dataset_mode
#         dataset_option_setter = data.get_option_setter(dataset_name)
#         parser = dataset_option_setter(parser, self.is_train)
# 
#         # save and return the parser
#         self.parser = parser
#         return parser.parse_args()
# 
#     def print_options(self, opt):
#         """Print and save options
# 
#         It will print both current options and default values(if different).
#         It will save options into a text file / [checkpoints_dir] / opt.txt
#         """
#         message = ''
#         message += '----------------- Options ---------------\n'
#         for k, v in sorted(vars(opt).items()):
#             comment = ''
#             default = self.parser.get_default(k)
#             if v != default:
#                 comment = '\t[default: %s]' % str(default)
#             message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
#         message += '----------------- End -------------------'
#         print(message)
# 
#         # save to the disk
#         expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
#         util.mkdirs(expr_dir)
#         file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
#         with open(file_name, 'wt') as opt_file:
#             opt_file.write(message)
#             opt_file.write('\n')
# 
#     def parse(self):
#         """Parse our options, create checkpoints directory suffix, and set up gpu device."""
#         opt = self.gather_options()
#         opt.is_train = self.is_train   # train or test
# 
#         # process opt.suffix
#         if opt.suffix:
#             suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
#             opt.name = opt.name + suffix
# 
#         self.print_options(opt)
# 
#         # set gpu ids
#         str_ids = opt.gpu_ids.split(',')
#         opt.gpu_ids = []
#         for str_id in str_ids:
#             id = int(str_id)
#             if id >= 0:
#                 opt.gpu_ids.append(id)
#         if len(opt.gpu_ids) > 0:
#             torch.cuda.set_device(opt.gpu_ids[0])
# 
#         self.opt = opt
#         return self.opt
# 
# class TestOptions(BaseOptions):
#     """This class includes test options.
# 
#     It also includes shared options defined in BaseOptions.
#     """
# 
#     def initialize(self, parser):
#         parser = BaseOptions.initialize(self, parser)  # define shared options
#         parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
#         parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
#         parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
#         # Dropout and Batchnorm has different behavioir during training and test.
#         parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
#         parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')
#         # rewrite devalue values
#         parser.set_defaults(model='test')
#         # To avoid cropping, the load_size should be the same as crop_size
#         parser.set_defaults(load_size=parser.get_default('crop_size'))
#         self.is_train = False
#         return parser
# 