import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch_scatter import scatter_add
from torch_geometric.nn import knn
from config.config import Config

def knn_interpolate(
    x: torch.Tensor, 
    pos_x: torch.Tensor, 
    pos_y: torch.Tensor, k: int = 3, 
    interpolate_method:int = 0, cut_off_th:float = 0):
    """ refer to torch_geometric.nn.unpool.knn_interpolate
        we let knn weights participate in gradient computation
        INPUT
            x: features attached to reference nodes to be interpolated
            pos_x: location of reference nodes, shoule be the particle@t
            pos_y: location of the query points
            K: #. nearest neighbours
            interpolate_method: 0, original from torch_geometric
                                1, softmax distance with cut-off static 
                                (when no dynamic particles is nereby, turn to static)
    """
    with torch.no_grad():
        assign_index = knn(pos_x, pos_y, k, batch_x=None, batch_y=None,
                            num_workers=1)
        y_idx, x_idx = assign_index[0], assign_index[1]
    diff = pos_x[x_idx] - pos_y[y_idx]
    squared_distance = (diff * diff).sum(dim=-1, keepdim=True)

    if interpolate_method == 0:
        weights = 1.0 / torch.clamp(squared_distance, min=1e-16)
        y = scatter_add(x[x_idx] * weights, y_idx, dim=0, dim_size=pos_y.size(0))
        y = y / scatter_add(weights, y_idx, dim=0, dim_size=pos_y.size(0))
        return y

    elif interpolate_method == 1:
        squared_distance_k = squared_distance.view(-1, k)
        # weights = torch.softmax(-squared_distance_k, dim=-1).view(-1, 1)
        # y = scatter_add(x[x_idx] * weights, y_idx, dim=0, dim_size=pos_y.size(0))
        weights = 1.0 / torch.clamp(squared_distance, min=1e-16)
        y = scatter_add(x[x_idx] * weights, y_idx, dim=0, dim_size=pos_y.size(0))
        y = y / scatter_add(weights, y_idx, dim=0, dim_size=pos_y.size(0))

        # cut-off static: All neighbours exceeded the threshold
        y_staic_mask = (squared_distance_k > cut_off_th**2).all(dim=-1)

        dynamic_rate = 1.
        if y_staic_mask.any():
            dynamic_rate = 1 - y_staic_mask.sum()/y_staic_mask.numel()

        y[y_staic_mask] = pos_y[y_staic_mask]
        return y, dynamic_rate
    else:
        raise NotImplementedError(
            f'interpolate_method={interpolate_method} is not supported.')

class DVGO(nn.Module):
    def __init__(self, 
        cfg:Config):
        super(DVGO, self).__init__()
        self.grid_size = cfg.canonical_grid_size_fine
        self.xyz_min = cfg.scene_aabb[:3]
        self.xyz_max = cfg.scene_aabb[3:]
        self.xyz_enc  = SinusoidalEncoder(3, 0, cfg.xyz_encoder_degrees, True)
        self.view_enc = SinusoidalEncoder(3, 0, cfg.view_encoder_degrees, True)
        # TODO: the shape variables are names only here, 
        # but to be clear which grid dim is the fastest changing one
        gx, gy, gz = self.grid_size
        self.density_grid_dim = cfg.density_grid_dim # e.g. 1
        self.k0_grid_dim = cfg.k0_grid_dim # e.g. 12

        self.grid_density = nn.Parameter(
            torch.randn([1, self.density_grid_dim, gx, gy, gz]))
        self.grid_k0 = nn.Parameter(
            torch.randn([1, self.k0_grid_dim, gx, gy, gz]))
        rgb_net_input_dim = self.k0_grid_dim \
            + self.view_enc.latent_dim\
            + self.xyz_enc.latent_dim
        rd = cfg.rgb_net_hidden_dim
        self.rgb_net = nn.Sequential(
                nn.Linear(
                    rgb_net_input_dim,
                    rd),  # grid + dir + xyz'
                nn.ReLU(inplace=True),
                nn.Linear(rd, rd),
                nn.ReLU(inplace=True),
                nn.Linear(rd, 3),
            )

        # TODO: check difference from DVGO cuda implementation (Raw2Alpha)
        # also known as 'shifted softplus' mentioned in Mip-NeRF
        self.softplus = torch.nn.Softplus()

    def forward_grid(self, xyz, grid):
        xyz = xyz.reshape(1, 1, 1, -1, 3) # [1, F, cx, cy, cz] ->
        
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1
        out = F.grid_sample(grid, ind_norm, mode='bilinear', align_corners=True)
        out = out.view(out.shape[1], out.shape[-1]).transpose(0, 1)
        return out

    def query_opacity(self, x, step_size):
        density = self.query_density(x)
        # if the density is small enough those two are the same.
        # opacity = 1.0 - torch.exp(-density * step_size)
        opacity = density * step_size
        return opacity

    def query_density(self, x):
        """
            x[N, 3]
        """
        density = self.softplus(
            self.forward_grid(x, self.grid_density)
        )  #  + self.act_shift
        return density

    def query_cfeature(self, x):
        cfeature = self.forward_grid(x, self.grid_k0)
        return cfeature

    def query_cfeat2clr(self, x, cfeat, view_dir):
        """
        INPUT
            x: position
            cfeat: #.queries x k0_grid_dim
            view_dir: #.queries x (direction-rep)
        RETURN
            #.queries x 3 (rgb)
        """
        rgb_code = torch.cat([
            cfeat, 
            self.view_enc(view_dir), 
            self.xyz_enc(x)], dim=-1)
        rgb = torch.sigmoid(self.rgb_net(rgb_code))
        return rgb

    def forward(self, x, view_dir):
        """
            x[N, 3], view_dir[N, 3]
        """
        density = self.query_density(x)
        cfeature = self.query_cfeature(x)
        rgb = self.query_cfeat2clr(x, cfeature, view_dir)
                
        return rgb, density

class DNeRFParticle(nn.Module):
    def __init__(self, cfg:Config):
        super(DNeRFParticle, self).__init__()
        self.canonical_nerf = DVGO(cfg)
        self.time_enc = SinusoidalEncoder(1, 0, cfg.time_encoder_degrees, 
            True, sin_only=True)
        self.xyz_min = cfg.scene_aabb[:3]; xyz_min = self.xyz_min
        self.xyz_max = cfg.scene_aabb[3:]; xyz_max = self.xyz_max
        self.scale = self.xyz_max - self.xyz_min

        l = torch.linspace
        particle_num_base = 11
        x, y, z = torch.meshgrid(
            l(xyz_min[0], xyz_max[0], particle_num_base), 
            l(xyz_min[1], xyz_max[1], particle_num_base), 
            l(xyz_min[2], xyz_max[2], particle_num_base), indexing='ij')
        particle_xyz0 = torch.stack(
                [x.flatten(), y.flatten(), z.flatten()]
            ).T.to(torch.float32).to(cfg.device)
        
        particle_dim = 16
        particle_num = particle_xyz0.shape[0]  # ~1k particle
        self.particle_signature = nn.Parameter(
            torch.randn(particle_num, particle_dim)*0.01)

        # particle-to-"temporal feature", 
        # <"temporal feature", time code> makes trajectories
        self.particle_net = nn.Sequential(
                nn.Linear(particle_dim, 64, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(64, 64, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(64, 64, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(64, 3 * self.time_enc.latent_dim, bias=False),
        )
        
        # searching neighbor particles
        k = 8
        
        # to determine static querying points
        cut_off_th = 1. * (xyz_max-xyz_min).max()/(particle_num_base-1.)

        """ additional stuff for checkpoint """
        self.register_buffer('particle_xyz0', particle_xyz0)
        # TODO: the meta-information can be saved statically
        # self.register_buffer('xyz_min', xyz_min)
        # self.register_buffer('xyz_max', xyz_max)
        self.register_buffer('can_grid_size', torch.tensor(cfg.canonical_grid_size_fine))
        self.register_buffer('k', torch.tensor(k))
        self.register_buffer('cut_off_th', cut_off_th)

        # for logging
        self.dynamic_rate = 1.

    def get_particle_xyz_at(self, t):
        # INPUT
        #   t: [1, 1] 
        #      or [N_sample, 1] auto expand by 'render_image()'
        assert ((t - t.mean()) < 1e-6).all(), "timestamps in batch must be the same."
        t = t.mean().view(1, 1)
        time_code = self.time_enc(t)
        pcode = self.particle_net(self.particle_signature)
        pcode = pcode.view(-1, 3, time_code.shape[-1])
        
        # [#.particles, 3] <- sum([#.particles, 3, TD] * [1, TD])
        dxyz = (pcode * time_code).sum(-1) # inner prod
        # dxyz = torch.tanh(dxyz) * self.scale  # TODO: smooth the movement

        return dxyz + self.particle_xyz0

    def query_opacity(self, x, timestamps, step_size):
        # random time during training
        t = timestamps[0]
        density = self.query_density(x, t)
        # if the density is small enough those two are the same.
        # opacity = 1.0 - torch.exp(-density * step_size)
        opacity = density * step_size
        return opacity

    def query_density(self, x, t):
        """
            x[N, 3]
            t[N, 1] values are the same
        """
        if x.shape[0] == 0:
            return x[:, :1]
        particle_xyz = self.get_particle_xyz_at(t)
        # query_xyz0: the query points position at t0
        query_xyz0, self.dynamic_rate = knn_interpolate(
            self.particle_xyz0, particle_xyz, x, k=self.k.item(), 
            interpolate_method=1, cut_off_th=self.cut_off_th)

        return self.canonical_nerf.query_density(query_xyz0)

    def forward(self, x, t, view_dir):
        """
            x[N, 3], view_dir[N, 3]
            t[N, 1] values are the same
        """
        if x.shape[0] == 0:
            return x, x[:, :1]

        particle_xyz = self.get_particle_xyz_at(t)
        # query_xyz0: the query points position at t0

        query_xyz0, self.dynamic_rate = knn_interpolate(
            self.particle_xyz0, particle_xyz, x, k=self.k.item(),
            interpolate_method=1, cut_off_th=self.cut_off_th)
        rgb, density = self.canonical_nerf(query_xyz0, view_dir)
        return rgb, density

class SinusoidalEncoder(nn.Module):
    """Sinusoidal Positional Encoder used in Nerf."""

    def __init__(self, x_dim, min_deg, max_deg, 
        use_identity: bool = True, sin_only=False):
        super().__init__()
        self.x_dim = x_dim
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.use_identity = use_identity
        self.sin_only = sin_only
        self.register_buffer(
            "scales", torch.tensor([2**i for i in range(min_deg, max_deg)])
        )

    @property
    def latent_dim(self) -> int:
        return (
            int(self.use_identity) + (self.max_deg - self.min_deg) * 2
        ) * self.x_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., x_dim]
        Returns:
            latent: [..., latent_dim]
        """
        if self.max_deg == self.min_deg:
            return x
        xb = torch.reshape(
            (x[Ellipsis, None, :] * self.scales[:, None]),
            list(x.shape[:-1]) + [(self.max_deg - self.min_deg) * self.x_dim],
        )
        if self.sin_only:
            df = 1.0
        else:
            df = 0.5
        latent = torch.sin(torch.cat([xb, xb + df * math.pi], dim=-1))
        if self.use_identity:
            latent = torch.cat([x] + [latent], dim=-1)
        return latent