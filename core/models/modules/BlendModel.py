import os
import pickle
import numpy as np
# Modified from smplx code for FLAME
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_rotation_6d
from skimage.io import imread
from loguru import logger

from yacs.config import CfgNode as CN

from pytorch3d.structures import Meshes
from pytorch3d.io import save_obj, load_obj

from pytorch3d.renderer import (
    FoVPerspectiveCameras, RasterizationSettings, MeshRenderer, MeshRasterizer, SoftPhongShader, PointLights
)

from pytorch3d.renderer import RasterizationSettings, PointLights, MeshRenderer, MeshRasterizer, TexturesVertex, SoftPhongShader, look_at_view_transform, PerspectiveCameras, BlendParams

from core.libs.flame_model.renderer_utils import RenderMesh

cfg = CN()
cfg.ict_face = 'face.npy'
cfg.ict_exp_model = 'blend_exp.npz'

cfg.bs = './flame/bs/exp/'

blend_name = ['Basis', 'browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight', 
              'eyeSquintLeft', 'eyeSquintRight', 'eyeWideLeft', 'eyeWideRight', 'jawForward', 'jawLeft',
              'jawRight', 'mouthFrownLeft', 'mouthFrownRight', 'mouthPucker', 'mouthShrugLower', 'mouthShrugUpper', 'noseSneerLeft', 'noseSneerRight', 'mouthLowerDownLeft', 'mouthLowerDownRight', 'mouthLeft', 'mouthRight', 'eyeLookDownLeft', 'eyeLookDownRight', 'eyeLookUpLeft', 'eyeLookUpRight', 'eyeLookInLeft', 'eyeLookInRight', 'eyeLookOutLeft', 'eyeLookOutRight', 'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight', 'jawOpen', 'mouthClose', 'mouthFunnel', 'mouthDimpleLeft', 'mouthDimpleRight', 'mouthStretchLeft', 'mouthStretchRight', 'mouthRollLower', 'mouthRollUpper', 'mouthPressLeft', 'mouthPressRight', 'mouthUpperUpLeft', 'mouthUpperUpRight', 'mouthSmileLeft', 'mouthSmileRight', 'tongueOut', 'eyeBlinkLeft', 'eyeBlinkRight']

# emage github:https://github.com/PantoMatrix/PantoMatrix/tree/main/scripts/EMAGE_2024
# mediapipe name
media_name = ['Basis', 'browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight', 
              'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight', 'eyeBlinkLeft', 'eyeBlinkRight', 'eyeLookDownLeft', 'eyeLookDownRight', 'eyeLookInLeft', 'eyeLookInRight', 'eyeLookOutLeft', 'eyeLookOutRight', 'eyeLookUpLeft', 'eyeLookUpRight',
              'eyeSquintLeft', 'eyeSquintRight', 'eyeWideLeft', 'eyeWideRight', 'jawForward', 'jawLeft', 
              'jawOpen', 'jawRight', 'mouthClose', 'mouthDimpleLeft', 'mouthDimpleRight', 'mouthFrownLeft', 'mouthFrownRight', 'mouthFunnel', 'mouthLeft', 'mouthLowerDownLeft', 'mouthLowerDownRight', 'mouthPressLeft', 'mouthPressRight', 
              'mouthPucker', 'mouthRight', 'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower', 'mouthShrugUpper', 'mouthSmileLeft', 'mouthSmileRight', 'mouthStretchLeft', 'mouthStretchRight', 'mouthUpperUpLeft', 'mouthUpperUpRight', 'noseSneerLeft', 'noseSneerRight', 
              'tongueOut']

I = matrix_to_rotation_6d(torch.eye(3)[None].cuda())

def to_tensor(array, dtype=torch.float32):
    if 'torch.tensor' not in str(type(array)):
        return torch.tensor(array, dtype=dtype)

def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)

class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

def rot_mat_to_euler(rot_mats):
    # Calculates rotation matrix to euler angles
    # Careful for extreme cases of eular angles like [0.0, pi, 0.0]
    sy = torch.sqrt(rot_mats[:, 0, 0] * rot_mats[:, 0, 0] +
                    rot_mats[:, 1, 0] * rot_mats[:, 1, 0])
    return torch.atan2(-rot_mats[:, 2, 0], sy)

class BlendModel(nn.Module):

    def __init__(self, actor_name='actor', scale = 5):
        super(BlendModel, self).__init__()
        logger.info(f"[Blend] Creating the Blend model")
        
        self.scale = scale

        exp_face = np.zeros((5023, 3, 52))             #
        #canonical = np.zeros((5023, 3))                # 5023, 6045
        
        # from emage
        self.bs = "/opt/nas/n/local/yyj/GAGAvatar/exp/"
        verts, faces, aux = load_obj(self.bs + 'Basis.obj')
        exp_face[:,:,0] = verts.numpy()[:5023,:]
        canonical = exp_face[:,:,0]

        for i, name in enumerate(media_name[1:-1]):
            verts, faces, aux = load_obj( self.bs + name + '.obj')
            exp_face[:5023,:, i+1]= verts.numpy() - canonical[:5023, :]
        
        self.register_parameter('exp_face', nn.Parameter(to_tensor(exp_face), requires_grad=False))
        self.register_parameter('exp_face_displace', nn.Parameter(to_tensor(exp_face), requires_grad=True))

        self.max_offset = 0.1

        # from IPython import embed 
        # embed()
        # print('faces shape.', faces.verts_idx.shape)
        # for render mesh
        self.faces = faces
        self.image_size = 512
        self.device = 'cuda'
        self.renderMesh = RenderMesh(image_size = self.image_size, faces = faces.verts_idx)

        raster_settings = RasterizationSettings(
            image_size=self.image_size,
            faces_per_pixel=1,
            cull_backfaces=True,
            perspective_correct=True
        )

        self.lights = PointLights(
            device=self.device,
            location=((0.0, 0.0, 5.0),),
            ambient_color=((0.5, 0.5, 0.5),),
            diffuse_color=((0.5, 0.5, 0.5),)
        )

        self.mesh_rasterizer = MeshRasterizer(raster_settings=raster_settings)
        self.debug_renderer = MeshRenderer(
            rasterizer=self.mesh_rasterizer,
            shader=SoftPhongShader(device=self.device, lights=self.lights)
        )

    def render(self, vertices, transform, focal_length):
        image, _ = self.renderMesh(vertices, transform_matrix=transform, focal_length = focal_length)
        return image
    
    # input 
    def forward(self, exp_parms):
        batch_size = exp_parms.shape[0]
        exp_face  = self.exp_face.expand(batch_size, -1, -1, -1)
        exp_face_displace = self.exp_face_displace.expand(batch_size, -1, -1, -1)

        exp_face = exp_face + exp_face_displace * self.max_offset          # 

        exp_parms = exp_parms.unsqueeze(dim = 1).unsqueeze(dim = 2)
        out = exp_face[:,:,:,0] + (exp_face * exp_parms).sum(axis = 3)
        
        out = out * self.scale
        return out

    def render_shape(self, vertices, transform_matrix, focal_length):
        B = vertices.shape[0]
        V = vertices.shape[1]
        faces = None
        white = None
        if faces is None:
            faces = self.faces.verts_idx.cuda()[None].repeat(B, 1, 1)
        if not white:
            verts_rgb = torch.from_numpy(np.array([80, 140, 200]) / 255.).cuda().float()[None, None, :].repeat(B, V, 1)
        else:
            verts_rgb = torch.from_numpy(np.array([1.0, 1.0, 1.0])).cuda().float()[None, None, :].repeat(B, V, 1)
        textures = TexturesVertex(verts_features=verts_rgb.cuda())
        meshes_world = Meshes(verts=[vertices[i] for i in range(B)], faces=[faces[i] for i in range(B)], textures=textures)

        blend = BlendParams(background_color=(1.0, 1.0, 1.0))

        cameras =  self._build_cameras(transform_matrix, focal_length)

        fragments = self.mesh_rasterizer(meshes_world, cameras=cameras)
        rendering = self.debug_renderer.shader(fragments, meshes_world, cameras=cameras, blend_params=blend)
        rendering = rendering.permute(0, 3, 1, 2).detach()
        return rendering[:, 0:3, :, :]

    def _build_cameras(self, transform_matrix, focal_length):
        batch_size = transform_matrix.shape[0]
        screen_size = torch.tensor(
            [self.image_size, self.image_size], device=self.device
        ).float()[None].repeat(batch_size, 1)
        cameras_kwargs = {
            'principal_point': torch.zeros(batch_size, 2, device=self.device).float(), 'focal_length': focal_length, 
            'image_size': screen_size, 'device': self.device,
        }
        cameras = PerspectiveCameras(**cameras_kwargs, R=transform_matrix[:, :3, :3], T=transform_matrix[:, :3, 3])
        return cameras