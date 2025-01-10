
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
import numpy as np
import lightning
import os
import torch
import torchvision

import time
from core.models import build_model
from core.libs.utils import ConfigDict
from core.libs.GAGAvatar_track.engines import CoreEngine as TrackEngine

MP_TASK_FILE='face_landmarker_with_blendshapes.task'
MP_TASK_FILE='face_landmarker.task'

FOCAL_LENGTH = 12.0

def is_image(image_path):
    extension_name = image_path.split('.')[-1].lower()
    return extension_name in ['jpg', 'png', 'jpeg']

def get_tracked_results(image_path, track_engine, force_retrack=False):
    if not is_image(image_path):
        print(f'Please input a image path, got {image_path}.')
        return None
    tracked_pt_path = 'render_results/tracked/tracked.pt'
    if not os.path.exists(tracked_pt_path):
        os.makedirs('render_results/tracked', exist_ok=True)
        torch.save({}, tracked_pt_path)
    tracked_data = torch.load(tracked_pt_path, weights_only=False)
    image_base = os.path.basename(image_path)
    if image_base in tracked_data and not force_retrack:
        print(f'Load tracking result from cache: {tracked_pt_path}.')
    else:
        print(f'Tracking {image_path}...')
        image = torchvision.io.read_image(image_path, mode=torchvision.io.ImageReadMode.RGB).float()
        feature_data = track_engine.track_image([image], [image_path])
        if feature_data is not None:
            feature_data = feature_data[image_path]
            torchvision.utils.save_image(
                torch.tensor(feature_data['vis_image']), 'render_results/tracked/{}.jpg'.format(image_base.split('.')[0])
            )
        else:
            print(f'No face detected in {image_path}.')
            return None
        tracked_data[image_base] = feature_data
        # track all images in this folder
        other_names = [i for i in os.listdir(os.path.dirname(image_path)) if is_image(i)]
        other_paths = [os.path.join(os.path.dirname(image_path), i) for i in other_names]
        if len(other_paths) <= 35:
            print('Track on all images in this folder to save time.')
            other_images = [torchvision.io.read_image(imp, mode=torchvision.io.ImageReadMode.RGB).float() for imp in other_paths]
            try:
                other_feature_data = track_engine.track_image(other_images, other_names)
                for key in other_feature_data:
                    torchvision.utils.save_image(
                        torch.tensor(other_feature_data[key]['vis_image']), 'render_results/tracked/{}.jpg'.format(key.split('.')[0])
                    )
                tracked_data.update(other_feature_data)
            except Exception as e:
                print(f'Error: {e}.')
        # save tracking result
        torch.save(tracked_data, tracked_pt_path)
    feature_data = tracked_data[image_base]
    for key in list(feature_data.keys()):
        if isinstance(feature_data[key], np.ndarray):
            feature_data[key] = torch.tensor(feature_data[key])
    return feature_data

def build_points_planes(plane_size, transforms, focal_length = 12.0):
    x, y = torch.meshgrid(
        torch.linspace(1, -1, plane_size, dtype=torch.float32), 
        torch.linspace(1, -1, plane_size, dtype=torch.float32), 
        indexing="xy",
    )
    R = transforms[:3, :3]; T = transforms[:3, 3:]
    cam_dirs = torch.tensor([[0., 0., 1.]], dtype=torch.float32)
    ray_dirs = torch.nn.functional.pad(
        torch.stack([x/focal_length, y/focal_length], dim=-1), (0, 1), value=1.0
    )
    cam_dirs = torch.matmul(R, cam_dirs.reshape(-1, 3)[:, :, None])[..., 0]
    ray_dirs = torch.matmul(R, ray_dirs.reshape(-1, 3)[:, :, None])[..., 0]
    origins = (-torch.matmul(R, T)[..., 0]).broadcast_to(ray_dirs.shape).squeeze()
    distance = ((origins[0] * cam_dirs[0]).sum()).abs()
    plane_points = origins + distance * ray_dirs
    return {'plane_points': plane_points, 'plane_dirs': cam_dirs[0]}

class FaceMeshDetector:
    def __init__(self, MP_TASK_FILE = "face_landmarker_with_blendshapes.task"):
        with open(MP_TASK_FILE, mode="rb") as f:
            f_buffer = f.read()
        base_options = mp_python.BaseOptions(model_asset_buffer=f_buffer)
        options = mp_python.vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=1)
        self.model = mp_python.vision.FaceLandmarker.create_from_options(
            options)
 
    def getBlend(self, blendshapes):
        res = np.zeros(52)
        for i, bld in enumerate(blendshapes):
            res[i] = bld.score
        return res
    
    def getLmk(self, lmks):
        res = np.zeros((478, 3))
        for i, lmk in enumerate(lmks):
            res[i][0], res[i][1], res[i][2] = lmk.x, lmk.y, lmk.z
        return res
    
    def update(self, frame):
        frame_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        result = self.model.detect(frame_mp)
        blend = self.getBlend(result.face_blendshapes[0])
        lmk = self.getLmk(result.face_landmarks[0])
        matrix = result.facial_transformation_matrixes[0]

        return lmk, blend, matrix

video_path = '/opt/nas/n/local/yyj/GAGAvatar/core/libs/GAGAvatar_track/demos/obama.mp4'
cap = cv2.VideoCapture(video_path)  # 0
face_mesh = FaceMeshDetector(MP_TASK_FILE)

device='cuda'
resume_path='/opt/nas/n/local/yyj/GAGAvatar/outputs/GAGAvatar_spade32_VFHQ/Jan10_yyj/checkpoints/best_12000_0.871.pt'

image_path = '/opt/nas/n/local/yyj/GAGAvatar/demos/examples/yyj512.png'
force_retrack = False

torch.set_float32_matmul_precision('high')

print(f'Loading model...')
lightning_fabric = lightning.Fabric(accelerator=device, strategy='auto', devices=[0],)
lightning_fabric.launch()
full_checkpoint = lightning_fabric.load(resume_path)
meta_cfg = ConfigDict(init_dict=full_checkpoint['meta_cfg'])
model = build_model(model_cfg=meta_cfg.MODEL)
model.load_state_dict(full_checkpoint['model'])
model = lightning_fabric.setup(model)
model.eval()

track_engine = TrackEngine(focal_length=12.0, device=device)
# build input data
feature_name = os.path.basename(image_path).split('.')[0]
feature_data = get_tracked_results(image_path, track_engine, force_retrack=force_retrack)
if feature_data is None:
    print(f'Finish inference, no face in input: {image_path}.')
# from IPython import embed 
# embed()
point_plane_size = 296
batch={}
batch['f_image'] =  torchvision.transforms.functional.resize(feature_data['image'].cpu(), (518, 518), antialias=True).unsqueeze(dim=0).to(device)
batch['f_planes'] =  build_points_planes(point_plane_size, feature_data['transform_matrix'])

batch['f_transform'] = feature_data['transform_matrix'].unsqueeze(dim=0).to(device)

batch['f_planes']['plane_dirs'] = batch['f_planes']['plane_dirs'].unsqueeze(dim=0).to(device)
batch['f_planes']['plane_points'] = batch['f_planes']['plane_points'].unsqueeze(dim=0).to(device)
batch['f_blend'] = feature_data['blend'].unsqueeze(dim=0).to(device)

result = model.inference_init(batch, mesh = True)

if result!= -1:
    t_image = result['sr_gen_image']
    torchvision.utils.save_image(t_image[0],  os.path.join(f'./temp/start_image.png'))
start = time.time()
tot = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the image horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the image from BGR to RGB (MediaPipe uses RGB format)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    lmk, blend, matrix = face_mesh.update(rgb_frame)
    
    batch['t_transform'] = batch['f_transform']
    batch['t_blend'] = torch.from_numpy(blend).to(torch.float32).unsqueeze(dim=0).to(device)

    result = model.inference_blend(batch, mesh = True)
    
    tot += 1
end = time.time()
print((end - start) / tot)