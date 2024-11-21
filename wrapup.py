import yaml
import imageio
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
import torch
from tqdm.auto import tqdm
from os.path import splitext
from tempfile import NamedTemporaryFile
import ffmpeg
from scipy.spatial import ConvexHull
from modules.keypoint_detector import KPDetector
from modules.expression_encoder import ExpressionEncoder
from srt.model import FSRT
from srt.checkpoint import Checkpoint


class FaceReenactment:
    def __init__(self, config_path, checkpoint_path, kp_checkpoint_path, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.kp_checkpoint_path = kp_checkpoint_path

        # Load configuration
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)

        # Initialize models
        self.kp_detector = KPDetector().to(self.device)
        self.expression_encoder = None
        self.model = None

        # Load weights and prepare models
        self._initialize_models()

    def _initialize_models(self):
        self.kp_detector.load_state_dict(torch.load(self.kp_checkpoint_path, map_location=self.device))
        self.expression_encoder = ExpressionEncoder(
            expression_size=self.cfg['model']['expression_size'],
            in_channels=self.kp_detector.predictor.out_filters
        ).to(self.device)
        self.model = FSRT(self.cfg['model'], expression_encoder=self.expression_encoder).to(self.device)

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.encoder.load_state_dict(checkpoint['encoder'])
        self.model.decoder.load_state_dict(checkpoint['decoder'])
        self.expression_encoder.load_state_dict(checkpoint['expression_encoder'])

        self.model.eval()
        self.kp_detector.eval()

    def normalize_kp(self, kp_source, kp_driving, kp_driving_initial, adapt_movement_scale=False, use_relative_movement=False):
        if adapt_movement_scale:
            source_area = ConvexHull(kp_source.data.cpu().numpy()).volume
            driving_area = ConvexHull(kp_driving_initial[0].data.cpu().numpy()).volume
            adapt_movement_scale = np.sqrt(source_area) / np.sqrt(driving_area)
        else:
            adapt_movement_scale = 1

        kp_new = kp_driving
        if use_relative_movement:
            kp_value_diff = (kp_driving - kp_driving_initial)
            kp_value_diff *= adapt_movement_scale
            kp_new = kp_value_diff + kp_source

        return kp_new

    def extract_keypoints_and_expression(self, img, src=False):
        bs, c, h, w = img.shape
        nkp = self.kp_detector.num_kp
        with torch.no_grad():
            kps, latent_dict = self.kp_detector(img)
            heatmaps = latent_dict['heatmap'].view(bs, nkp, latent_dict['heatmap'].shape[-2], latent_dict['heatmap'].shape[-1])
            feature_maps = latent_dict['feature_map'].view(bs, latent_dict['feature_map'].shape[-3], latent_dict['feature_map'].shape[-2], latent_dict['feature_map'].shape[-1])

        if kps.shape[1] == 1:
            kps = kps.squeeze(1)

        expression_vector = self.expression_encoder(feature_maps, heatmaps)
        if src:
            expression_vector = expression_vector[None]
        return kps, expression_vector

    def forward_model(self, expression_vector_src, keypoints_src, expression_vector_driv, keypoints_driv, img_src, idx_grids, max_num_pixels, z=None):
        if len(img_src.shape) < 5:
            img_src = img_src.unsqueeze(1)
        if len(keypoints_src.shape) < 4:
            keypoints_src = keypoints_src.unsqueeze(1)

        if z is None:
            z = self.model.encoder(img_src, keypoints_src, idx_grids[:, :1].repeat(1, img_src.shape[1], 1, 1, 1), expression_vector=expression_vector_src)

        target_pos = idx_grids[:, 1]
        target_kps = keypoints_driv
        _, height, width = target_pos.shape[:3]
        target_pos = target_pos.flatten(1, 2)
        target_kps = target_kps.unsqueeze(1).repeat(1, target_pos.shape[1], 1, 1)
        num_pixels = target_pos.shape[1]

        img = torch.zeros((target_pos.shape[0], num_pixels, 3)).to(self.device)
        for i in range(0, num_pixels, max_num_pixels):
            img[:, i:i + max_num_pixels], _ = self.model.decoder(
                z.clone(), target_pos[:, i:i + max_num_pixels], target_kps[:, i:i + max_num_pixels], expression_vector=expression_vector_driv
            )

        return img.view(img.shape[0], height, width, 3), z

    def make_animation(self, source_image, driving_video, relative=False, adapt_scale=False, max_num_pixels=65536):
        _, y, x = np.meshgrid(np.zeros(2), np.arange(source_image.shape[-3]), np.arange(source_image.shape[-2]), indexing="ij")
        idx_grids = np.stack([x, y], axis=-1).astype(np.float32)
        idx_grids[..., 0] = (idx_grids[..., 0] + 0.5 - (source_image.shape[-3] / 2.0)) / (source_image.shape[-3] / 2.0)
        idx_grids[..., 1] = (idx_grids[..., 1] + 0.5 - (source_image.shape[-2] / 2.0)) / (source_image.shape[-2] / 2.0)
        idx_grids = torch.from_numpy(idx_grids).cuda().unsqueeze(0)
        z = None

        with torch.no_grad():
            predictions = []
            source = torch.tensor(source_image.astype(np.float32)).permute(0, 3, 1, 2).cuda()
            driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3).cuda()
            kp_source, expression_vector_src = self.extract_keypoints_and_expression(source.clone(), src=True)
            kp_driving_initial, _ = self.extract_keypoints_and_expression(driving[:, :, 0].clone())

            for frame_idx in tqdm(range(driving.shape[2])):
                driving_frame = driving[:, :, frame_idx].clone()
                kp_driving, expression_vector_driv = self.extract_keypoints_and_expression(driving_frame)
                kp_norm = self.normalize_kp(kp_source[0], kp_driving, kp_driving_initial, adapt_scale, relative)
                out, z = self.forward_model(expression_vector_src, kp_source, expression_vector_driv, kp_norm, source.unsqueeze(0), idx_grids, max_num_pixels, z=z)
                predictions.append(torch.clamp(out[0], 0., 1.).cpu().numpy())
        return predictions

    def run(self, source_image_path, driving_video_path, output_path, relative=False, adapt_scale=False):
        source_image = imageio.imread(source_image_path)
        source_image = [resize(source_image, (256, 256))[..., :3]]

        reader = imageio.get_reader(driving_video_path)
        driving_video = [resize(frame, (256, 256))[..., :3] for frame in reader]
        reader.close()

        predictions = self.make_animation(np.array(source_image), driving_video, relative, adapt_scale)
        imageio.mimsave(output_path, [img_as_ubyte(frame) for frame in predictions], fps=20)


if __name__ == "__main__":
    reenactor = FaceReenactment(
        config_path="runs/vox256/vox256.yaml",
        checkpoint_path="vox256.pt",
        kp_checkpoint_path="fsrt_checkpoints/kp_detector.pt"
    )
    reenactor.run(
        source_image_path="moon.jpg",
        driving_video_path="video5.mp4",
        output_path="result.mp4",
        relative=True,
        adapt_scale=True
    )
