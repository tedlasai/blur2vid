import torch
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from torchvision.transforms.functional import resize
import imageio.v2 as imageio
import numpy as np
import threading
# Load RAFT model with pre-trained weights and set to eval mode
weights = Raft_Large_Weights.DEFAULT
raft_model = raft_large(weights=weights).eval().cuda()


def preprocess_frame(img):
    """
    Preprocess a frame tensor of shape [H, W, 3] to [1, 3, H, W],
    ensuring size is divisible by 8 as required by RAFT.
    """
    if torch.max(img) > 1.0:
        img = img / 255.0  # Ensure it's in [0, 1] range
        
    img = img.permute(2, 0, 1)  # [3, H, W]
    h, w = img.shape[1], img.shape[2]
    h, w = h // 8 * 8, w // 8 * 8
    img = resize(img, [h, w])
    return img.unsqueeze(0) # [1, 3, H, W]


# create a module‐wide lock
_raft_lock = threading.Lock()

def estimate_flow(img1, img2):
    with torch.no_grad():
        img1 = preprocess_frame(img1)
        img2 = preprocess_frame(img2)
        with _raft_lock:
            flow = raft_model(img1.cuda(), img2.cuda())[-1]  # [1, 2, H, W]

        # Convert flow to pixel units (if you need to)
        # _, _, H, W = flow.shape
    return flow


def compute_epe(flow_pred, flow_gt):
    """
    Compute End-Point Error between predicted and ground truth flows.
    """
    return torch.norm(flow_pred - flow_gt, dim=0)

def compute_bidirectional_epe(gen_first, gen_last, gt_first, gt_last, per_pixel_mode=False):
    """
    Compute minimum EPE between forward and backward generated flows
    against GT forward flow.
    """
    # Estimate flows
    flow_gen_fw = estimate_flow(gen_first, gen_last)

    flow_gen_bw = estimate_flow(gen_last, gen_first)
    flow_gt = estimate_flow(gt_first, gt_last)


    # Resize predicted flows to match GT flow resolution
    target_size = flow_gt.shape[-2:]
    flow_gen_fw = F.interpolate(flow_gen_fw, size=target_size, mode='bilinear', align_corners=False)
    flow_gen_bw = F.interpolate(flow_gen_bw, size=target_size, mode='bilinear', align_corners=False)

    # Compute EPE
    epe_fw = compute_epe(flow_gen_fw.squeeze(0), flow_gt.squeeze(0))
    epe_bw = compute_epe(flow_gen_bw.squeeze(0), flow_gt.squeeze(0))
    if per_pixel_mode:
        epe_fw = torch.min(epe_fw, epe_bw)
        return epe_fw.mean()
    else:
        epe_fw = epe_fw.mean()
        epe_bw = epe_bw.mean()
        return min(epe_fw.item(), epe_bw.item())

def video_to_torch(path):
    """
    Load video into a torch.FloatTensor of shape [T, H, W, 3], normalized to [0, 1].
    """
    reader = imageio.get_reader(path)
    frames = [frame for frame in reader]
    reader.close()
    return torch.tensor(np.stack(frames), dtype=torch.float32) / 255.0

# base_dir = "/home/tedlasai/genCamera/GoProResults"
# methods = {
#     "Ours": "Ours",
#     "Favaro": "Favaro",
#     "MotionETR": "MotionETR",
#     "GT": "GT"
# }
# import os
# from glob import glob

# def find_video_files(method_dir):
#     return sorted(glob(os.path.join(base_dir, method_dir, "**/*.mp4"), recursive=True))

# # Assume that GT has the complete list of video paths
# gt_video_paths = find_video_files(methods["GT"])

# for gt_path in gt_video_paths:
#     rel_path = os.path.relpath(gt_path, os.path.join(base_dir, methods["GT"]))
    
#     try:
#         # Construct corresponding paths
#         ours_path = os.path.join(base_dir, methods["Ours"], rel_path)
#         favaro_path = os.path.join(base_dir, methods["Favaro"], rel_path)
#         motionetr_path = os.path.join(base_dir, methods["MotionETR"], rel_path)

#         # Load videos
#         ours = video_to_torch(ours_path)
#         favaro = video_to_torch(favaro_path)
#         motionetr = video_to_torch(motionetr_path)
#         gt = video_to_torch(gt_path)

#         # Compute EPEs
#         epe_ours = compute_bidirectional_epe(ours[0], ours[-1], gt[0], gt[-1], per_pixel_mode=True)
#         epe_favaro = compute_bidirectional_epe(favaro[0], favaro[-1], gt[0], gt[-1], per_pixel_mode=True)
#         epe_motionetr = compute_bidirectional_epe(motionetr[0], motionetr[-1], gt[0], gt[-1], per_pixel_mode=True)

#         print(f"PER PIXEL: {rel_path}: Ours EPE = {epe_ours:.4f}, Favaro EPE = {epe_favaro:.4f}, MotionETR EPE = {epe_motionetr:.4f}")

#         #compute EPE but w/ per-pixel mode off
#         epe_ours = compute_bidirectional_epe(ours[0], ours[-1], gt[0], gt[-1], per_pixel_mode=False)
#         epe_favaro = compute_bidirectional_epe(favaro[0], favaro[-1], gt[0], gt[-1], per_pixel_mode=False)
#         epe_motionetr = compute_bidirectional_epe(motionetr[0], motionetr[-1], gt[0], gt[-1], per_pixel_mode=False)

#         print(f"         : {rel_path}: Ours EPE = {epe_ours:.4f}, Favaro EPE = {epe_favaro:.4f}, MotionETR EPE = {epe_motionetr:.4f}")

#     except Exception as e:
#         print(f"[Error] Skipping {rel_path} due to: {str(e)}")