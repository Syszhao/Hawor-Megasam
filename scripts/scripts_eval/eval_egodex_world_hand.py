'''
运行命令
/home/user/miniconda3/envs/hawor/bin/python scripts/scripts_eval/eval_egodex_world_hand.py   --video_path /data/HaWoR/test/color/13.mp4   --generate_world   --force_slam   --force_world --render_visuals
'''


import argparse
import copy
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import h5py
import joblib
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__) + "/../..")

from hawor.utils.process import run_mano, run_mano_left
from hawor.utils.rotation import angle_axis_to_rotation_matrix
from lib.pipeline.slam_paths import get_slam_filename, get_slam_path


HAND_INDEX = {"left": 0, "right": 1}
FINGERTIP_IDXS = np.asarray([4, 8, 12, 16, 20], dtype=np.int64)
SELECTION_METRICS = (
    "median_wrist_offset_wrist_rte_mean_mm",
    "median_wrist_offset_mpjpe_mean_mm",
    "camera_rpe_trans_mean_mm",
    "camera_rpe_rot_mean_deg",
    "pred_jitter_frame_mean_mm",
)
SELECTION_NORM_KEYS = {
    "median_wrist_offset_wrist_rte_mean_mm": "selection_offset_wrist_norm",
    "median_wrist_offset_mpjpe_mean_mm": "selection_offset_mpjpe_norm",
    "camera_rpe_trans_mean_mm": "selection_camera_rpe_norm",
    "camera_rpe_rot_mean_deg": "selection_camera_rot_norm",
    "pred_jitter_frame_mean_mm": "selection_pred_jitter_norm",
}

MANO_TO_EGODEX = [
    "Hand",
    "ThumbKnuckle",
    "ThumbIntermediateBase",
    "ThumbIntermediateTip",
    "ThumbTip",
    "IndexFingerKnuckle",
    "IndexFingerIntermediateBase",
    "IndexFingerIntermediateTip",
    "IndexFingerTip",
    "MiddleFingerKnuckle",
    "MiddleFingerIntermediateBase",
    "MiddleFingerIntermediateTip",
    "MiddleFingerTip",
    "RingFingerKnuckle",
    "RingFingerIntermediateBase",
    "RingFingerIntermediateTip",
    "RingFingerTip",
    "LittleFingerKnuckle",
    "LittleFingerIntermediateBase",
    "LittleFingerIntermediateTip",
    "LittleFingerTip",
]


def hand_dataset_name(hand, joint_name):
    return f"{hand}{joint_name[0].upper()}{joint_name[1:]}"


def infer_paths(video_path, end_idx=None):
    video_path = Path(video_path).resolve()
    video_root = video_path.parent
    video_name = video_path.stem
    seq_folder = video_root / video_name
    hdf5_path = video_root / f"{video_name}.hdf5"

    if not hdf5_path.exists():
        raise FileNotFoundError(f"EgoDex HDF5 not found: {hdf5_path}")
    with h5py.File(hdf5_path, "r") as f:
        frame_count = int(f["transforms/camera"].shape[0])
    if end_idx is None:
        end_idx = frame_count

    return {
        "video_path": video_path,
        "seq_folder": seq_folder,
        "hdf5_path": hdf5_path,
        "frame_count": frame_count,
        "start_idx": 0,
        "end_idx": int(end_idx),
    }


def default_world_path(seq_folder, backend):
    return Path(seq_folder) / f"world_space_res_{backend}.pth"


def default_slam_path(seq_folder, start_idx, end_idx, backend):
    return Path(seq_folder) / "SLAM" / get_slam_filename(start_idx, end_idx, backend)


def quaternion_xyzw_to_matrix(quat):
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    quat = quat / np.clip(norm, 1e-12, None)
    x, y, z, w = [quat[..., i] for i in range(4)]

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    matrix = np.empty(quat.shape[:-1] + (3, 3), dtype=np.float64)
    matrix[..., 0, 0] = 1.0 - 2.0 * (yy + zz)
    matrix[..., 0, 1] = 2.0 * (xy - wz)
    matrix[..., 0, 2] = 2.0 * (xz + wy)
    matrix[..., 1, 0] = 2.0 * (xy + wz)
    matrix[..., 1, 1] = 1.0 - 2.0 * (xx + zz)
    matrix[..., 1, 2] = 2.0 * (yz - wx)
    matrix[..., 2, 0] = 2.0 * (xz - wy)
    matrix[..., 2, 1] = 2.0 * (yz + wx)
    matrix[..., 2, 2] = 1.0 - 2.0 * (xx + yy)
    return matrix


def load_slam_c2w(slam_path):
    slam_path = Path(slam_path)
    if not slam_path.exists():
        raise FileNotFoundError(f"SLAM result not found: {slam_path}")

    data = np.load(slam_path, allow_pickle=True)
    if "cam_c2w" in data.files:
        cam_c2w = np.asarray(data["cam_c2w"], dtype=np.float64)
    else:
        traj = np.asarray(data["traj"], dtype=np.float64)
        scale = float(np.asarray(data["scale"])) if "scale" in data.files else 1.0
        cam_c2w = np.repeat(np.eye(4, dtype=np.float64)[None], len(traj), axis=0)
        cam_c2w[:, :3, 3] = traj[:, :3] * scale
        cam_c2w[:, :3, :3] = quaternion_xyzw_to_matrix(traj[:, 3:7])
    if cam_c2w.ndim != 3 or cam_c2w.shape[1:] != (4, 4):
        raise ValueError(f"Expected Nx4x4 cam_c2w in {slam_path}, got {cam_c2w.shape}")
    return cam_c2w


def load_gt_camera(hdf5_path):
    with h5py.File(hdf5_path, "r") as f:
        return np.asarray(f["transforms/camera"], dtype=np.float64)


def load_gt_hand_points(hdf5_path, conf_thresh):
    with h5py.File(hdf5_path, "r") as f:
        frame_count = int(f["transforms/camera"].shape[0])
        points = np.full((2, frame_count, len(MANO_TO_EGODEX), 3), np.nan, dtype=np.float64)
        rotations = np.full((2, frame_count, 3, 3), np.nan, dtype=np.float64)
        valid = np.zeros((2, frame_count, len(MANO_TO_EGODEX)), dtype=bool)
        root_valid = np.zeros((2, frame_count), dtype=bool)

        for hand, hand_idx in HAND_INDEX.items():
            for joint_idx, joint_name in enumerate(MANO_TO_EGODEX):
                name = hand_dataset_name(hand, joint_name)
                transform_key = f"transforms/{name}"
                if transform_key not in f:
                    continue
                transforms = np.asarray(f[transform_key], dtype=np.float64)
                points[hand_idx, :, joint_idx] = transforms[:, :3, 3]

                conf_key = f"confidences/{name}"
                if conf_key in f:
                    conf = np.asarray(f[conf_key], dtype=np.float64)
                    valid[hand_idx, :, joint_idx] = conf >= conf_thresh
                else:
                    valid[hand_idx, :, joint_idx] = True

            root_name = hand_dataset_name(hand, "Hand")
            rotations[hand_idx] = np.asarray(f[f"transforms/{root_name}"], dtype=np.float64)[:, :3, :3]
            root_valid[hand_idx] = valid[hand_idx, :, 0]

    valid &= np.isfinite(points).all(axis=-1)
    root_valid &= np.isfinite(rotations).all(axis=(-1, -2))
    return points, rotations, valid, root_valid


def load_world_result(world_path):
    world_path = Path(world_path)
    if not world_path.exists():
        raise FileNotFoundError(f"World-space result not found: {world_path}")
    pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = joblib.load(world_path)
    pred_trans = torch.as_tensor(pred_trans).float().cpu()
    pred_rot = torch.as_tensor(pred_rot).float().cpu()
    pred_hand_pose = torch.as_tensor(pred_hand_pose).float().cpu()
    pred_betas = torch.as_tensor(pred_betas).float().cpu()
    pred_valid = np.asarray(pred_valid).astype(bool)
    return pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid


def fit_alignment(source_xyz, target_xyz, with_scale=False):
    source_xyz = np.asarray(source_xyz, dtype=np.float64)
    target_xyz = np.asarray(target_xyz, dtype=np.float64)
    mask = np.isfinite(source_xyz).all(axis=1) & np.isfinite(target_xyz).all(axis=1)
    source_xyz = source_xyz[mask]
    target_xyz = target_xyz[mask]
    if len(source_xyz) < 3:
        raise ValueError("Need at least 3 valid camera poses to fit alignment")

    x = source_xyz.T
    y = target_xyz.T
    n = x.shape[1]
    mu_x = x.mean(axis=1, keepdims=True)
    mu_y = y.mean(axis=1, keepdims=True)
    x_centered = x - mu_x
    y_centered = y - mu_y
    cov = (y_centered @ x_centered.T) / n
    u, d, vt = np.linalg.svd(cov)
    s_mat = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        s_mat[-1, -1] = -1.0
    rot = u @ s_mat @ vt

    if with_scale:
        var_x = np.square(x_centered).sum() / n
        scale = np.trace(np.diag(d) @ s_mat) / max(var_x, 1e-12)
    else:
        scale = 1.0
    trans = (mu_y - scale * rot @ mu_x).reshape(3)
    return float(scale), rot, trans


def transform_points(points, scale, rot, trans):
    return scale * np.einsum("ij,htpj->htpi", rot, points) + trans.reshape(1, 1, 1, 3)


def transform_rotations(rotations, rot):
    return np.einsum("ij,htjk->htik", rot, rotations)


def transform_camera_c2w(camera_c2w, scale, rot, trans):
    camera_c2w = np.asarray(camera_c2w, dtype=np.float64)
    transformed = camera_c2w.copy()
    transformed[:, :3, :3] = np.einsum("ij,tjk->tik", rot, camera_c2w[:, :3, :3])
    transformed[:, :3, 3] = scale * np.einsum("ij,tj->ti", rot, camera_c2w[:, :3, 3]) + trans
    return transformed


def invert_transforms(transforms):
    transforms = np.asarray(transforms, dtype=np.float64)
    inverted = np.repeat(np.eye(4, dtype=np.float64)[None], len(transforms), axis=0)
    rotation = transforms[:, :3, :3]
    translation = transforms[:, :3, 3]
    inverted[:, :3, :3] = np.transpose(rotation, (0, 2, 1))
    inverted[:, :3, 3] = -np.einsum("tij,tj->ti", inverted[:, :3, :3], translation)
    return inverted


def camera_motion_metrics(slam_camera_c2w, gt_camera_c2w):
    camera_error = np.linalg.norm(slam_camera_c2w[:, :3, 3] - gt_camera_c2w[:, :3, 3], axis=-1)

    if len(slam_camera_c2w) < 2 or len(gt_camera_c2w) < 2:
        rpe_translation = np.asarray([], dtype=np.float64)
        rpe_rotation = np.asarray([], dtype=np.float64)
    else:
        slam_rel = np.matmul(invert_transforms(slam_camera_c2w[:-1]), slam_camera_c2w[1:])
        gt_rel = np.matmul(invert_transforms(gt_camera_c2w[:-1]), gt_camera_c2w[1:])
        delta_rel = np.matmul(invert_transforms(gt_rel), slam_rel)
        rpe_translation = np.linalg.norm(delta_rel[:, :3, 3], axis=-1)
        rpe_rotation = rotation_angle_deg(delta_rel[:, :3, :3])

    camera_ate = summarize(camera_error, scale=1000.0)
    camera_rpe_trans = summarize(rpe_translation, scale=1000.0)
    camera_rpe_rot = summarize(rpe_rotation, scale=1.0)
    return {
        "camera_ate_mean_mm": camera_ate["mean"],
        "camera_ate_p90_mm": camera_ate["p90"],
        "camera_rpe_trans_mean_mm": camera_rpe_trans["mean"],
        "camera_rpe_trans_p90_mm": camera_rpe_trans["p90"],
        "camera_rpe_rot_mean_deg": camera_rpe_rot["mean"],
        "camera_rpe_rot_p90_deg": camera_rpe_rot["p90"],
    }


def rotation_angle_deg(delta_rot):
    trace = np.trace(delta_rot, axis1=-2, axis2=-1)
    cos_angle = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def get_video_fps(video_path, default_fps):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 0.0
    cap.release()
    if not np.isfinite(fps) or fps <= 1e-6:
        return float(default_fps)
    return float(fps)


def run_mano_points(pred_trans, pred_rot, pred_hand_pose, pred_betas, use_cuda, chunk_size):
    device = torch.device("cuda") if use_cuda and torch.cuda.is_available() else torch.device("cpu")
    pred_joints = []
    pred_root_rot = []

    with torch.no_grad():
        for hand in ("left", "right"):
            hand_idx = HAND_INDEX[hand]
            hand_joints = []
            total = pred_trans.shape[1]
            for start in range(0, total, chunk_size):
                end = min(start + chunk_size, total)
                trans = pred_trans[hand_idx : hand_idx + 1, start:end].to(device)
                root = pred_rot[hand_idx : hand_idx + 1, start:end].to(device)
                pose = pred_hand_pose[hand_idx : hand_idx + 1, start:end].to(device)
                betas = pred_betas[hand_idx : hand_idx + 1, start:end].to(device)
                if hand == "left":
                    output = run_mano_left(trans, root, pose, betas=betas, use_cuda=device.type == "cuda")
                else:
                    output = run_mano(trans, root, pose, betas=betas, use_cuda=device.type == "cuda")
                hand_joints.append(output["joints"][0, :, : len(MANO_TO_EGODEX)].detach().cpu())
            pred_joints.append(torch.cat(hand_joints, dim=0))

            root_rot = angle_axis_to_rotation_matrix(pred_rot[hand_idx].reshape(-1, 3))
            pred_root_rot.append(root_rot.detach().cpu())

    joints = torch.stack(pred_joints, dim=0).numpy().astype(np.float64)
    root_rotations = torch.stack(pred_root_rot, dim=0).numpy().astype(np.float64)
    return joints, root_rotations


def summarize(values, scale=1.0):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {
            "count": 0,
            "mean": np.nan,
            "median": np.nan,
            "rmse": np.nan,
            "p90": np.nan,
        }
    scaled = values * scale
    return {
        "count": int(len(scaled)),
        "mean": float(np.mean(scaled)),
        "median": float(np.median(scaled)),
        "rmse": float(np.sqrt(np.mean(np.square(scaled)))),
        "p90": float(np.percentile(scaled, 90)),
    }


def acceleration_errors(pred_points, gt_points, mask, fps):
    if pred_points.shape[1] < 3:
        return np.asarray([]), np.asarray([])

    pred_acc = pred_points[:, :-2] - 2.0 * pred_points[:, 1:-1] + pred_points[:, 2:]
    gt_acc = gt_points[:, :-2] - 2.0 * gt_points[:, 1:-1] + gt_points[:, 2:]
    acc_mask = mask[:, :-2] & mask[:, 1:-1] & mask[:, 2:]
    acc_mask &= np.isfinite(pred_acc).all(axis=-1) & np.isfinite(gt_acc).all(axis=-1)

    pred_raw = np.linalg.norm(pred_acc, axis=-1)[acc_mask] * (fps**2)
    error = np.linalg.norm(pred_acc - gt_acc, axis=-1)[acc_mask] * (fps**2)
    return error, pred_raw


def root_relative_errors(pred_points, gt_points, point_mask, root_mask):
    pred_root = pred_points[:, [0], :]
    gt_root = gt_points[:, [0], :]
    pred_rel = pred_points - pred_root
    gt_rel = gt_points - gt_root
    errors = np.linalg.norm(pred_rel - gt_rel, axis=-1)
    rel_mask = point_mask & root_mask[:, None]
    rel_mask &= np.isfinite(pred_rel).all(axis=-1) & np.isfinite(gt_rel).all(axis=-1)
    return np.where(rel_mask, errors, np.nan)


def median_wrist_offset_corrected_errors(pred_points, gt_points, point_mask, root_mask):
    wrist_delta = gt_points[:, 0, :] - pred_points[:, 0, :]
    offset_mask = root_mask & np.isfinite(wrist_delta).all(axis=-1)
    if not np.any(offset_mask):
        return np.full(pred_points.shape[:2], np.nan, dtype=np.float64), np.full(3, np.nan)

    median_offset = np.median(wrist_delta[offset_mask], axis=0)
    corrected = pred_points + median_offset.reshape(1, 1, 3)
    errors = np.linalg.norm(corrected - gt_points, axis=-1)
    corrected_mask = point_mask & np.isfinite(corrected).all(axis=-1) & np.isfinite(gt_points).all(axis=-1)
    return np.where(corrected_mask, errors, np.nan), median_offset


def backend_metrics(
    backend,
    video_info,
    world_path,
    slam_path,
    alignment_mode,
    gt_points,
    gt_rotations,
    gt_valid,
    gt_root_valid,
    fps,
    use_cuda,
    chunk_size,
):
    pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = load_world_result(world_path)
    pred_points, pred_root_rot = run_mano_points(
        pred_trans,
        pred_rot,
        pred_hand_pose,
        pred_betas,
        use_cuda=use_cuda,
        chunk_size=chunk_size,
    )

    gt_camera = load_gt_camera(video_info["hdf5_path"])
    slam_camera = load_slam_c2w(slam_path)
    frame_count = min(
        video_info["frame_count"],
        len(gt_camera),
        len(slam_camera),
        pred_points.shape[1],
        gt_points.shape[1],
        pred_valid.shape[1],
    )

    pred_points = pred_points[:, :frame_count]
    pred_root_rot = pred_root_rot[:, :frame_count]
    pred_valid = pred_valid[:, :frame_count]
    gt_points = gt_points[:, :frame_count]
    gt_rotations = gt_rotations[:, :frame_count]
    gt_valid = gt_valid[:, :frame_count]
    gt_root_valid = gt_root_valid[:, :frame_count]

    with_scale = alignment_mode == "sim3"
    scale, align_rot, align_trans = fit_alignment(
        slam_camera[:frame_count, :3, 3],
        gt_camera[:frame_count, :3, 3],
        with_scale=with_scale,
    )
    slam_camera_aligned = transform_camera_c2w(
        slam_camera[:frame_count],
        scale,
        align_rot,
        align_trans,
    )
    camera_metrics = camera_motion_metrics(slam_camera_aligned, gt_camera[:frame_count])
    pred_points_aligned = transform_points(pred_points, scale, align_rot, align_trans)
    pred_root_rot_aligned = transform_rotations(pred_root_rot, align_rot)

    summaries = []
    per_frame_rows = []
    raw = {
        "wrist": [],
        "fingertip": [],
        "mpjpe": [],
        "root_relative_mpjpe": [],
        "median_wrist_offset_wrist": [],
        "median_wrist_offset_mpjpe": [],
        "median_wrist_offset_norm": [],
        "jitter_error": [],
        "pred_jitter": [],
        "root_rot": [],
    }
    valid_ratios = []
    gt_point_valid_ratios = []

    for hand, hand_idx in HAND_INDEX.items():
        hand_pred_valid = pred_valid[hand_idx].astype(bool)
        point_mask = gt_valid[hand_idx] & hand_pred_valid[:, None]
        root_mask = gt_root_valid[hand_idx] & hand_pred_valid

        point_errors = np.linalg.norm(
            pred_points_aligned[hand_idx] - gt_points[hand_idx],
            axis=-1,
        )
        point_errors = np.where(point_mask, point_errors, np.nan)

        wrist_errors = point_errors[:, 0]
        fingertip_errors = point_errors[:, FINGERTIP_IDXS]
        mpjpe_errors = point_errors
        root_relative_mpjpe_errors = root_relative_errors(
            pred_points_aligned[hand_idx],
            gt_points[hand_idx],
            point_mask,
            root_mask,
        )
        median_wrist_offset_mpjpe_errors, median_wrist_offset = median_wrist_offset_corrected_errors(
            pred_points_aligned[hand_idx],
            gt_points[hand_idx],
            point_mask,
            root_mask,
        )
        median_wrist_offset_wrist_errors = median_wrist_offset_mpjpe_errors[:, 0]
        median_wrist_offset_norm = np.linalg.norm(median_wrist_offset)

        root_delta = np.einsum(
            "tij,tjk->tik",
            np.transpose(gt_rotations[hand_idx], (0, 2, 1)),
            pred_root_rot_aligned[hand_idx],
        )
        root_errors = rotation_angle_deg(root_delta)
        root_errors = np.where(root_mask, root_errors, np.nan)

        jitter_mask = point_mask
        jitter_error, pred_jitter = acceleration_errors(
            pred_points_aligned[[hand_idx]],
            gt_points[[hand_idx]],
            jitter_mask[None],
            fps,
        )

        hand_summary = build_summary_row(
            video_info,
            backend,
            hand,
            alignment_mode,
            scale,
            frame_count,
            fps,
            hand_pred_valid,
            point_mask,
            wrist_errors,
            fingertip_errors,
            mpjpe_errors,
            root_relative_mpjpe_errors,
            median_wrist_offset_mpjpe_errors,
            median_wrist_offset_wrist_errors,
            np.asarray([median_wrist_offset_norm], dtype=np.float64),
            camera_metrics,
            jitter_error,
            pred_jitter,
            root_errors,
        )
        summaries.append(hand_summary)

        raw["wrist"].append(wrist_errors.reshape(-1))
        raw["fingertip"].append(fingertip_errors.reshape(-1))
        raw["mpjpe"].append(mpjpe_errors.reshape(-1))
        raw["root_relative_mpjpe"].append(root_relative_mpjpe_errors.reshape(-1))
        raw["median_wrist_offset_wrist"].append(median_wrist_offset_wrist_errors.reshape(-1))
        raw["median_wrist_offset_mpjpe"].append(median_wrist_offset_mpjpe_errors.reshape(-1))
        raw["median_wrist_offset_norm"].append(np.asarray([median_wrist_offset_norm], dtype=np.float64))
        raw["jitter_error"].append(jitter_error.reshape(-1))
        raw["pred_jitter"].append(pred_jitter.reshape(-1))
        raw["root_rot"].append(root_errors.reshape(-1))
        valid_ratios.append(float(np.mean(hand_pred_valid)))
        gt_point_valid_ratios.append(float(np.mean(point_mask)))

        for frame_idx in range(frame_count):
            per_frame_rows.append(
                {
                    "video": video_info["video_path"].stem,
                    "backend": backend,
                    "hand": hand,
                    "frame": frame_idx,
                    "valid_pred": bool(hand_pred_valid[frame_idx]),
                    "wrist_rte_mm": safe_nanmean([wrist_errors[frame_idx]], scale=1000.0),
                    "fingertip_error_mm": safe_nanmean(fingertip_errors[frame_idx], scale=1000.0),
                    "world_mpjpe_mm": safe_nanmean(mpjpe_errors[frame_idx], scale=1000.0),
                    "root_relative_mpjpe_mm": safe_nanmean(root_relative_mpjpe_errors[frame_idx], scale=1000.0),
                    "median_wrist_offset_wrist_rte_mm": safe_nanmean(
                        [median_wrist_offset_wrist_errors[frame_idx]],
                        scale=1000.0,
                    ),
                    "median_wrist_offset_mpjpe_mm": safe_nanmean(
                        median_wrist_offset_mpjpe_errors[frame_idx],
                        scale=1000.0,
                    ),
                    "root_rot_error_deg": safe_nanmean([root_errors[frame_idx]], scale=1.0),
                }
            )

    all_summary = build_summary_row(
        video_info,
        backend,
        "all",
        alignment_mode,
        scale,
        frame_count,
        fps,
        np.asarray(valid_ratios),
        np.asarray(gt_point_valid_ratios),
        np.concatenate(raw["wrist"]),
        np.concatenate(raw["fingertip"]),
        np.concatenate(raw["mpjpe"]),
        np.concatenate(raw["root_relative_mpjpe"]),
        np.concatenate(raw["median_wrist_offset_mpjpe"]),
        np.concatenate(raw["median_wrist_offset_wrist"]),
        np.concatenate(raw["median_wrist_offset_norm"]),
        camera_metrics,
        np.concatenate(raw["jitter_error"]),
        np.concatenate(raw["pred_jitter"]),
        np.concatenate(raw["root_rot"]),
        values_are_ratios=True,
    )
    summaries.append(all_summary)

    return {
        "summary_rows": summaries,
        "per_frame_rows": per_frame_rows,
        "alignment": {
            "backend": backend,
            "world_path": str(Path(world_path).resolve()),
            "slam_path": str(Path(slam_path).resolve()),
            "scale": scale,
            "rotation": align_rot.tolist(),
            "translation": align_trans.tolist(),
        },
    }


def safe_nanmean(values, scale):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    return float(np.mean(values) * scale)


def build_summary_row(
    video_info,
    backend,
    hand,
    alignment_mode,
    fit_scale,
    frame_count,
    fps,
    pred_valid,
    point_valid,
    wrist_errors,
    fingertip_errors,
    mpjpe_errors,
    root_relative_mpjpe_errors,
    median_wrist_offset_mpjpe_errors,
    median_wrist_offset_wrist_errors,
    median_wrist_offset_norms,
    camera_metrics,
    jitter_error,
    pred_jitter,
    root_errors,
    values_are_ratios=False,
):
    wrist = summarize(wrist_errors, scale=1000.0)
    tips = summarize(fingertip_errors, scale=1000.0)
    mpjpe = summarize(mpjpe_errors, scale=1000.0)
    root_relative_mpjpe = summarize(root_relative_mpjpe_errors, scale=1000.0)
    median_wrist_offset_mpjpe = summarize(median_wrist_offset_mpjpe_errors, scale=1000.0)
    median_wrist_offset_wrist = summarize(median_wrist_offset_wrist_errors, scale=1000.0)
    median_wrist_offset_norm = summarize(median_wrist_offset_norms, scale=1000.0)
    jitter = summarize(jitter_error, scale=1000.0)
    raw_jitter = summarize(pred_jitter, scale=1000.0)
    root = summarize(root_errors, scale=1.0)

    if values_are_ratios:
        pred_valid_ratio = float(np.nanmean(pred_valid))
        gt_point_valid_ratio = float(np.nanmean(point_valid))
    else:
        pred_valid_ratio = float(np.mean(pred_valid))
        gt_point_valid_ratio = float(np.mean(point_valid))

    row = {
        "video": video_info["video_path"].stem,
        "backend": backend,
        "hand": hand,
        "alignment": alignment_mode,
        "fit_scale": fit_scale,
        "frames": int(frame_count),
        "pred_valid_ratio": pred_valid_ratio,
        "gt_point_valid_ratio": gt_point_valid_ratio,
        "wrist_rte_mean_mm": wrist["mean"],
        "wrist_rte_median_mm": wrist["median"],
        "wrist_rte_p90_mm": wrist["p90"],
        "fingertip_error_mean_mm": tips["mean"],
        "fingertip_error_median_mm": tips["median"],
        "fingertip_error_p90_mm": tips["p90"],
        "world_mpjpe_mean_mm": mpjpe["mean"],
        "world_mpjpe_median_mm": mpjpe["median"],
        "world_mpjpe_p90_mm": mpjpe["p90"],
        "root_relative_mpjpe_mean_mm": root_relative_mpjpe["mean"],
        "root_relative_mpjpe_median_mm": root_relative_mpjpe["median"],
        "root_relative_mpjpe_p90_mm": root_relative_mpjpe["p90"],
        "median_wrist_offset_mpjpe_mean_mm": median_wrist_offset_mpjpe["mean"],
        "median_wrist_offset_mpjpe_median_mm": median_wrist_offset_mpjpe["median"],
        "median_wrist_offset_mpjpe_p90_mm": median_wrist_offset_mpjpe["p90"],
        "median_wrist_offset_wrist_rte_mean_mm": median_wrist_offset_wrist["mean"],
        "median_wrist_offset_wrist_rte_median_mm": median_wrist_offset_wrist["median"],
        "median_wrist_offset_wrist_rte_p90_mm": median_wrist_offset_wrist["p90"],
        "median_wrist_offset_norm_mm": median_wrist_offset_norm["mean"],
        "camera_ate_mean_mm": camera_metrics["camera_ate_mean_mm"],
        "camera_ate_p90_mm": camera_metrics["camera_ate_p90_mm"],
        "camera_rpe_trans_mean_mm": camera_metrics["camera_rpe_trans_mean_mm"],
        "camera_rpe_trans_p90_mm": camera_metrics["camera_rpe_trans_p90_mm"],
        "camera_rpe_rot_mean_deg": camera_metrics["camera_rpe_rot_mean_deg"],
        "camera_rpe_rot_p90_deg": camera_metrics["camera_rpe_rot_p90_deg"],
        "jitter_error_mean_mm_s2": jitter["mean"],
        "jitter_error_p90_mm_s2": jitter["p90"],
        "jitter_error_frame_mean_mm": jitter["mean"] / (fps**2),
        "jitter_error_frame_p90_mm": jitter["p90"] / (fps**2),
        "pred_jitter_mean_mm_s2": raw_jitter["mean"],
        "pred_jitter_p90_mm_s2": raw_jitter["p90"],
        "pred_jitter_frame_mean_mm": raw_jitter["mean"] / (fps**2),
        "pred_jitter_frame_p90_mm": raw_jitter["p90"] / (fps**2),
        "root_rot_error_mean_deg": root["mean"],
        "root_rot_error_p90_deg": root["p90"],
        "wrist_count": wrist["count"],
        "fingertip_count": tips["count"],
        "mpjpe_count": mpjpe["count"],
        "root_relative_mpjpe_count": root_relative_mpjpe["count"],
        "median_wrist_offset_wrist_rte_count": median_wrist_offset_wrist["count"],
        "median_wrist_offset_mpjpe_count": median_wrist_offset_mpjpe["count"],
        "jitter_count": jitter["count"],
    }
    row["selection_score"] = np.nan
    for norm_key in SELECTION_NORM_KEYS.values():
        row[norm_key] = np.nan
    return row


def generate_world_results(eval_args, video_info, backends):
    from scripts.scripts_test_video.detect_track_video import detect_track_video
    from scripts.scripts_test_video.hawor_megasam_slam import hawor_megasam_slam
    from scripts.scripts_test_video.hawor_slam import hawor_slam
    from scripts.scripts_test_video.hawor_video import hawor_infiller, hawor_motion_estimation

    pipeline_args = argparse.Namespace(
        video_path=str(video_info["video_path"]),
        input_type=eval_args.input_type,
        checkpoint=eval_args.checkpoint,
        infiller_weight=eval_args.infiller_weight,
        img_focal=eval_args.img_focal,
        slam_backend="droid",
        force_slam=eval_args.force_slam,
        megasam_root=eval_args.megasam_root,
        megasam_python=eval_args.megasam_python,
        megasam_disable_full_ba=eval_args.megasam_disable_full_ba,
    )

    start_idx, end_idx, seq_folder, _ = detect_track_video(pipeline_args)
    frame_chunks_all, img_focal = hawor_motion_estimation(pipeline_args, start_idx, end_idx, seq_folder)
    pipeline_args.img_focal = img_focal

    for backend in backends:
        out_path = default_world_path(seq_folder, backend)
        if out_path.exists() and not eval_args.force_world:
            print(f"skip world result for {backend}: {out_path}")
            continue

        pipeline_args.slam_backend = backend
        slam_path = get_slam_path(seq_folder, start_idx, end_idx, backend)
        if eval_args.force_slam and os.path.exists(slam_path):
            os.remove(slam_path)
        if not os.path.exists(slam_path):
            if backend == "megasam":
                hawor_megasam_slam(pipeline_args, start_idx, end_idx, force=eval_args.force_slam)
            else:
                hawor_slam(pipeline_args, start_idx, end_idx)
        result = hawor_infiller(pipeline_args, start_idx, end_idx, frame_chunks_all)
        joblib.dump(list(result), out_path)
        print(f"generated world result for {backend}: {out_path}")


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def add_selection_scores(summary_rows):
    groups = {}
    for row in summary_rows:
        key = (row["video"], row["hand"], row["alignment"])
        groups.setdefault(key, []).append(row)

    for rows in groups.values():
        for metric_name in SELECTION_METRICS:
            values = np.asarray([float(row[metric_name]) for row in rows], dtype=np.float64)
            finite = np.isfinite(values)
            norm_key = SELECTION_NORM_KEYS[metric_name]
            if not np.any(finite):
                continue

            min_value = float(np.min(values[finite]))
            max_value = float(np.max(values[finite]))
            denom = max_value - min_value
            for row, value in zip(rows, values):
                if not np.isfinite(value):
                    row[norm_key] = np.nan
                elif denom <= 1e-12:
                    row[norm_key] = 0.0
                else:
                    row[norm_key] = float((value - min_value) / denom)

        for row in rows:
            norm_values = [float(row[key]) for key in SELECTION_NORM_KEYS.values()]
            norm_values = [value for value in norm_values if np.isfinite(value)]
            row["selection_score"] = float(np.mean(norm_values)) if norm_values else np.nan
    return summary_rows


def print_table(summary_rows):
    rows = [row for row in summary_rows if row["hand"] == "all"]
    if not rows:
        return
    rows = sorted(
        rows,
        key=lambda row: (
            not np.isfinite(row["selection_score"]),
            row["selection_score"] if np.isfinite(row["selection_score"]) else np.inf,
        ),
    )
    print("\nBias-reduced SLAM selection metrics (lower is better):")
    header = (
        f"{'video':<12} {'backend':<8} {'score':>12} {'OffWrist':>10} "
        f"{'OffMPJPE':>10} {'CamT':>10} {'CamR':>8} {'RootRel':>10} {'PredJitFr':>12}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['video']:<12} {row['backend']:<8} "
            f"{row['selection_score']:>12.2f} "
            f"{row['median_wrist_offset_wrist_rte_mean_mm']:>10.2f} "
            f"{row['median_wrist_offset_mpjpe_mean_mm']:>10.2f} "
            f"{row['camera_rpe_trans_mean_mm']:>10.2f} "
            f"{row['camera_rpe_rot_mean_deg']:>8.2f} "
            f"{row['root_relative_mpjpe_mean_mm']:>10.2f} "
            f"{row['pred_jitter_frame_mean_mm']:>12.2f}"
        )
    print("\nselection_score = mean of per-video normalized OffWrist, OffMPJPE, CamT, CamR, PredJitFr")
    print("PredJitFr = pred_jitter_mean_mm_s2 / fps^2, i.e. frame-level second difference in mm")
    print("OffWrist/OffMPJPE remove one median wrist offset per hand; CamT/CamR are frame-to-frame camera motion errors")
    print("Raw Wrist/MPJPE are still saved in summary.csv, but are no longer the main SLAM selection score")


def render_visual_comparisons(args, video_info):
    script_path = Path(__file__).resolve().parent.parent / "scripts_test_video" / "compare_hawor_megasam_vis.py"
    rows = []

    for mode in args.visual_modes:
        mode_label = "2d_cam" if mode == "cam" else "3d_world"
        suffix = "overlay_gt" if args.visual_compare_layout == "overlay" and mode == "world" else args.visual_compare_layout
        if args.visual_hand_alignment == "median_wrist" and args.visual_compare_layout == "overlay" and mode == "world":
            suffix += "_wrist"
        if suffix == "side_by_side":
            suffix = "side_by_side"
        elif suffix == "overlay":
            suffix = "overlay"
        output_path = video_info["seq_folder"] / "vis_compare" / f"hawor_droid_vs_megasam_{mode}_{suffix}.mp4"
        command = [
            sys.executable,
            str(script_path),
            "--video_path",
            str(video_info["video_path"]),
            "--vis_mode",
            mode,
            "--compare_layout",
            args.visual_compare_layout,
            "--overlay_world_alignment",
            args.visual_world_alignment,
            "--overlay_hand_alignment",
            args.visual_hand_alignment,
            "--reuse_world_results",
            "--output",
            str(output_path),
            "--megasam_root",
            str(args.megasam_root),
            "--megasam_python",
            str(args.megasam_python),
        ]
        if args.img_focal is not None:
            command.extend(["--img_focal", str(args.img_focal)])
        if args.droid_world:
            command.extend(["--droid_world", str(args.droid_world)])
        if args.megasam_world:
            command.extend(["--megasam_world", str(args.megasam_world)])
        if args.force_render_visuals:
            command.append("--force_render")

        print("\nRendering visual comparison:")
        print("$ " + " ".join(command))
        subprocess.run(command, check=True)
        rows.append(
            {
                "video": video_info["video_path"].stem,
                "mode": f"{mode_label}_{suffix}",
                "path": str(output_path),
            }
        )
    return rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate HaWoR world-space hand results against EgoDex HDF5 GT hand transforms."
    )
    parser.add_argument(
        "--video_path",
        nargs="+",
        required=True,
        help="One or more EgoDex mp4 paths, e.g. /home/user/data/HaWoR/test/add_remove_lid/0.mp4",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--alignment", choices=["se3", "sim3"], default="se3")
    parser.add_argument("--gt_conf_thresh", type=float, default=0.5)
    parser.add_argument("--fps", type=float, default=30.0, help="Fallback FPS if the video metadata is unavailable.")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--mano_chunk_size", type=int, default=512)
    parser.add_argument(
        "--render_visuals",
        action="store_true",
        help="After scoring, render 2D camera-view and/or 3D world-view DROID-vs-MegaSAM comparison videos.",
    )
    parser.add_argument("--visual_modes", nargs="+", choices=["cam", "world"], default=["cam", "world"])
    parser.add_argument("--visual_compare_layout", choices=["overlay", "side_by_side"], default="overlay")
    parser.add_argument("--visual_world_alignment", choices=["gt", "droid", "none"], default="gt")
    parser.add_argument("--visual_hand_alignment", choices=["none", "median_wrist"], default="none")
    parser.add_argument("--force_render_visuals", action="store_true")

    parser.add_argument("--droid_world", type=str, default=None, help="Only for single-video evaluation.")
    parser.add_argument("--megasam_world", type=str, default=None, help="Only for single-video evaluation.")
    parser.add_argument("--droid_slam", type=str, default=None, help="Only for single-video evaluation.")
    parser.add_argument("--megasam_slam", type=str, default=None, help="Only for single-video evaluation.")

    parser.add_argument("--generate_world", action="store_true", help="Generate backend-specific world_space_res files before evaluation.")
    parser.add_argument("--force_world", action="store_true")
    parser.add_argument("--force_slam", action="store_true")
    parser.add_argument("--input_type", type=str, default="file")
    parser.add_argument("--checkpoint", type=str, default="./weights/hawor/checkpoints/hawor.ckpt")
    parser.add_argument("--infiller_weight", type=str, default="./weights/hawor/checkpoints/infiller.pt")
    parser.add_argument("--img_focal", type=float)
    parser.add_argument("--megasam_root", type=str, default="/home/user/data/evo/mega-sam")
    parser.add_argument("--megasam_python", type=str, default="/home/user/miniconda3/envs/mega_sam/bin/python")
    parser.add_argument(
        "--megasam_disable_full_ba",
        action="store_true",
        help="Disable MegaSAM full BA for debugging. By default full BA is enabled.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if len(args.video_path) > 1 and any(
        [args.droid_world, args.megasam_world, args.droid_slam, args.megasam_slam]
    ):
        raise ValueError("Custom world/slam paths are only supported for a single --video_path")

    all_summary_rows = []
    all_per_frame_rows = []
    visual_rows = []
    metadata = {"alignment": args.alignment, "videos": []}
    use_cuda = (not args.cpu) and torch.cuda.is_available()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else None

    for video_path in args.video_path:
        video_info = infer_paths(video_path)
        if args.generate_world:
            generate_world_results(args, video_info, ["droid", "megasam"])

        if output_dir is None:
            if len(args.video_path) == 1:
                output_dir = video_info["seq_folder"] / "world_hand_eval"
            else:
                output_dir = Path(args.video_path[0]).resolve().parent / "world_hand_eval"

        fps = get_video_fps(video_info["video_path"], args.fps)

        gt_points, gt_rotations, gt_valid, gt_root_valid = load_gt_hand_points(
            video_info["hdf5_path"],
            conf_thresh=args.gt_conf_thresh,
        )

        backend_inputs = {
            "droid": {
                "world": Path(args.droid_world) if args.droid_world else default_world_path(video_info["seq_folder"], "droid"),
                "slam": Path(args.droid_slam)
                if args.droid_slam
                else default_slam_path(video_info["seq_folder"], video_info["start_idx"], video_info["end_idx"], "droid"),
            },
            "megasam": {
                "world": Path(args.megasam_world)
                if args.megasam_world
                else default_world_path(video_info["seq_folder"], "megasam"),
                "slam": Path(args.megasam_slam)
                if args.megasam_slam
                else default_slam_path(video_info["seq_folder"], video_info["start_idx"], video_info["end_idx"], "megasam"),
            },
        }

        for backend, paths in backend_inputs.items():
            result = backend_metrics(
                backend=backend,
                video_info=video_info,
                world_path=paths["world"],
                slam_path=paths["slam"],
                alignment_mode=args.alignment,
                gt_points=copy.deepcopy(gt_points),
                gt_rotations=copy.deepcopy(gt_rotations),
                gt_valid=copy.deepcopy(gt_valid),
                gt_root_valid=copy.deepcopy(gt_root_valid),
                fps=fps,
                use_cuda=use_cuda,
                chunk_size=args.mano_chunk_size,
            )
            all_summary_rows.extend(result["summary_rows"])
            all_per_frame_rows.extend(result["per_frame_rows"])
            metadata["videos"].append(
                {
                    "video_path": str(video_info["video_path"]),
                    "hdf5_path": str(video_info["hdf5_path"]),
                    "fps": fps,
                    "backend": backend,
                    "alignment": result["alignment"],
                }
            )
        if args.render_visuals:
            visual_rows.extend(render_visual_comparisons(args, video_info))

    add_selection_scores(all_summary_rows)
    write_csv(output_dir / "summary.csv", all_summary_rows)
    write_csv(output_dir / "per_frame.csv", all_per_frame_rows)
    if visual_rows:
        write_csv(output_dir / "visual_compare.csv", visual_rows)
        metadata["visual_compare"] = visual_rows
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print_table(all_summary_rows)
    print("\nSaved:")
    print(f"  {output_dir / 'summary.csv'}")
    print(f"  {output_dir / 'per_frame.csv'}")
    if visual_rows:
        print(f"  {output_dir / 'visual_compare.csv'}")
        for row in visual_rows:
            print(f"  {row['mode']}: {row['path']}")
    print(f"  {output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
