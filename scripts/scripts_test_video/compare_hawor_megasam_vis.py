import argparse
import joblib
import os
import subprocess
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch

import sys
sys.path.insert(0, os.path.dirname(__file__) + "/../..")

from scripts.scripts_test_video.detect_track_video import detect_track_video
from scripts.scripts_test_video.hawor_megasam_slam import hawor_megasam_slam
from scripts.scripts_test_video.hawor_slam import hawor_slam
from scripts.scripts_test_video.hawor_video import hawor_infiller, hawor_motion_estimation
from hawor.utils.process import get_mano_faces, run_mano, run_mano_left
from lib.eval_utils.custom_utils import load_slam_cam
from lib.pipeline.slam_paths import get_slam_path
from lib.vis.run_vis2 import (
    run_vis2_on_video,
    run_vis2_on_video_cam,
    run_vis2_overlay_on_video,
    run_vis2_overlay_on_video_cam,
)


HAND_INDEX = {"left": 0, "right": 1}
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
GT_BONE_PAIRS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


def hand_dataset_name(hand, joint_name):
    return f"{hand}{joint_name[0].upper()}{joint_name[1:]}"


def build_faces():
    faces = get_mano_faces()
    faces_new = np.array(
        [
            [92, 38, 234],
            [234, 38, 239],
            [38, 122, 239],
            [239, 122, 279],
            [122, 118, 279],
            [279, 118, 215],
            [118, 117, 215],
            [215, 117, 214],
            [117, 119, 214],
            [214, 119, 121],
            [119, 120, 121],
            [121, 120, 78],
            [120, 108, 78],
            [78, 108, 79],
        ]
    )
    faces_right = np.concatenate([faces, faces_new], axis=0)
    faces_left = faces_right[:, [0, 2, 1]]
    return faces_left, faces_right


def ensure_slam(args, backend, start_idx, end_idx, seq_folder, force=False):
    slam_path = Path(get_slam_path(seq_folder, start_idx, end_idx, backend))
    if force and slam_path.exists():
        slam_path.unlink()
    if slam_path.exists():
        return slam_path

    args.slam_backend = backend
    if backend == "megasam":
        hawor_megasam_slam(args, start_idx, end_idx, force=force)
    else:
        hawor_slam(args, start_idx, end_idx)
    if not slam_path.exists():
        raise FileNotFoundError(f"SLAM output was not created: {slam_path}")
    return slam_path


def assert_megasam_output(megasam_path):
    with np.load(megasam_path, allow_pickle=True) as data:
        if "cam_c2w" not in data:
            raise KeyError(f"MegaSAM SLAM output must contain cam_c2w: {megasam_path}")
        if "source_backend" in data and str(data["source_backend"]) != "megasam":
            raise ValueError(f"Unexpected source_backend in {megasam_path}: {data['source_backend']}")


def default_world_path(seq_folder, backend):
    return Path(seq_folder) / f"world_space_res_{backend}.pth"


def read_render_focal(args, seq_folder):
    if args.img_focal is not None:
        return float(args.img_focal)

    focal_path = Path(seq_folder) / "est_focal.txt"
    try:
        return float(focal_path.read_text(encoding="utf-8").strip())
    except Exception:
        pass

    hdf5_path = Path(args.video_path).with_suffix(".hdf5")
    if hdf5_path.exists():
        with h5py.File(hdf5_path, "r") as f:
            if "camera/intrinsic" in f:
                intrinsic = np.asarray(f["camera/intrinsic"], dtype=np.float64).reshape(3, 3)
                return float((intrinsic[0, 0] + intrinsic[1, 1]) * 0.5)

    return 600.0


def load_world_result(world_path):
    world_path = Path(world_path)
    if not world_path.exists():
        raise FileNotFoundError(f"World-space result not found: {world_path}")

    pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = joblib.load(world_path)
    return (
        torch.as_tensor(pred_trans).float().cpu(),
        torch.as_tensor(pred_rot).float().cpu(),
        torch.as_tensor(pred_hand_pose).float().cpu(),
        torch.as_tensor(pred_betas).float().cpu(),
        pred_valid,
    )


def build_backend_scene_from_world_result(args, backend, start_idx, end_idx, seq_folder, faces_left, faces_right):
    args.slam_backend = backend
    slam_path = get_slam_path(seq_folder, start_idx, end_idx, backend)
    R_w2c_sla_all, t_w2c_sla_all, R_c2w_sla_all, t_c2w_sla_all = load_slam_cam(slam_path)

    if backend == "droid" and args.droid_world:
        world_path = Path(args.droid_world)
    elif backend == "megasam" and args.megasam_world:
        world_path = Path(args.megasam_world)
    else:
        world_path = default_world_path(seq_folder, backend)
    pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = load_world_result(world_path)
    del pred_valid

    frame_count = min(pred_trans.shape[1], len(R_c2w_sla_all), len(R_w2c_sla_all))
    if frame_count < 2:
        raise ValueError(f"Need at least 2 frames to render {backend}, got {frame_count}")
    vis_start = 0
    vis_end = frame_count - 1
    use_cuda = torch.cuda.is_available()

    right_idx = 1
    pred_glob_r = run_mano(
        pred_trans[right_idx:right_idx + 1, vis_start:vis_end],
        pred_rot[right_idx:right_idx + 1, vis_start:vis_end],
        pred_hand_pose[right_idx:right_idx + 1, vis_start:vis_end],
        betas=pred_betas[right_idx:right_idx + 1, vis_start:vis_end],
        use_cuda=use_cuda,
    )
    right_dict = {
        "vertices": pred_glob_r["vertices"][[0]].cpu(),
        "joints": pred_glob_r["joints"][[0]].cpu(),
        "faces": faces_right,
    }

    left_idx = 0
    pred_glob_l = run_mano_left(
        pred_trans[left_idx:left_idx + 1, vis_start:vis_end],
        pred_rot[left_idx:left_idx + 1, vis_start:vis_end],
        pred_hand_pose[left_idx:left_idx + 1, vis_start:vis_end],
        betas=pred_betas[left_idx:left_idx + 1, vis_start:vis_end],
        use_cuda=use_cuda,
    )
    left_dict = {
        "vertices": pred_glob_l["vertices"][[0]].cpu(),
        "joints": pred_glob_l["joints"][[0]].cpu(),
        "faces": faces_left,
    }

    R_x = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).float()
    R_c2w_sla_all = torch.einsum("ij,njk->nik", R_x, R_c2w_sla_all)
    t_c2w_sla_all = torch.einsum("ij,nj->ni", R_x, t_c2w_sla_all)
    R_w2c_sla_all = R_c2w_sla_all.transpose(-1, -2)
    t_w2c_sla_all = -torch.einsum("bij,bj->bi", R_w2c_sla_all, t_c2w_sla_all)
    left_dict["vertices"] = torch.einsum("ij,btnj->btni", R_x, left_dict["vertices"])
    right_dict["vertices"] = torch.einsum("ij,btnj->btni", R_x, right_dict["vertices"])
    left_dict["joints"] = torch.einsum("ij,btnj->btni", R_x, left_dict["joints"])
    right_dict["joints"] = torch.einsum("ij,btnj->btni", R_x, right_dict["joints"])

    return {
        "left": left_dict,
        "right": right_dict,
        "R_c2w": R_c2w_sla_all[vis_start:vis_end],
        "t_c2w": t_c2w_sla_all[vis_start:vis_end],
        "R_w2c": R_w2c_sla_all[vis_start:vis_end],
        "t_w2c": t_w2c_sla_all[vis_start:vis_end],
        "vis_start": vis_start,
        "vis_end": vis_end,
        "world_path": str(world_path),
    }


def build_backend_scene(args, backend, start_idx, end_idx, seq_folder, frame_chunks_all, faces_left, faces_right):
    args.slam_backend = backend
    slam_path = get_slam_path(seq_folder, start_idx, end_idx, backend)
    R_w2c_sla_all, t_w2c_sla_all, R_c2w_sla_all, t_c2w_sla_all = load_slam_cam(slam_path)

    pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = hawor_infiller(
        args,
        start_idx,
        end_idx,
        frame_chunks_all,
    )
    del pred_valid

    vis_start = 0
    vis_end = pred_trans.shape[1] - 1
    hand2idx = {"left": 0, "right": 1}

    right_idx = hand2idx["right"]
    pred_glob_r = run_mano(
        pred_trans[right_idx:right_idx + 1, vis_start:vis_end],
        pred_rot[right_idx:right_idx + 1, vis_start:vis_end],
        pred_hand_pose[right_idx:right_idx + 1, vis_start:vis_end],
        betas=pred_betas[right_idx:right_idx + 1, vis_start:vis_end],
    )
    right_dict = {
        "vertices": pred_glob_r["vertices"][[0]],
        "joints": pred_glob_r["joints"][[0]],
        "faces": faces_right,
    }

    left_idx = hand2idx["left"]
    pred_glob_l = run_mano_left(
        pred_trans[left_idx:left_idx + 1, vis_start:vis_end],
        pred_rot[left_idx:left_idx + 1, vis_start:vis_end],
        pred_hand_pose[left_idx:left_idx + 1, vis_start:vis_end],
        betas=pred_betas[left_idx:left_idx + 1, vis_start:vis_end],
    )
    left_dict = {
        "vertices": pred_glob_l["vertices"][[0]],
        "joints": pred_glob_l["joints"][[0]],
        "faces": faces_left,
    }

    R_x = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).float()
    R_c2w_sla_all = torch.einsum("ij,njk->nik", R_x, R_c2w_sla_all)
    t_c2w_sla_all = torch.einsum("ij,nj->ni", R_x, t_c2w_sla_all)
    R_w2c_sla_all = R_c2w_sla_all.transpose(-1, -2)
    t_w2c_sla_all = -torch.einsum("bij,bj->bi", R_w2c_sla_all, t_c2w_sla_all)
    left_dict["vertices"] = torch.einsum("ij,btnj->btni", R_x, left_dict["vertices"].cpu())
    right_dict["vertices"] = torch.einsum("ij,btnj->btni", R_x, right_dict["vertices"].cpu())
    left_dict["joints"] = torch.einsum("ij,btnj->btni", R_x, left_dict["joints"].cpu())
    right_dict["joints"] = torch.einsum("ij,btnj->btni", R_x, right_dict["joints"].cpu())

    return {
        "left": left_dict,
        "right": right_dict,
        "R_c2w": R_c2w_sla_all[vis_start:vis_end],
        "t_c2w": t_c2w_sla_all[vis_start:vis_end],
        "R_w2c": R_w2c_sla_all[vis_start:vis_end],
        "t_w2c": t_w2c_sla_all[vis_start:vis_end],
        "vis_start": vis_start,
        "vis_end": vis_end,
    }

def save_render_payload(payload_path, scene, img_focal, image_names):
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        payload_path,
        img_focal=np.float32(img_focal),
        image_names=np.asarray([str(Path(path).resolve()) for path in image_names]),
        left_vertices=scene["left"]["vertices"].cpu().numpy().astype(np.float32),
        right_vertices=scene["right"]["vertices"].cpu().numpy().astype(np.float32),
        faces_left=np.asarray(scene["left"]["faces"], dtype=np.int32),
        faces_right=np.asarray(scene["right"]["faces"], dtype=np.int32),
        R_c2w=scene["R_c2w"].cpu().numpy().astype(np.float32),
        t_c2w=scene["t_c2w"].cpu().numpy().astype(np.float32),
        R_w2c=scene["R_w2c"].cpu().numpy().astype(np.float32),
        t_w2c=scene["t_w2c"].cpu().numpy().astype(np.float32),
    )
    return payload_path


def render_payload(payload_path, output_dir, vis_mode):
    data = np.load(payload_path, allow_pickle=True)
    left = {
        "vertices": torch.from_numpy(data["left_vertices"]),
        "faces": data["faces_left"],
    }
    right = {
        "vertices": torch.from_numpy(data["right_vertices"]),
        "faces": data["faces_right"],
    }
    img_focal = float(data["img_focal"])
    image_names = [str(path) for path in data["image_names"].tolist()]
    if vis_mode == "world":
        video_path = run_vis2_on_video(
            left,
            right,
            str(output_dir),
            img_focal,
            image_names,
            R_c2w=torch.from_numpy(data["R_c2w"]),
            t_c2w=torch.from_numpy(data["t_c2w"]),
            interactive=False,
        )
    else:
        video_path = run_vis2_on_video_cam(
            left,
            right,
            str(output_dir),
            img_focal,
            image_names,
            R_w2c=torch.from_numpy(data["R_w2c"]),
            t_w2c=torch.from_numpy(data["t_w2c"]),
            interactive=False,
        )
    print(f"Rendered backend video: {video_path}")
    return Path(video_path)


def rendered_video_path(output_dir):
    for name in ("video.mp4", "video_0.mp4"):
        video_path = output_dir / "aitviewer" / name
        if video_path.exists():
            return video_path
    return output_dir / "aitviewer" / "video.mp4"


def render_backend_subprocess(payload_path, output_dir, vis_mode):
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--render_payload",
        str(payload_path),
        "--render_output_dir",
        str(output_dir),
        "--render_vis_mode",
        vis_mode,
    ]
    print("$ " + " ".join(command))
    subprocess.run(command, check=True)
    return rendered_video_path(output_dir)


def add_label(frame, text):
    bar_h = 48
    out = cv2.copyMakeBorder(frame, bar_h, 0, 0, 0, cv2.BORDER_CONSTANT, value=(18, 18, 18))
    cv2.putText(out, text, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (245, 245, 245), 2, cv2.LINE_AA)
    return out


def resize_to_height(frame, height):
    if frame.shape[0] == height:
        return frame
    width = int(round(frame.shape[1] * height / frame.shape[0]))
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def stack_videos(left_video, right_video, output_video, left_label, right_label):
    cap_l = cv2.VideoCapture(str(left_video))
    cap_r = cv2.VideoCapture(str(right_video))
    if not cap_l.isOpened():
        raise FileNotFoundError(f"Cannot open rendered video: {left_video}")
    if not cap_r.isOpened():
        raise FileNotFoundError(f"Cannot open rendered video: {right_video}")

    fps = cap_l.get(cv2.CAP_PROP_FPS) or 30.0
    writer = None
    written = 0
    output_video.parent.mkdir(parents=True, exist_ok=True)

    while True:
        ok_l, frame_l = cap_l.read()
        ok_r, frame_r = cap_r.read()
        if not ok_l or not ok_r:
            break

        height = min(frame_l.shape[0], frame_r.shape[0])
        frame_l = add_label(resize_to_height(frame_l, height), left_label)
        frame_r = add_label(resize_to_height(frame_r, height), right_label)
        if frame_l.shape[0] != frame_r.shape[0]:
            height = min(frame_l.shape[0], frame_r.shape[0])
            frame_l = resize_to_height(frame_l, height)
            frame_r = resize_to_height(frame_r, height)
        stacked = np.concatenate([frame_l, frame_r], axis=1)

        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_video), fourcc, fps, (stacked.shape[1], stacked.shape[0]))
            if not writer.isOpened():
                raise IOError(f"Cannot create comparison video: {output_video}")
        writer.write(stacked)
        written += 1

    cap_l.release()
    cap_r.release()
    if writer is not None:
        writer.release()
    if written == 0:
        raise RuntimeError("No frames were written to the comparison video")
    return written


def label_video(input_video, output_video, label):
    input_video = Path(input_video)
    output_video = Path(output_video)
    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open rendered video: {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer = None
    written = 0
    output_video.parent.mkdir(parents=True, exist_ok=True)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = add_label(frame, label)
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_video), fourcc, fps, (frame.shape[1], frame.shape[0]))
            if not writer.isOpened():
                raise IOError(f"Cannot create labeled video: {output_video}")
        writer.write(frame)
        written += 1

    cap.release()
    if writer is not None:
        writer.release()
    if written == 0:
        raise RuntimeError("No frames were written to the labeled video")
    return written


def fit_alignment(source_xyz, target_xyz, with_scale=False):
    source_xyz = np.asarray(source_xyz, dtype=np.float64)
    target_xyz = np.asarray(target_xyz, dtype=np.float64)
    mask = np.isfinite(source_xyz).all(axis=1) & np.isfinite(target_xyz).all(axis=1)
    source_xyz = source_xyz[mask]
    target_xyz = target_xyz[mask]
    if len(source_xyz) < 3:
        raise ValueError("Need at least 3 valid camera poses to fit visual alignment")

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
    return float(scale), rot.astype(np.float32), trans.astype(np.float32)


def clone_scene(scene):
    cloned = {
        "left": {
            "vertices": scene["left"]["vertices"].clone(),
            "faces": scene["left"]["faces"],
        },
        "right": {
            "vertices": scene["right"]["vertices"].clone(),
            "faces": scene["right"]["faces"],
        },
        "R_c2w": scene["R_c2w"].clone(),
        "t_c2w": scene["t_c2w"].clone(),
        "R_w2c": scene["R_w2c"].clone(),
        "t_w2c": scene["t_w2c"].clone(),
        "vis_start": scene["vis_start"],
        "vis_end": scene["vis_end"],
    }
    for hand in ("left", "right"):
        if "joints" in scene[hand]:
            cloned[hand]["joints"] = scene[hand]["joints"].clone()
    if "world_path" in scene:
        cloned["world_path"] = scene["world_path"]
    return cloned


def transform_scene(scene, scale, rot, trans):
    transformed = clone_scene(scene)
    rot = torch.from_numpy(np.asarray(rot, dtype=np.float32))
    trans = torch.from_numpy(np.asarray(trans, dtype=np.float32))
    for hand in ("left", "right"):
        vertices = transformed[hand]["vertices"]
        transformed[hand]["vertices"] = scale * torch.einsum("ij,btvj->btvi", rot, vertices) + trans
        if "joints" in transformed[hand]:
            joints = transformed[hand]["joints"]
            transformed[hand]["joints"] = scale * torch.einsum("ij,btqj->btqi", rot, joints) + trans

    transformed["R_c2w"] = torch.einsum("ij,tjk->tik", rot, transformed["R_c2w"])
    transformed["t_c2w"] = scale * torch.einsum("ij,tj->ti", rot, transformed["t_c2w"]) + trans
    transformed["R_w2c"] = transformed["R_c2w"].transpose(-1, -2)
    transformed["t_w2c"] = -torch.einsum("tij,tj->ti", transformed["R_w2c"], transformed["t_c2w"])
    return transformed


def load_gt_camera_for_overlay(video_path):
    hdf5_path = Path(video_path).with_suffix(".hdf5")
    if not hdf5_path.exists():
        return None
    with h5py.File(hdf5_path, "r") as f:
        if "transforms/camera" not in f:
            return None
        cam_c2w = torch.from_numpy(np.asarray(f["transforms/camera"], dtype=np.float32))

    R_x = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).float()
    R_c2w = torch.einsum("ij,tjk->tik", R_x, cam_c2w[:, :3, :3])
    t_c2w = torch.einsum("ij,tj->ti", R_x, cam_c2w[:, :3, 3])
    return R_c2w, t_c2w


def load_gt_hand_points_for_overlay(video_path, frame_count):
    hdf5_path = Path(video_path).with_suffix(".hdf5")
    if not hdf5_path.exists():
        return None

    points = {
        "left": np.full((frame_count, len(MANO_TO_EGODEX), 3), np.nan, dtype=np.float32),
        "right": np.full((frame_count, len(MANO_TO_EGODEX), 3), np.nan, dtype=np.float32),
    }
    with h5py.File(hdf5_path, "r") as f:
        if "transforms/camera" not in f:
            return None
        available_frames = int(f["transforms/camera"].shape[0])
        frame_count = min(frame_count, available_frames)
        for hand in ("left", "right"):
            for joint_idx, joint_name in enumerate(MANO_TO_EGODEX):
                key = f"transforms/{hand_dataset_name(hand, joint_name)}"
                if key not in f:
                    continue
                transforms = np.asarray(f[key][:frame_count], dtype=np.float32)
                points[hand][:frame_count, joint_idx] = transforms[:, :3, 3]

    R_x = np.asarray([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
    for hand in ("left", "right"):
        points[hand] = np.einsum("ij,tqj->tqi", R_x, points[hand])
    return points


def transform_gt_points(gt_points, scale, rot, trans):
    if gt_points is None:
        return None
    rot = np.asarray(rot, dtype=np.float32)
    trans = np.asarray(trans, dtype=np.float32)
    return {
        hand: scale * np.einsum("ij,tqj->tqi", rot, points) + trans
        for hand, points in gt_points.items()
    }


def octahedron_offsets(radius):
    return np.asarray(
        [
            [radius, 0.0, 0.0],
            [-radius, 0.0, 0.0],
            [0.0, radius, 0.0],
            [0.0, -radius, 0.0],
            [0.0, 0.0, radius],
            [0.0, 0.0, -radius],
        ],
        dtype=np.float32,
    )


def octahedron_faces():
    return np.asarray(
        [
            [0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
            [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5],
        ],
        dtype=np.int32,
    )


def make_joint_sphere_mesh(points, radius=0.010):
    points = np.asarray(points, dtype=np.float32)
    offsets = octahedron_offsets(radius)
    base_faces = octahedron_faces()
    frame_count, joint_count, _ = points.shape

    centers = np.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)
    vertices = centers[:, :, None, :] + offsets[None, None, :, :]
    vertices = vertices.reshape(frame_count, joint_count * len(offsets), 3)

    faces = []
    for joint_idx in range(joint_count):
        faces.append(base_faces + joint_idx * len(offsets))
    return vertices.astype(np.float32), np.concatenate(faces, axis=0).astype(np.int32)


def make_bone_cylinder_mesh(points, pairs=GT_BONE_PAIRS, radius=0.005, segments=8):
    points = np.asarray(points, dtype=np.float32)
    frame_count = points.shape[0]
    vertices = np.zeros((frame_count, len(pairs) * segments * 2, 3), dtype=np.float32)
    faces = []
    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False).astype(np.float32)

    for bone_idx, (start_idx, end_idx) in enumerate(pairs):
        base = bone_idx * segments * 2
        start = np.nan_to_num(points[:, start_idx], nan=0.0, posinf=0.0, neginf=0.0)
        end = np.nan_to_num(points[:, end_idx], nan=0.0, posinf=0.0, neginf=0.0)
        direction = end - start
        lengths = np.linalg.norm(direction, axis=1, keepdims=True)
        safe_direction = direction / np.clip(lengths, 1e-8, None)

        ref = np.tile(np.asarray([0.0, 0.0, 1.0], dtype=np.float32), (frame_count, 1))
        nearly_parallel = np.abs(np.sum(safe_direction * ref, axis=1)) > 0.9
        ref[nearly_parallel] = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        u = np.cross(safe_direction, ref)
        u = u / np.clip(np.linalg.norm(u, axis=1, keepdims=True), 1e-8, None)
        v = np.cross(safe_direction, u)

        for segment_idx, angle in enumerate(angles):
            offset = radius * (np.cos(angle) * u + np.sin(angle) * v)
            vertices[:, base + segment_idx] = start + offset
            vertices[:, base + segments + segment_idx] = end + offset

        for segment_idx in range(segments):
            next_idx = (segment_idx + 1) % segments
            faces.append([base + segment_idx, base + next_idx, base + segments + segment_idx])
            faces.append([base + next_idx, base + segments + next_idx, base + segments + segment_idx])

    return vertices, np.asarray(faces, dtype=np.int32)


def combine_meshes(meshes):
    vertices = []
    faces = []
    offset = 0
    for mesh_vertices, mesh_faces in meshes:
        vertices.append(mesh_vertices)
        faces.append(mesh_faces + offset)
        offset += mesh_vertices.shape[1]
    return np.concatenate(vertices, axis=1), np.concatenate(faces, axis=0)


def make_gt_skeleton_scene(gt_points):
    if gt_points is None:
        return None
    gt_scene = {}
    for hand in ("left", "right"):
        joints = make_joint_sphere_mesh(gt_points[hand])
        bones = make_bone_cylinder_mesh(gt_points[hand])
        vertices, faces = combine_meshes([joints, bones])
        gt_scene[hand] = {
            "vertices": torch.from_numpy(vertices).unsqueeze(0),
            "faces": faces,
        }
    return gt_scene


def apply_median_wrist_offset(scene, gt_points):
    if gt_points is None:
        return scene
    scene = clone_scene(scene)
    for hand in ("left", "right"):
        if "joints" not in scene[hand]:
            continue
        pred_wrist = scene[hand]["joints"][0, :, 0].cpu().numpy()
        gt_wrist = gt_points[hand][: len(pred_wrist), 0]
        valid = np.isfinite(pred_wrist).all(axis=1) & np.isfinite(gt_wrist).all(axis=1)
        if not np.any(valid):
            continue
        offset = torch.from_numpy(np.median(gt_wrist[valid] - pred_wrist[valid], axis=0).astype(np.float32))
        scene[hand]["vertices"] = scene[hand]["vertices"] + offset
        scene[hand]["joints"] = scene[hand]["joints"] + offset
    return scene


def align_world_overlay_scenes(args, droid_scene, megasam_scene):
    droid_scene = clone_scene(droid_scene)
    megasam_scene = clone_scene(megasam_scene)
    frame_count = min(len(droid_scene["t_c2w"]), len(megasam_scene["t_c2w"]))
    droid_scene = slice_scene(droid_scene, frame_count)
    megasam_scene = slice_scene(megasam_scene, frame_count)

    if args.overlay_world_alignment == "none":
        return droid_scene, megasam_scene, droid_scene["R_c2w"], droid_scene["t_c2w"]

    if args.overlay_world_alignment == "gt":
        gt_camera = load_gt_camera_for_overlay(args.video_path)
        if gt_camera is not None:
            gt_R_c2w, gt_t_c2w = gt_camera
            frame_count = min(frame_count, len(gt_t_c2w))
            droid_scene = slice_scene(droid_scene, frame_count)
            megasam_scene = slice_scene(megasam_scene, frame_count)
            target_t = gt_t_c2w[:frame_count].cpu().numpy()
            d_scale, d_rot, d_trans = fit_alignment(droid_scene["t_c2w"].cpu().numpy(), target_t)
            m_scale, m_rot, m_trans = fit_alignment(megasam_scene["t_c2w"].cpu().numpy(), target_t)
            return (
                transform_scene(droid_scene, d_scale, d_rot, d_trans),
                transform_scene(megasam_scene, m_scale, m_rot, m_trans),
                gt_R_c2w[:frame_count],
                gt_t_c2w[:frame_count],
            )
        print("GT camera not found; fall back to DROID world alignment for overlay.")

    m_scale, m_rot, m_trans = fit_alignment(
        megasam_scene["t_c2w"].cpu().numpy(),
        droid_scene["t_c2w"].cpu().numpy(),
    )
    return droid_scene, transform_scene(megasam_scene, m_scale, m_rot, m_trans), droid_scene["R_c2w"], droid_scene["t_c2w"]


def slice_scene(scene, frame_count):
    scene = clone_scene(scene)
    for hand in ("left", "right"):
        scene[hand]["vertices"] = scene[hand]["vertices"][:, :frame_count]
        if "joints" in scene[hand]:
            scene[hand]["joints"] = scene[hand]["joints"][:, :frame_count]
    for key in ("R_c2w", "t_c2w", "R_w2c", "t_w2c"):
        scene[key] = scene[key][:frame_count]
    scene["vis_start"] = 0
    scene["vis_end"] = frame_count
    return scene


def scene_to_camera_space(scene):
    scene = clone_scene(scene)
    frame_count = len(scene["R_w2c"])
    for hand in ("left", "right"):
        vertices = scene[hand]["vertices"][:, :frame_count]
        scene[hand]["vertices"] = (
            torch.einsum("tij,btvj->btvi", scene["R_w2c"], vertices)
            + scene["t_w2c"][None, :, None, :]
        )
        if "joints" in scene[hand]:
            joints = scene[hand]["joints"][:, :frame_count]
            scene[hand]["joints"] = (
                torch.einsum("tij,btqj->btqi", scene["R_w2c"], joints)
                + scene["t_w2c"][None, :, None, :]
            )
    eye = torch.eye(3).float().repeat(frame_count, 1, 1)
    zeros = torch.zeros(frame_count, 3)
    scene["R_c2w"] = eye
    scene["t_c2w"] = zeros
    scene["R_w2c"] = eye
    scene["t_w2c"] = zeros
    return scene


def overlay_mesh_entries(droid_scene, megasam_scene, gt_scene=None):
    entries = [
        {
            "key": "single_droid_left",
            "name": "DROID left",
            "vertices": droid_scene["left"]["vertices"],
            "faces": droid_scene["left"]["faces"],
            "color": "red",
        },
        {
            "key": "single_droid_right",
            "name": "DROID right",
            "vertices": droid_scene["right"]["vertices"],
            "faces": droid_scene["right"]["faces"],
            "color": "red",
        },
        {
            "key": "single_megasam_left",
            "name": "MegaSAM left",
            "vertices": megasam_scene["left"]["vertices"],
            "faces": megasam_scene["left"]["faces"],
            "color": "director-blue",
        },
        {
            "key": "single_megasam_right",
            "name": "MegaSAM right",
            "vertices": megasam_scene["right"]["vertices"],
            "faces": megasam_scene["right"]["faces"],
            "color": "director-blue",
        },
    ]
    if gt_scene is not None:
        entries.extend(
            [
                {
                    "key": "single_gt_left",
                    "name": "HDF5 GT left",
                    "vertices": gt_scene["left"]["vertices"],
                    "faces": gt_scene["left"]["faces"],
                    "color": "green",
                },
                {
                    "key": "single_gt_right",
                    "name": "HDF5 GT right",
                    "vertices": gt_scene["right"]["vertices"],
                    "faces": gt_scene["right"]["faces"],
                    "color": "green",
                },
            ]
        )
    return entries


def save_overlay_payload(payload_path, args, droid_scene, megasam_scene, img_focal, image_names):
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = min(droid_scene["vis_end"], megasam_scene["vis_end"], len(image_names))
    droid_scene = slice_scene(droid_scene, frame_count)
    megasam_scene = slice_scene(megasam_scene, frame_count)
    gt_scene = None

    if args.vis_mode == "cam":
        droid_scene = scene_to_camera_space(droid_scene)
        megasam_scene = scene_to_camera_space(megasam_scene)
        ref_R_c2w = torch.eye(3).float().repeat(frame_count, 1, 1)
        ref_t_c2w = torch.zeros(frame_count, 3)
        ref_R_w2c = ref_R_c2w
        ref_t_w2c = ref_t_c2w
    else:
        droid_scene, megasam_scene, ref_R_c2w, ref_t_c2w = align_world_overlay_scenes(
            args,
            droid_scene,
            megasam_scene,
        )
        frame_count = min(frame_count, len(ref_R_c2w), len(ref_t_c2w))
        droid_scene = slice_scene(droid_scene, frame_count)
        megasam_scene = slice_scene(megasam_scene, frame_count)
        ref_R_c2w = ref_R_c2w[:frame_count]
        ref_t_c2w = ref_t_c2w[:frame_count]
        ref_R_w2c = ref_R_c2w.transpose(-1, -2)
        ref_t_w2c = -torch.einsum("tij,tj->ti", ref_R_w2c, ref_t_c2w)
        if not args.no_gt_overlay:
            gt_points = load_gt_hand_points_for_overlay(args.video_path, frame_count)
            gt_camera = load_gt_camera_for_overlay(args.video_path)
            if gt_points is not None and gt_camera is not None:
                _, gt_t_c2w = gt_camera
                frame_count = min(frame_count, len(gt_t_c2w))
                if args.overlay_world_alignment != "gt":
                    gt_scale, gt_rot, gt_trans = fit_alignment(
                        gt_t_c2w[:frame_count].cpu().numpy(),
                        ref_t_c2w[:frame_count].cpu().numpy(),
                    )
                    gt_points = transform_gt_points(gt_points, gt_scale, gt_rot, gt_trans)
                gt_points = {hand: points[:frame_count] for hand, points in gt_points.items()}
                if args.overlay_hand_alignment == "median_wrist":
                    droid_scene = apply_median_wrist_offset(droid_scene, gt_points)
                    megasam_scene = apply_median_wrist_offset(megasam_scene, gt_points)
                gt_scene = make_gt_skeleton_scene(gt_points)

    droid_scene = slice_scene(droid_scene, frame_count)
    megasam_scene = slice_scene(megasam_scene, frame_count)
    ref_R_c2w = ref_R_c2w[:frame_count]
    ref_t_c2w = ref_t_c2w[:frame_count]
    ref_R_w2c = ref_R_w2c[:frame_count]
    ref_t_w2c = ref_t_w2c[:frame_count]

    np.savez_compressed(
        payload_path,
        img_focal=np.float32(img_focal),
        image_names=np.asarray([str(Path(path).resolve()) for path in image_names[:frame_count]]),
        droid_left_vertices=droid_scene["left"]["vertices"].cpu().numpy().astype(np.float32),
        droid_right_vertices=droid_scene["right"]["vertices"].cpu().numpy().astype(np.float32),
        megasam_left_vertices=megasam_scene["left"]["vertices"].cpu().numpy().astype(np.float32),
        megasam_right_vertices=megasam_scene["right"]["vertices"].cpu().numpy().astype(np.float32),
        faces_left=np.asarray(droid_scene["left"]["faces"], dtype=np.int32),
        faces_right=np.asarray(droid_scene["right"]["faces"], dtype=np.int32),
        R_c2w=ref_R_c2w.cpu().numpy().astype(np.float32),
        t_c2w=ref_t_c2w.cpu().numpy().astype(np.float32),
        R_w2c=ref_R_w2c.cpu().numpy().astype(np.float32),
        t_w2c=ref_t_w2c.cpu().numpy().astype(np.float32),
        has_gt=np.asarray(gt_scene is not None, dtype=np.bool_),
        gt_left_vertices=gt_scene["left"]["vertices"].cpu().numpy().astype(np.float32)
        if gt_scene is not None
        else np.zeros((0, 0, 3), dtype=np.float32),
        gt_right_vertices=gt_scene["right"]["vertices"].cpu().numpy().astype(np.float32)
        if gt_scene is not None
        else np.zeros((0, 0, 3), dtype=np.float32),
        gt_faces_left=np.asarray(gt_scene["left"]["faces"], dtype=np.int32)
        if gt_scene is not None
        else np.zeros((0, 3), dtype=np.int32),
        gt_faces_right=np.asarray(gt_scene["right"]["faces"], dtype=np.int32)
        if gt_scene is not None
        else np.zeros((0, 3), dtype=np.int32),
    )
    return payload_path


def render_overlay_payload(payload_path, output_dir, vis_mode):
    data = np.load(payload_path, allow_pickle=True)
    faces_left = data["faces_left"]
    faces_right = data["faces_right"]
    gt_scene = None
    if bool(np.asarray(data["has_gt"])):
        gt_scene = {
            "left": {
                "vertices": torch.from_numpy(data["gt_left_vertices"]),
                "faces": data["gt_faces_left"],
            },
            "right": {
                "vertices": torch.from_numpy(data["gt_right_vertices"]),
                "faces": data["gt_faces_right"],
            },
        }
    mesh_entries = overlay_mesh_entries(
        {
            "left": {"vertices": torch.from_numpy(data["droid_left_vertices"]), "faces": faces_left},
            "right": {"vertices": torch.from_numpy(data["droid_right_vertices"]), "faces": faces_right},
        },
        {
            "left": {"vertices": torch.from_numpy(data["megasam_left_vertices"]), "faces": faces_left},
            "right": {"vertices": torch.from_numpy(data["megasam_right_vertices"]), "faces": faces_right},
        },
        gt_scene=gt_scene,
    )
    img_focal = float(data["img_focal"])
    image_names = [str(path) for path in data["image_names"].tolist()]
    if vis_mode == "world":
        video_path = run_vis2_overlay_on_video(
            mesh_entries,
            str(output_dir),
            img_focal,
            image_names,
            R_c2w=torch.from_numpy(data["R_c2w"]),
            t_c2w=torch.from_numpy(data["t_c2w"]),
            interactive=False,
        )
    else:
        video_path = run_vis2_overlay_on_video_cam(
            mesh_entries,
            str(output_dir),
            img_focal,
            image_names,
            interactive=False,
        )
    print(f"Rendered overlay video: {video_path}")
    return Path(video_path)


def render_overlay_subprocess(payload_path, output_dir, vis_mode):
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--render_overlay_payload",
        str(payload_path),
        "--render_output_dir",
        str(output_dir),
        "--render_vis_mode",
        vis_mode,
    ]
    print("$ " + " ".join(command))
    subprocess.run(command, check=True)
    return rendered_video_path(output_dir)


def build_scene_for_backend(args, backend, start_idx, end_idx, seq_folder, frame_chunks_all, faces_left, faces_right):
    if args.reuse_world_results:
        return build_backend_scene_from_world_result(
            args,
            backend,
            start_idx,
            end_idx,
            seq_folder,
            faces_left,
            faces_right,
        )
    return build_backend_scene(args, backend, start_idx, end_idx, seq_folder, frame_chunks_all, faces_left, faces_right)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_focal", type=float)
    parser.add_argument("--video_path", type=str, default="example/video_0.mp4")
    parser.add_argument("--input_type", type=str, default="file")
    parser.add_argument("--checkpoint", type=str, default="./weights/hawor/checkpoints/hawor.ckpt")
    parser.add_argument("--infiller_weight", type=str, default="./weights/hawor/checkpoints/infiller.pt")
    parser.add_argument("--vis_mode", type=str, default="world", choices=["world", "cam"])
    parser.add_argument("--compare_layout", type=str, default="overlay", choices=["overlay", "side_by_side"])
    parser.add_argument("--overlay_world_alignment", type=str, default="gt", choices=["gt", "droid", "none"])
    parser.add_argument(
        "--overlay_hand_alignment",
        type=str,
        default="none",
        choices=["none", "median_wrist"],
        help="Optional GT-assisted hand translation for visualization only.",
    )
    parser.add_argument("--no_gt_overlay", action="store_true", help="Do not draw HDF5 GT skeleton in world overlay mode.")
    parser.add_argument("--megasam_root", type=str, default="/home/user/data/evo/mega-sam")
    parser.add_argument("--megasam_python", type=str, default="/home/user/miniconda3/envs/mega_sam/bin/python")
    parser.add_argument("--force_droid_slam", action="store_true")
    parser.add_argument("--force_megasam_slam", action="store_true")
    parser.add_argument("--force_all_slam", action="store_true")
    parser.add_argument("--force_render", action="store_true")
    parser.add_argument(
        "--reuse_world_results",
        action="store_true",
        help="Render from existing world_space_res_droid/megasam.pth instead of rerunning HaWoR infiller.",
    )
    parser.add_argument("--droid_world", type=str)
    parser.add_argument("--megasam_world", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--render_payload", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--render_overlay_payload", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--render_output_dir", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--render_vis_mode", type=str, choices=["world", "cam"], help=argparse.SUPPRESS)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.render_payload:
        render_payload(
            Path(args.render_payload),
            Path(args.render_output_dir),
            args.render_vis_mode,
        )
        return
    if args.render_overlay_payload:
        render_overlay_payload(
            Path(args.render_overlay_payload),
            Path(args.render_output_dir),
            args.render_vis_mode,
        )
        return

    start_idx, end_idx, seq_folder, imgfiles = detect_track_video(args)
    if args.reuse_world_results:
        frame_chunks_all = None
        img_focal = read_render_focal(args, seq_folder)
    else:
        frame_chunks_all, img_focal = hawor_motion_estimation(args, start_idx, end_idx, seq_folder)
        args.img_focal = img_focal

    force_droid = args.force_all_slam or args.force_droid_slam
    force_megasam = args.force_all_slam or args.force_megasam_slam
    droid_slam_path = ensure_slam(args, "droid", start_idx, end_idx, seq_folder, force=force_droid)
    megasam_slam_path = ensure_slam(args, "megasam", start_idx, end_idx, seq_folder, force=force_megasam)
    assert_megasam_output(megasam_slam_path)

    faces_left, faces_right = build_faces()
    output_root = Path(seq_folder) / "vis_compare"
    if args.compare_layout == "overlay":
        overlay_render_dir = output_root / f"overlay_{args.vis_mode}"
        gt_suffix = "_gt" if args.vis_mode == "world" and not args.no_gt_overlay else ""
        hand_suffix = "_wrist" if args.overlay_hand_alignment == "median_wrist" else ""
        compare_video = (
            Path(args.output)
            if args.output
            else output_root / f"hawor_droid_vs_megasam_{args.vis_mode}_overlay{gt_suffix}{hand_suffix}.mp4"
        )
        if args.force_render or not compare_video.exists():
            droid_scene = build_scene_for_backend(
                args,
                "droid",
                start_idx,
                end_idx,
                seq_folder,
                frame_chunks_all,
                faces_left,
                faces_right,
            )
            megasam_scene = build_scene_for_backend(
                args,
                "megasam",
                start_idx,
                end_idx,
                seq_folder,
                frame_chunks_all,
                faces_left,
                faces_right,
            )
            frame_count = min(droid_scene["vis_end"], megasam_scene["vis_end"], len(imgfiles))
            overlay_payload = save_overlay_payload(
                output_root / f"overlay_{args.vis_mode}_payload.npz",
                args,
                droid_scene,
                megasam_scene,
                img_focal,
                imgfiles[:frame_count],
            )
            overlay_video = render_overlay_subprocess(overlay_payload, overlay_render_dir, args.vis_mode)
            mode_label = "2D camera view" if args.vis_mode == "cam" else "3D world view"
            label = f"{mode_label} overlay: DROID-SLAM red | MegaSAM blue"
            if args.vis_mode == "world" and not args.no_gt_overlay:
                label += " | HDF5 GT green"
            if args.overlay_hand_alignment == "median_wrist":
                label += " | median wrist corrected"
            written = label_video(
                overlay_video,
                compare_video,
                label,
            )
        else:
            written = "reused"
        print(f"Original HaWoR SLAM: {droid_slam_path}")
        print(f"MegaSAM HaWoR SLAM: {megasam_slam_path}")
        print(f"Overlay comparison: {compare_video} ({written} frames)")
        return

    droid_render_dir = output_root / f"droid_{args.vis_mode}"
    megasam_render_dir = output_root / f"megasam_{args.vis_mode}"
    compare_video = Path(args.output) if args.output else output_root / f"hawor_droid_vs_megasam_{args.vis_mode}.mp4"

    droid_video = rendered_video_path(droid_render_dir)
    megasam_video = rendered_video_path(megasam_render_dir)

    if args.force_render or not droid_video.exists():
        droid_scene = build_scene_for_backend(
            args,
            "droid",
            start_idx,
            end_idx,
            seq_folder,
            frame_chunks_all,
            faces_left,
            faces_right,
        )
        droid_payload = save_render_payload(
            output_root / f"droid_{args.vis_mode}_payload.npz",
            droid_scene,
            img_focal,
            imgfiles[droid_scene["vis_start"]:droid_scene["vis_end"]],
        )
        droid_video = render_backend_subprocess(droid_payload, droid_render_dir, args.vis_mode)
    if args.force_render or not megasam_video.exists():
        megasam_scene = build_scene_for_backend(
            args,
            "megasam",
            start_idx,
            end_idx,
            seq_folder,
            frame_chunks_all,
            faces_left,
            faces_right,
        )
        megasam_payload = save_render_payload(
            output_root / f"megasam_{args.vis_mode}_payload.npz",
            megasam_scene,
            img_focal,
            imgfiles[megasam_scene["vis_start"]:megasam_scene["vis_end"]],
        )
        megasam_video = render_backend_subprocess(megasam_payload, megasam_render_dir, args.vis_mode)

    written = stack_videos(
        droid_video,
        megasam_video,
        compare_video,
        "Original HaWoR (DROID-SLAM)",
        "HaWoR with MegaSAM SLAM",
    )
    print(f"Original HaWoR SLAM: {droid_slam_path}")
    print(f"MegaSAM HaWoR SLAM: {megasam_slam_path}")
    print(f"Original render: {droid_video}")
    print(f"MegaSAM render: {megasam_video}")
    print(f"Side-by-side comparison: {compare_video} ({written} frames)")


if __name__ == "__main__":
    main()
