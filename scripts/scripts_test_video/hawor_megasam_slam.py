import hashlib
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__) + "/../..")

import cv2
import h5py
import numpy as np
from glob import glob
from natsort import natsorted
from scipy.spatial.transform import Rotation as R

from lib.pipeline.slam_paths import get_slam_path


DEFAULT_MEGASAM_ROOT = Path("/home/user/data/evo/mega-sam")
DEFAULT_MEGASAM_PYTHON = Path("/home/user/miniconda3/envs/mega_sam/bin/python")


def _get_arg(args, name, default):
    return getattr(args, name, default)


def _sanitize_scene_name(text):
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    text = re.sub(r"_+", "_", text).strip("._")
    return text or "hawor_sequence"


def _scene_name(seq_folder, start_idx, end_idx):
    seq_path = Path(seq_folder).resolve()
    digest = hashlib.sha1(str(seq_path).encode("utf-8")).hexdigest()[:8]
    return _sanitize_scene_name(f"hawor_{seq_path.name}_{start_idx}_{end_idx}_{digest}")


def _run_command(command, cwd, env=None):
    print("$ " + " ".join(shlex.quote(str(part)) for part in command))
    subprocess.run([str(part) for part in command], cwd=str(cwd), env=env, check=True)


def _read_or_write_focal(args, seq_folder):
    focal = _get_arg(args, "img_focal", None)
    focal_path = Path(seq_folder) / "est_focal.txt"
    if focal is not None:
        focal = float(focal)
        focal_path.write_text(str(focal), encoding="utf-8")
        return focal
    try:
        return float(focal_path.read_text(encoding="utf-8").strip())
    except Exception:
        focal = 600.0
        print(f"No focal length provided, use default {focal}")
        focal_path.write_text(str(focal), encoding="utf-8")
        return focal


def _read_egodex_intrinsics(video_path):
    hdf5_path = Path(video_path).with_suffix(".hdf5")
    if not hdf5_path.exists():
        return None
    with h5py.File(hdf5_path, "r") as root:
        if "camera/intrinsic" not in root:
            return None
        intrinsic = np.asarray(root["camera/intrinsic"], dtype=np.float32)
    if intrinsic.shape != (3, 3):
        intrinsic = intrinsic.reshape(3, 3)
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    if not np.isfinite([fx, fy, cx, cy]).all() or fx <= 0 or fy <= 0:
        return None
    print(f"Using EgoDex HDF5 intrinsics: {hdf5_path}")
    return np.asarray([fx, fy, cx, cy], dtype=np.float32)


def _calib_from_images(imgfiles, focal):
    image = cv2.imread(imgfiles[0])
    if image is None:
        raise FileNotFoundError(f"Failed to read first image: {imgfiles[0]}")
    h, w = image.shape[:2]
    return np.asarray([float(focal), float(focal), w / 2.0, h / 2.0], dtype=np.float32)


def _calib_for_megasam(args, seq_folder, imgfiles):
    calib = _read_egodex_intrinsics(args.video_path)
    if calib is not None:
        return calib
    focal = _read_or_write_focal(args, seq_folder)
    return _calib_from_images(imgfiles, focal)


def _write_intrinsics(path, calib):
    fx, fy, cx, cy = [float(value) for value in calib]
    K = np.eye(3, dtype=np.float32)
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, K, fmt="%.9f")
    return path


def _prepare_valid_masks(seq_folder, start_idx, end_idx, imgfiles, mask_dir, force=False):
    masks_path = Path(seq_folder) / f"tracks_{start_idx}_{end_idx}" / "model_masks.npy"
    if not masks_path.exists():
        return None

    mask_dir.mkdir(parents=True, exist_ok=True)
    expected = mask_dir / f"{Path(imgfiles[0]).stem}.png"
    if expected.exists() and not force:
        return mask_dir

    masks = np.load(masks_path, allow_pickle=True)
    masks = np.asarray(masks).astype(bool)
    count = min(len(imgfiles), len(masks))
    print(f"Writing MegaSAM valid masks: {mask_dir}")
    for index in range(count):
        valid = np.uint8(~masks[index]) * 255
        out_path = mask_dir / f"{Path(imgfiles[index]).stem}.png"
        if not cv2.imwrite(str(out_path), valid):
            raise IOError(f"Failed to write valid mask: {out_path}")
    return mask_dir


def _run_unidepth(megasam_python, megasam_root, images_dir, metric_root, scene_name, force):
    scene_dir = metric_root / scene_name
    frame_count = len(list(Path(images_dir).glob("*.jpg"))) + len(list(Path(images_dir).glob("*.png")))
    existing = list(scene_dir.glob("*.npz")) if scene_dir.exists() else []
    if len(existing) >= frame_count and not force:
        print(f"Reuse MegaSAM UniDepth outputs: {scene_dir}")
        return
    if scene_dir.exists():
        shutil.rmtree(scene_dir)

    env = os.environ.copy()
    unidepth_path = str(megasam_root / "UniDepth")
    env["PYTHONPATH"] = unidepth_path if not env.get("PYTHONPATH") else f"{unidepth_path}:{env['PYTHONPATH']}"
    _run_command(
        [
            megasam_python,
            "UniDepth/scripts/demo_mega-sam.py",
            "--scene-name",
            scene_name,
            "--img-path",
            images_dir,
            "--outdir",
            metric_root,
        ],
        cwd=megasam_root,
        env=env,
    )


def _run_megasam_tracking(
    args,
    megasam_python,
    megasam_root,
    images_dir,
    scene_name,
    mono_root,
    metric_root,
    intrinsics_path,
    valid_mask_dir,
):
    weights = Path(_get_arg(args, "megasam_weights", megasam_root / "checkpoints" / "megasam_final.pth"))
    command = [
        megasam_python,
        "camera_tracking_scripts/test_demo.py",
        "--datapath",
        images_dir,
        "--weights",
        weights,
        "--scene_name",
        scene_name,
        "--mono_depth_path",
        mono_root,
        "--metric_depth_path",
        metric_root,
        "--intrinsics",
        intrinsics_path,
        "--depth_source",
        _get_arg(args, "megasam_depth_source", "metric"),
        "--disable_vis",
        "--buffer",
        _get_arg(args, "megasam_buffer", 1024),
        "--beta",
        _get_arg(args, "megasam_beta", 0.3),
        "--filter_thresh",
        _get_arg(args, "megasam_filter_thresh", 2.0),
        "--warmup",
        _get_arg(args, "megasam_warmup", 8),
        "--keyframe_thresh",
        _get_arg(args, "megasam_keyframe_thresh", 2.0),
        "--frontend_thresh",
        _get_arg(args, "megasam_frontend_thresh", 12.0),
        "--frontend_window",
        _get_arg(args, "megasam_frontend_window", 25),
        "--frontend_radius",
        _get_arg(args, "megasam_frontend_radius", 2),
        "--frontend_nms",
        _get_arg(args, "megasam_frontend_nms", 1),
        "--backend_thresh",
        _get_arg(args, "megasam_backend_thresh", 16.0),
        "--backend_radius",
        _get_arg(args, "megasam_backend_radius", 2),
        "--backend_nms",
        _get_arg(args, "megasam_backend_nms", 3),
    ]
    if _get_arg(args, "megasam_disable_full_ba", False):
        command.append("--disable_full_ba")
    if _get_arg(args, "megasam_upsample", False):
        command.append("--upsample")
    if valid_mask_dir is not None:
        command.extend(["--valid_mask_path", valid_mask_dir])

    _run_command(command, cwd=megasam_root)
    return megasam_root / "outputs" / f"{scene_name}_droid.npz"


def _load_megasam_cam_c2w(npz_path):
    with np.load(npz_path) as data:
        if "cam_c2w" not in data:
            raise KeyError(f"{npz_path} does not contain cam_c2w")
        cam_c2w = np.asarray(data["cam_c2w"], dtype=np.float64)
        depths = np.asarray(data["depths"], dtype=np.float32) if "depths" in data else None
    if cam_c2w.ndim != 3 or cam_c2w.shape[1:] != (4, 4):
        raise ValueError(f"cam_c2w must have shape Nx4x4, got {cam_c2w.shape}")
    return cam_c2w, depths


def _fit_pose_count(cam_c2w, target_count):
    if len(cam_c2w) == target_count:
        return cam_c2w
    if len(cam_c2w) > target_count:
        print(f"MegaSAM returned {len(cam_c2w)} poses; trimming to {target_count}")
        return cam_c2w[:target_count]
    if len(cam_c2w) == 0:
        raise ValueError("MegaSAM returned zero poses")
    pad = np.repeat(cam_c2w[-1:,:,:], target_count - len(cam_c2w), axis=0)
    print(f"MegaSAM returned {len(cam_c2w)} poses; padding to {target_count}")
    return np.concatenate([cam_c2w, pad], axis=0)


def _cam_c2w_to_traj(cam_c2w):
    translations = cam_c2w[:, :3, 3]
    quaternions_xyzw = R.from_matrix(cam_c2w[:, :3, :3]).as_quat()
    return np.concatenate([translations, quaternions_xyzw], axis=1).astype(np.float32)


def hawor_megasam_slam(args, start_idx, end_idx, force=False):
    file = args.video_path
    video_root = os.path.dirname(file)
    video = os.path.basename(file).split(".")[0]
    seq_folder = os.path.join(video_root, video)
    video_folder = os.path.join(video_root, video)
    img_folder = (Path(video_folder) / "extracted_images").resolve()
    imgfiles = natsorted(glob(str(img_folder / "*.jpg")))
    if not imgfiles:
        raise FileNotFoundError(f"No extracted images found: {img_folder}")

    save_path = Path(get_slam_path(seq_folder, start_idx, end_idx, "megasam"))
    if save_path.exists() and not force:
        print(f"skip MegaSAM SLAM: {save_path}")
        return str(save_path)

    megasam_root = Path(_get_arg(args, "megasam_root", os.environ.get("MEGASAM_ROOT", DEFAULT_MEGASAM_ROOT))).expanduser().resolve()
    megasam_python = Path(_get_arg(args, "megasam_python", os.environ.get("MEGASAM_PYTHON", DEFAULT_MEGASAM_PYTHON))).expanduser().resolve()
    if not megasam_root.exists():
        raise FileNotFoundError(f"MegaSAM root not found: {megasam_root}")
    if not megasam_python.exists():
        raise FileNotFoundError(f"MegaSAM Python not found: {megasam_python}")

    calib = _calib_for_megasam(args, seq_folder, imgfiles)
    scene_name = _scene_name(seq_folder, start_idx, end_idx)
    cache_root = (Path(seq_folder) / "SLAM" / "megasam_cache").resolve()
    mono_root = cache_root / "Depth-Anything" / "video_visualization"
    metric_root = cache_root / "UniDepth_outputs"
    intrinsics_path = _write_intrinsics(cache_root / "camera_intrinsic.txt", calib)
    valid_mask_dir = _prepare_valid_masks(
        seq_folder,
        start_idx,
        end_idx,
        imgfiles,
        cache_root / "valid_masks",
        force=force,
    )

    print(f"Running MegaSAM SLAM on {video_folder} ...")
    _run_unidepth(megasam_python, megasam_root, img_folder, metric_root, scene_name, force)

    global_npz = megasam_root / "outputs" / f"{scene_name}_droid.npz"
    if global_npz.exists():
        global_npz.unlink()
    global_npz = _run_megasam_tracking(
        args,
        megasam_python,
        megasam_root,
        img_folder,
        scene_name,
        mono_root,
        metric_root,
        intrinsics_path,
        valid_mask_dir,
    )
    if not global_npz.exists():
        raise FileNotFoundError(f"MegaSAM did not create expected output: {global_npz}")

    cache_npz = cache_root / f"{scene_name}_droid.npz"
    cache_npz.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(global_npz, cache_npz)

    cam_c2w, depths = _load_megasam_cam_c2w(cache_npz)
    cam_c2w = _fit_pose_count(cam_c2w, len(imgfiles))
    traj = _cam_c2w_to_traj(cam_c2w)
    if depths is not None and len(depths) > 0:
        disps = 1.0 / np.clip(depths[: len(cam_c2w)], 1e-6, None)
    else:
        disps = np.empty((0,), dtype=np.float32)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        save_path,
        tstamp=np.arange(len(cam_c2w), dtype=np.int32),
        disps=disps.astype(np.float32),
        traj=traj,
        img_focal=float((calib[0] + calib[1]) * 0.5),
        img_focal_xy=calib[:2].astype(np.float32),
        img_center=calib[-2:].astype(np.float32),
        scale=np.float32(1.0),
        cam_c2w=cam_c2w.astype(np.float32),
        source_backend="megasam",
        source_npz=str(cache_npz),
        full_ba=np.asarray(not _get_arg(args, "megasam_disable_full_ba", False)),
    )
    print(f"Saved MegaSAM HaWoR SLAM: {save_path}")
    return str(save_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--img_focal", type=float)
    parser.add_argument("--video_path", type=str, default="example/video_0.mp4")
    parser.add_argument("--megasam_root", type=str, default=str(DEFAULT_MEGASAM_ROOT))
    parser.add_argument("--megasam_python", type=str, default=str(DEFAULT_MEGASAM_PYTHON))
    parser.add_argument("--megasam_disable_full_ba", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int)
    parsed = parser.parse_args()

    if parsed.end_idx is None:
        video_root = os.path.dirname(parsed.video_path)
        video = os.path.basename(parsed.video_path).split(".")[0]
        img_folder = Path(video_root) / video / "extracted_images"
        parsed.end_idx = len(natsorted(glob(str(img_folder / "*.jpg"))))
    hawor_megasam_slam(parsed, parsed.start_idx, parsed.end_idx, force=parsed.force)
