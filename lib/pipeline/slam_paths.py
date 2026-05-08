import os


DEFAULT_SLAM_BACKEND = "droid"
MEGASAM_SLAM_BACKEND = "megasam"


def normalize_slam_backend(backend=None):
    if backend is None:
        return DEFAULT_SLAM_BACKEND
    backend = str(backend).lower().replace("-", "_")
    if backend in {"hawor", "droid", "droid_slam", "masked_droid", "masked_droid_slam"}:
        return DEFAULT_SLAM_BACKEND
    if backend in {"mega_sam", "megasam"}:
        return MEGASAM_SLAM_BACKEND
    raise ValueError(f"Unsupported SLAM backend: {backend}")


def get_slam_backend(args):
    return normalize_slam_backend(getattr(args, "slam_backend", DEFAULT_SLAM_BACKEND))


def get_slam_filename(start_idx, end_idx, backend=None):
    backend = normalize_slam_backend(backend)
    if backend == MEGASAM_SLAM_BACKEND:
        return f"hawor_megasam_w_scale_{start_idx}_{end_idx}.npz"
    return f"hawor_slam_w_scale_{start_idx}_{end_idx}.npz"


def get_slam_path(seq_folder, start_idx, end_idx, backend=None):
    return os.path.join(seq_folder, "SLAM", get_slam_filename(start_idx, end_idx, backend))
