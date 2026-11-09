#!/usr/bin/env python3
"""
DeepLabV3-ResNet18 V2 Cityscapes Whole-Val Headless Evaluation Script
Optimized for NVIDIA Jetson Orin Nano / TensorRT execution.

Evaluates the model across all cities in the Cityscapes validation set,
computes per-image and dataset-level mIoU, per-class IoU, and latency,
and writes a detailed CSV/summary report.
"""

import os
import glob
import time
import shutil
import ctypes
import ctypes.util
import subprocess
import argparse
import numpy as np
import cv2

try:
    import tensorrt as trt
except ImportError as e:
    raise SystemExit(
        "Could not import the 'tensorrt' Python module.\n\n"
        "On Jetson, TensorRT's Python bindings are installed system-wide via apt (as part of\n"
        "nvidia-jetpack) and are usually NOT visible inside an isolated virtualenv.\n\n"
        "Fix: recreate your venv so it can see system packages:\n"
        "  deactivate\n"
        "  rm -rf orin_env\n"
        "  python3 -m venv --system-site-packages orin_env\n"
        "  source orin_env/bin/activate\n"
        "  pip install opencv-python matplotlib \"numpy<2\"\n\n"
        f"(original import error: {e})"
    )

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

TARGET_H, TARGET_W = 512, 1024

CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic_light", "traffic_sign", "vegetation",
    "terrain", "sky", "person", "rider", "car",
    "truck", "bus", "train", "motorcycle", "bicycle"
]

CITYSCAPES_PALETTE = np.array([
    [128,  64, 128], [244,  35, 232], [ 70,  70,  70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170,  30], [220, 220,   0],
    [107, 142,  35], [152, 251, 152], [ 70, 130, 180], [220,  20,  60],
    [255,   0,   0], [  0,   0, 142], [  0,   0,  70], [  0,  60, 100],
    [  0,  80, 100], [  0,   0, 230], [119,  11,  32],
], dtype=np.uint8)


# ============================================================================
# Mask Processing & Metric Calculations (V2 Optimized)
# ============================================================================

def rgb_to_class_ids(mask_rgb: np.ndarray) -> np.ndarray:
    """Fast, lossless exact color matching to 19 classes (255 = void/ignore)."""
    class_ids = np.full(mask_rgb.shape[:2], 255, dtype=np.uint8)
    for class_id, color in enumerate(CITYSCAPES_PALETTE):
        class_ids[np.all(mask_rgb == color, axis=-1)] = class_id
    return class_ids


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in enumerate(CITYSCAPES_PALETTE):
        color_mask[mask == class_id] = color
    return color_mask


def calculate_per_class_iou(pred_mask: np.ndarray, gt_mask: np.ndarray, num_classes: int = 19, ignore_index: int = 255):
    """Computes per-class IoU and mean IoU for a single image."""
    valid_mask = (gt_mask != ignore_index)
    class_ious = []
    valid_ious = []

    for c in range(num_classes):
        pred_c = (pred_mask == c) & valid_mask
        gt_c = (gt_mask == c) & valid_mask
        intersection = np.logical_and(pred_c, gt_c).sum()
        union = np.logical_or(pred_c, gt_c).sum()

        if union == 0:
            iou = float('nan')
        else:
            iou = (intersection / union) * 100.0
            valid_ious.append(iou)

        class_ious.append(iou)

    miou = float(np.mean(valid_ious)) if len(valid_ious) > 0 else 0.0
    return class_ious, miou


def update_confusion(conf: np.ndarray, pred: np.ndarray, gt: np.ndarray, num_classes: int = 19):
    """Accumulates dataset-level confusion matrix (rows = ground truth, cols = predicted)."""
    valid = (gt != 255)
    idx = gt[valid].astype(np.int64) * num_classes + pred[valid].astype(np.int64)
    conf += np.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def metrics_from_confusion(conf: np.ndarray):
    """Calculates official benchmark dataset mIoU, per-class IoU, and pixel accuracy."""
    inter = np.diag(conf).astype(np.float64)
    union = conf.sum(axis=0) + conf.sum(axis=1) - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
    pixel_acc = inter.sum() / max(conf.sum(), 1)
    return np.nanmean(iou) * 100.0, iou * 100.0, pixel_acc * 100.0


def preprocess_image(image_bgr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = (resized / 255.0 - mean) / std
    input_tensor = np.transpose(normalized, (2, 0, 1)).astype(np.float32)
    return np.expand_dims(input_tensor, axis=0)


def logits_to_pred(output_tensor: np.ndarray, h_orig: int, w_orig: int) -> np.ndarray:
    if output_tensor.ndim == 4:
        pred_ids = np.argmax(output_tensor[0], axis=0).astype(np.uint8)
    else:
        pred_ids = output_tensor[0].astype(np.uint8)
    return cv2.resize(pred_ids, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)


# ============================================================================
# Dataset Traversal (Supports Entire Val Set Across All Cities)
# ============================================================================

def find_dataset_pairs(img_dir: str, mask_dir: str):
    """
    Recursively scans img_dir and mask_dir to pair images with their ground truth masks.
    Supports either the whole val directory (containing all city subfolders) or a single city folder.
    """
    mask_map = {}
    for root, _, files in os.walk(mask_dir):
        for f in files:
            if f.endswith("_gtFine_color.png"):
                prefix = f.replace("_gtFine_color.png", "")
                mask_map[prefix] = os.path.join(root, f)

    pairs = []
    for root, _, files in os.walk(img_dir):
        for f in sorted(files):
            if "_leftImg8bit" in f and not f.startswith("."):
                prefix = f.split("_leftImg8bit")[0]
                if prefix in mask_map:
                    city = os.path.basename(root) if root != img_dir else "val"
                    img_path = os.path.join(root, f)
                    pairs.append((img_path, mask_map[prefix], prefix, city))

    pairs.sort(key=lambda x: (x[3], x[2]))
    return pairs


# ============================================================================
# Minimal CUDA Runtime Bindings via ctypes
# ============================================================================

def _load_libcudart():
    candidates = []
    found = ctypes.util.find_library("cudart")
    if found:
        candidates.append(found)
    candidates += ["libcudart.so"]
    for pattern in ("/usr/local/cuda/lib64/libcudart.so*", "/usr/local/cuda*/lib64/libcudart.so*"):
        candidates += sorted(glob.glob(pattern), reverse=True)
    last_err = None
    for name in candidates:
        try:
            return ctypes.CDLL(name)
        except OSError as e:
            last_err = e
    raise RuntimeError("Could not load libcudart.so. Ensure CUDA is in LD_LIBRARY_PATH.") from last_err

_cudart = _load_libcudart()
_cudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
_cudart.cudaMalloc.restype = ctypes.c_int
_cudart.cudaFree.argtypes = [ctypes.c_void_p]
_cudart.cudaFree.restype = ctypes.c_int
_cudart.cudaMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
_cudart.cudaMemcpyAsync.restype = ctypes.c_int
_cudart.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
_cudart.cudaStreamCreate.restype = ctypes.c_int
_cudart.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
_cudart.cudaStreamSynchronize.restype = ctypes.c_int

_CUDA_MEMCPY_HOST_TO_DEVICE = 1
_CUDA_MEMCPY_DEVICE_TO_HOST = 2

def _cuda_check(err, what):
    if err != 0:
        raise RuntimeError(f"{what} failed with CUDA error code {err}")

def cuda_malloc(nbytes: int) -> ctypes.c_void_p:
    ptr = ctypes.c_void_p()
    _cuda_check(_cudart.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(nbytes)), "cudaMalloc")
    return ptr

def cuda_free(ptr: ctypes.c_void_p):
    _cudart.cudaFree(ptr)

def cuda_stream_create() -> ctypes.c_void_p:
    stream = ctypes.c_void_p()
    _cuda_check(_cudart.cudaStreamCreate(ctypes.byref(stream)), "cudaStreamCreate")
    return stream

def cuda_stream_synchronize(stream: ctypes.c_void_p):
    _cuda_check(_cudart.cudaStreamSynchronize(stream), "cudaStreamSynchronize")

def cuda_memcpy_h2d(dst_ptr: ctypes.c_void_p, src_np: np.ndarray, nbytes: int, stream: ctypes.c_void_p):
    src = src_np.ctypes.data_as(ctypes.c_void_p)
    _cuda_check(_cudart.cudaMemcpyAsync(dst_ptr, src, ctypes.c_size_t(nbytes), _CUDA_MEMCPY_HOST_TO_DEVICE, stream), "cudaMemcpyAsync H2D")

def cuda_memcpy_d2h(dst_np: np.ndarray, src_ptr: ctypes.c_void_p, nbytes: int, stream: ctypes.c_void_p):
    dst = dst_np.ctypes.data_as(ctypes.c_void_p)
    _cuda_check(_cudart.cudaMemcpyAsync(dst, src_ptr, ctypes.c_size_t(nbytes), _CUDA_MEMCPY_DEVICE_TO_HOST, stream), "cudaMemcpyAsync D2H")


# ============================================================================
# TensorRT Engine Build & Inference Handler
# ============================================================================

def find_trtexec(explicit_path: str = None) -> str:
    if explicit_path:
        if os.path.isfile(explicit_path) and os.access(explicit_path, os.X_OK):
            return explicit_path
        raise FileNotFoundError(f"--trtexec_path {explicit_path} is not executable.")
    exe = shutil.which("trtexec")
    if exe:
        return exe
    for candidate in ("/usr/src/tensorrt/bin/trtexec", "/usr/bin/trtexec"):
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    raise FileNotFoundError("Could not find 'trtexec'. Please install TensorRT or specify --trtexec_path.")


def build_or_load_engine(onnx_path: str, engine_path: str, fp16: bool, trtexec_path: str,
                         input_name: str, rebuild: bool = False):
    need_build = rebuild or not (os.path.exists(engine_path) and
                                 os.path.getmtime(engine_path) >= os.path.getmtime(onnx_path))

    if need_build:
        os.makedirs(os.path.dirname(engine_path) or ".", exist_ok=True)
        shape_spec = f"{input_name}:1x3x{TARGET_H}x{TARGET_W}"
        cmd = [
            trtexec_path,
            f"--onnx={onnx_path}",
            f"--saveEngine={engine_path}",
            f"--minShapes={shape_spec}",
            f"--optShapes={shape_spec}",
            f"--maxShapes={shape_spec}",
        ]
        if fp16:
            cmd.append("--fp16")

        print(f"[TensorRT] Building engine with command:\n  {' '.join(cmd)}")
        t0 = time.perf_counter()
        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise RuntimeError(f"trtexec failed with return code {result.returncode}.")
        print(f"[TensorRT] Engine successfully generated in {time.perf_counter() - t0:.1f}s -> {engine_path}")
    else:
        print(f"[TensorRT] Loading cached engine: {engine_path}")

    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError("Failed to deserialize TensorRT engine. Consider running with --rebuild.")
    return engine


class TRTInference:
    def __init__(self, engine, target_h=TARGET_H, target_w=TARGET_W):
        self.engine = engine
        self.context = engine.create_execution_context()
        self.stream = cuda_stream_create()

        self.input_name = None
        self.output_name = None
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_name = name

        self.input_dtype = trt.nptype(engine.get_tensor_dtype(self.input_name))
        self.input_shape = (1, 3, target_h, target_w)
        self.context.set_input_shape(self.input_name, self.input_shape)

        self.output_dtype = trt.nptype(engine.get_tensor_dtype(self.output_name))
        self.output_shape = tuple(self.context.get_tensor_shape(self.output_name))

        self.in_nbytes = int(np.prod(self.input_shape)) * np.dtype(self.input_dtype).itemsize
        self.out_nbytes = int(np.prod(self.output_shape)) * np.dtype(self.output_dtype).itemsize

        self.d_input = cuda_malloc(self.in_nbytes)
        self.d_output = cuda_malloc(self.out_nbytes)

        self.context.set_tensor_address(self.input_name, self.d_input.value)
        self.context.set_tensor_address(self.output_name, self.d_output.value)

    def infer(self, input_np: np.ndarray) -> np.ndarray:
        input_np = np.ascontiguousarray(input_np, dtype=self.input_dtype)
        cuda_memcpy_h2d(self.d_input, input_np, self.in_nbytes, self.stream)
        ok = self.context.execute_async_v3(self.stream.value)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3() returned False.")
        output_np = np.empty(self.output_shape, dtype=self.output_dtype)
        cuda_memcpy_d2h(output_np, self.d_output, self.out_nbytes, self.stream)
        cuda_stream_synchronize(self.stream)
        return output_np

    def close(self):
        cuda_free(self.d_input)
        cuda_free(self.d_output)


# ============================================================================
# Main Evaluation Loop
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="DeepLabV3-ResNet18 V2 Whole-Val Cityscapes Evaluation")
    parser.add_argument("--model", type=str,
                        default="/home/khubonir/Documents/cityscapes_inference/Resnet18v2/deeplabv3_resnet18_optimized.onnx",
                        help="Path to optimized V2 ONNX model")
    parser.add_argument("--img_dir", type=str,
                        default="/home/khubonir/Documents/cityscapes_inference/data/leftImg8bit/leftImg8bit_blurred/val",
                        help="Root validation images folder (e.g. .../val or .../val/frankfurt)")
    parser.add_argument("--mask_dir", type=str,
                        default="/home/khubonir/Documents/cityscapes_inference/data/gtFine_trainvaltest/gtFine/val",
                        help="Root validation mask folder (e.g. .../val or .../val/frankfurt)")
    parser.add_argument("--output_file", type=str,
                        default="Resnet18v2/evaluation_results_val.txt",
                        help="Destination file for detailed per-image data and final summary")
    parser.add_argument("--input_name", type=str, default="input_image",
                        help="ONNX input tensor name exported in V2")
    parser.add_argument("--no_fp16", action="store_true",
                        help="Build engine in FP32 precision instead of FP16")
    parser.add_argument("--rebuild", action="store_true",
                        help="Force rebuild of the TensorRT engine")
    parser.add_argument("--trt_cache_dir", type=str, default="trt_cache",
                        help="Directory to cache the compiled .engine file")
    parser.add_argument("--trtexec_path", type=str, default=None,
                        help="Path to custom trtexec binary if not in PATH")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Optional folder to save colored prediction masks")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional limit on number of images to evaluate")
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        raise SystemExit(f"ONNX model not found: {args.model}")

    dataset_pairs = find_dataset_pairs(args.img_dir, args.mask_dir)
    if args.limit:
        dataset_pairs = dataset_pairs[:args.limit]

    total_imgs = len(dataset_pairs)
    print(f"Loaded {total_imgs} image-mask pairs across all cities in: {args.img_dir}")
    if total_imgs == 0:
        print("No matching pairs found. Check --img_dir and --mask_dir paths.")
        return

    # Count breakdown per city
    city_counts = {}
    for _, _, _, city in dataset_pairs:
        city_counts[city] = city_counts.get(city, 0) + 1
    print("Dataset distribution:", ", ".join(f"{c}: {n}" for c, n in sorted(city_counts.items())))

    fp16 = not args.no_fp16
    engine_name = os.path.splitext(os.path.basename(args.model))[0] + (".fp16.engine" if fp16 else ".fp32.engine")
    engine_path = os.path.join(args.trt_cache_dir, engine_name)
    trtexec_path = find_trtexec(args.trtexec_path)

    engine = build_or_load_engine(
        args.model, engine_path, fp16=fp16, trtexec_path=trtexec_path,
        input_name=args.input_name, rebuild=args.rebuild
    )
    trt_infer = TRTInference(engine, TARGET_H, TARGET_W)

    print("Warming up TensorRT engine...")
    dummy_input = np.zeros(trt_infer.input_shape, dtype=trt_infer.input_dtype)
    for _ in range(3):
        trt_infer.infer(dummy_input)

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    # Initialize Dataset-level Confusion Matrix (19x19)
    dataset_conf = np.zeros((19, 19), dtype=np.int64)

    # Prepare output CSV structure
    class_headers = ",".join([f"iou_{name}" for name in CLASS_NAMES])
    header_line = f"index,formatted_index,city,prefix,engine_time_ms,pipeline_time_ms,miou_percent,{class_headers}\n"

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)

    all_records = []
    engine_times = []
    pipeline_times = []

    print("\nStarting evaluation...")
    print(f"{'Index':<10} | {'City':<12} | {'Prefix':<26} | {'Engine (ms)':<11} | {'Pipeline (ms)':<13} | {'mIoU (%)':<8}")
    print("-" * 92)

    with open(args.output_file, "w") as f_out:
        f_out.write(header_line)

        try:
            for idx, (img_path, mask_path, prefix, city) in enumerate(dataset_pairs, start=1):
                idx_str = f"{idx}/{total_imgs}"

                orig_bgr = cv2.imread(img_path)
                gt_bgr = cv2.imread(mask_path)
                if orig_bgr is None or gt_bgr is None:
                    print(f"Skipping {prefix}: unable to load image or ground truth.")
                    continue

                t0 = time.perf_counter()

                # Pipeline Step 1: Preprocess
                input_tensor = preprocess_image(orig_bgr, TARGET_H, TARGET_W)

                # Pipeline Step 2: TensorRT Inference (Host -> Device -> Engine -> Device -> Host)
                t1 = time.perf_counter()
                output_tensor = trt_infer.infer(input_tensor)
                t2 = time.perf_counter()

                # Pipeline Step 3: Argmax and Resize back to original dimensions
                h_orig, w_orig = orig_bgr.shape[:2]
                pred_ids = logits_to_pred(output_tensor, h_orig, w_orig)
                t3 = time.perf_counter()

                engine_ms = (t2 - t1) * 1000.0
                pipeline_ms = (t3 - t0) * 1000.0
                engine_times.append(engine_ms)
                pipeline_times.append(pipeline_ms)

                # Ground truth processing (Fast exact RGB match)
                gt_ids = rgb_to_class_ids(cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB))

                # Update dataset confusion matrix
                update_confusion(dataset_conf, pred_ids, gt_ids)

                # Per-image IoU and mIoU calculation
                class_ious, img_miou = calculate_per_class_iou(pred_ids, gt_ids)

                record = {
                    "index": idx,
                    "formatted_index": idx_str,
                    "city": city,
                    "prefix": prefix,
                    "engine_time_ms": engine_ms,
                    "pipeline_time_ms": pipeline_ms,
                    "miou": img_miou,
                    "class_ious": class_ious,
                }
                all_records.append(record)

                if args.save_dir:
                    colored = colorize_mask(pred_ids)
                    colored[gt_ids == 255] = [0, 0, 0]
                    cv2.imwrite(os.path.join(args.save_dir, f"{prefix}_pred.png"), cv2.cvtColor(colored, cv2.COLOR_RGB2BGR))

                # Format row data for CSV output
                class_ious_str = ",".join([f"{iou:.2f}" if not np.isnan(iou) else "NaN" for iou in class_ious])
                row = f"{idx},{idx_str},{city},{prefix},{engine_ms:.4f},{pipeline_ms:.4f},{img_miou:.2f},{class_ious_str}\n"
                f_out.write(row)
                f_out.flush()

                # Console progress log
                print(f"{idx_str:<10} | {city:<12} | {prefix:<26} | {engine_ms:9.2f} ms | {pipeline_ms:11.2f} ms | {img_miou:6.2f}%")

        finally:
            trt_infer.close()

    # ============================================================================
    # Metric Aggregation & Report Generation
    # ============================================================================
    dataset_miou, dataset_per_class_iou, pixel_acc = metrics_from_confusion(dataset_conf)
    mean_img_miou = np.mean([r["miou"] for r in all_records]) if all_records else 0.0
    mean_engine_lat = np.mean(engine_times) if engine_times else 0.0
    mean_pipe_lat = np.mean(pipeline_times) if pipeline_times else 0.0
    engine_fps = 1000.0 / mean_engine_lat if mean_engine_lat > 0 else 0.0
    pipeline_fps = 1000.0 / mean_pipe_lat if mean_pipe_lat > 0 else 0.0

    # Sort best and worst performing images
    sorted_records = sorted(all_records, key=lambda x: x["miou"], reverse=True)
    top_5 = sorted_records[:5]
    lowest_5 = sorted_records[-5:][::-1]

    # Format per-class results
    per_class_str = "\n".join([
        f"  {name:<16}: {'NaN' if np.isnan(iou) else f'{iou:6.2f}%'}"
        for name, iou in zip(CLASS_NAMES, dataset_per_class_iou)
    ])

    summary_text = (
        f"\n{'=' * 75}\n"
        f"CITYSCAPES FULL VALIDATION EVALUATION SUMMARY ({len(all_records)} images processed)\n"
        f"{'=' * 75}\n"
        f"Model: {args.model}\n"
        f"Precision: {'FP16' if fp16 else 'FP32'}\n\n"
        f"ACCURACY BENCHMARKS:\n"
        f"  Dataset-Level mIoU (Official Confusion Matrix) : {dataset_miou:.2f}%\n"
        f"  Mean Per-Image mIoU                            : {mean_img_miou:.2f}%\n"
        f"  Global Pixel Accuracy                          : {pixel_acc:.2f}%\n\n"
        f"LATENCY & THROUGHPUT (Orin Nano):\n"
        f"  Mean TRT Engine Latency : {mean_engine_lat:.2f} ms  ({engine_fps:.1f} FPS)\n"
        f"  Mean Pipeline Latency   : {mean_pipe_lat:.2f} ms  ({pipeline_fps:.1f} FPS)\n\n"
        f"PER-CLASS IoU (%):\n{per_class_str}\n\n"
        f"--- TOP 5 HIGHEST mIoU IMAGES ---\n"
    )

    for rank, r in enumerate(top_5, 1):
        summary_text += f"  #{rank}: [{r['city']}] {r['prefix']} | mIoU: {r['miou']:.2f}% | Engine: {r['engine_time_ms']:.2f} ms\n"

    summary_text += f"\n--- 5 LOWEST mIoU IMAGES ---\n"
    for rank, r in enumerate(lowest_5, 1):
        summary_text += f"  #{rank}: [{r['city']}] {r['prefix']} | mIoU: {r['miou']:.2f}% | Engine: {r['engine_time_ms']:.2f} ms\n"
    summary_text += f"{'=' * 75}\n"

    print(summary_text)

    with open(args.output_file, "a") as f_out:
        f_out.write(summary_text)

    print(f"Results successfully saved to: {args.output_file}")


if __name__ == "__main__":
    main()