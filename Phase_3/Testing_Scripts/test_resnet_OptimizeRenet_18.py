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
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox

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

# Must match the resolution the ONNX model was exported at (1 x 3 x 512 x 1024)
TARGET_H, TARGET_W = 512, 1024

# --- CITYSCAPES 19-CLASS NAMES + COLOR PALETTE (RGB) ---
CITYSCAPES_CLASSES = [
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
    "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
    "truck", "bus", "train", "motorcycle", "bicycle",
]
CITYSCAPES_PALETTE = np.array([
    [128,  64, 128], [244,  35, 232], [ 70,  70,  70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170,  30], [220, 220,   0],
    [107, 142,  35], [152, 251, 152], [ 70, 130, 180], [220,  20,  60],
    [255,   0,   0], [  0,   0, 142], [  0,   0,  70], [  0,  60, 100],
    [  0,  80, 100], [  0,   0, 230], [119,  11,  32],
], dtype=np.uint8)


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in enumerate(CITYSCAPES_PALETTE):
        color_mask[mask == class_id] = color
    return color_mask


def rgb_to_class_ids(mask_rgb: np.ndarray) -> np.ndarray:
    """Convert a gtFine_color.png (RGB) into 19-class ids, 255 = ignore.

    FIX: the old version subtracted uint8 arrays (values wrap around instead of going
    negative) and built a huge H x W x 19 x 3 temporary array, which is very slow and
    memory hungry on a Jetson. The PNGs are lossless, so exact colour matching is
    both correct and much cheaper.
    """
    class_ids = np.full(mask_rgb.shape[:2], 255, dtype=np.uint8)
    for class_id, color in enumerate(CITYSCAPES_PALETTE):
        class_ids[np.all(mask_rgb == color, axis=-1)] = class_id
    return class_ids


def calculate_miou(pred_mask: np.ndarray, gt_mask: np.ndarray, num_classes: int = 19, ignore_index: int = 255) -> float:
    """Per-image mIoU (only classes present in this image). Used for the GUI title."""
    valid_mask = (gt_mask != ignore_index)
    ious = []
    for c in range(num_classes):
        pred_c = (pred_mask == c) & valid_mask
        gt_c = (gt_mask == c) & valid_mask
        intersection = np.logical_and(pred_c, gt_c).sum()
        union = np.logical_or(pred_c, gt_c).sum()
        if union == 0:
            continue
        ious.append(intersection / union)
    return (np.mean(ious) * 100.0) if len(ious) > 0 else 0.0


def update_confusion(conf: np.ndarray, pred: np.ndarray, gt: np.ndarray, num_classes: int = 19):
    """Accumulate a dataset-level confusion matrix (rows = ground truth, cols = prediction)."""
    valid = gt != 255
    idx = gt[valid].astype(np.int64) * num_classes + pred[valid].astype(np.int64)
    conf += np.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def metrics_from_confusion(conf: np.ndarray):
    inter = np.diag(conf).astype(np.float64)
    union = conf.sum(axis=0) + conf.sum(axis=1) - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
    pixel_acc = inter.sum() / max(conf.sum(), 1)
    return np.nanmean(iou) * 100.0, iou * 100.0, pixel_acc * 100.0


def preprocess_image(image_bgr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Identical to the training-time val pipeline: RGB -> resize -> /255 -> ImageNet norm."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = (resized / 255.0 - mean) / std
    input_tensor = np.transpose(normalized, (2, 0, 1)).astype(np.float32)
    return np.expand_dims(input_tensor, axis=0)


def logits_to_pred(output_tensor: np.ndarray, h_orig: int, w_orig: int) -> np.ndarray:
    """(1, 19, H, W) logits -> class-id map resized back to the original image size."""
    if output_tensor.ndim == 4:
        pred_ids = np.argmax(output_tensor[0], axis=0).astype(np.uint8)
    else:
        pred_ids = output_tensor[0].astype(np.uint8)
    return cv2.resize(pred_ids, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)


def find_dataset_pairs(img_dir: str, mask_dir: str):
    img_files = sorted(glob.glob(os.path.join(img_dir, "*_leftImg8bit*.*")))
    pairs = []
    for img_path in img_files:
        base_name = os.path.basename(img_path)
        prefix = base_name.split("_leftImg8bit")[0]
        mask_candidates = glob.glob(os.path.join(mask_dir, f"{prefix}_gtFine_color.png"))
        if mask_candidates:
            pairs.append((img_path, mask_candidates[0], prefix))
    return pairs


# ============================================================================
# Minimal CUDA runtime bindings via ctypes.
# ============================================================================

def _load_libcudart():
    candidates = []
    found = ctypes.util.find_library("cudart")
    if found: candidates.append(found)
    candidates += ["libcudart.so"]
    for pattern in ("/usr/local/cuda/lib64/libcudart.so*", "/usr/local/cuda*/lib64/libcudart.so*"):
        candidates += sorted(glob.glob(pattern), reverse=True)
    last_err = None
    for name in candidates:
        try: return ctypes.CDLL(name)
        except OSError as e: last_err = e
    raise RuntimeError("Could not load libcudart.so (is CUDA from JetPack installed? try "
                       "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH)") from last_err

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
    if err != 0: raise RuntimeError(f"{what} failed with CUDA error code {err}")

def cuda_malloc(nbytes: int) -> ctypes.c_void_p:
    ptr = ctypes.c_void_p()
    _cuda_check(_cudart.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(nbytes)), "cudaMalloc")
    return ptr

def cuda_free(ptr: ctypes.c_void_p): _cudart.cudaFree(ptr)

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
# TensorRT engine build (via trtexec) + inference wrapper
# ============================================================================

def find_trtexec(explicit_path: str = None) -> str:
    if explicit_path:
        if os.path.isfile(explicit_path) and os.access(explicit_path, os.X_OK): return explicit_path
        raise FileNotFoundError(f"--trtexec_path {explicit_path} is not an executable file.")
    exe = shutil.which("trtexec")
    if exe: return exe
    for candidate in ("/usr/src/tensorrt/bin/trtexec",):
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK): return candidate
    raise FileNotFoundError("Could not find 'trtexec'. On JetPack it lives at /usr/src/tensorrt/bin/trtexec.")


def build_or_load_engine(onnx_path: str, engine_path: str, fp16: bool, trtexec_path: str,
                         input_name: str, rebuild: bool = False):
    need_build = rebuild or not (os.path.exists(engine_path) and
                                 os.path.getmtime(engine_path) >= os.path.getmtime(onnx_path))

    if need_build:
        os.makedirs(os.path.dirname(engine_path) or ".", exist_ok=True)
        # FIX: the profile must use the ONNX input tensor's real name. Our export names it
        # 'input_image' (the old script hard-coded 'input', which makes trtexec fail).
        shape = f"{input_name}:1x3x{TARGET_H}x{TARGET_W}"
        cmd = [
            trtexec_path,
            f"--onnx={onnx_path}",
            f"--saveEngine={engine_path}",
            f"--minShapes={shape}",
            f"--optShapes={shape}",
            f"--maxShapes={shape}",
        ]
        if fp16: cmd.append("--fp16")

        print(f"[TensorRT] Building engine (first run only, can take several minutes on the Orin Nano):\n  {' '.join(cmd)}")
        t0 = time.perf_counter()
        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise RuntimeError(f"trtexec exited with code {result.returncode}. "
                               f"If the log says the input tensor was not found, pass --input_name with the "
                               f"correct ONNX input name.")
        print(f"[TensorRT] Engine built in {time.perf_counter() - t0:.1f}s -> {engine_path}")
    else:
        print(f"[TensorRT] Loading cached engine: {engine_path}")

    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError("Failed to deserialize the TensorRT engine. Engines are tied to the exact "
                           "device + TensorRT version: delete the trt_cache folder (or use --rebuild).")
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
        if not ok: raise RuntimeError("TensorRT execute_async_v3() returned False.")
        output_np = np.empty(self.output_shape, dtype=self.output_dtype)
        cuda_memcpy_d2h(output_np, self.d_output, self.out_nbytes, self.stream)
        cuda_stream_synchronize(self.stream)
        return output_np

    def close(self):
        cuda_free(self.d_input)
        cuda_free(self.d_output)


# ============================================================================
# Headless evaluation (no display needed - ideal over SSH)
# ============================================================================

def run_eval(dataset_pairs, trt_infer, limit=None, save_dir=None):
    pairs = dataset_pairs[:limit] if limit else dataset_pairs
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    conf = np.zeros((19, 19), dtype=np.int64)
    t_engine, t_pipeline = [], []

    print(f"Evaluating {len(pairs)} images...")
    for i, (img_path, mask_path, prefix) in enumerate(pairs, 1):
        orig_bgr, gt_bgr = cv2.imread(img_path), cv2.imread(mask_path)
        if orig_bgr is None or gt_bgr is None:
            print(f"  skipped (unreadable): {prefix}")
            continue
        gt_ids = rgb_to_class_ids(cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB))
        h_orig, w_orig = orig_bgr.shape[:2]

        t0 = time.perf_counter()
        input_tensor = preprocess_image(orig_bgr, TARGET_H, TARGET_W)
        t1 = time.perf_counter()
        output_tensor = trt_infer.infer(input_tensor)          # H2D + GPU + D2H
        t2 = time.perf_counter()
        pred = logits_to_pred(output_tensor, h_orig, w_orig)   # argmax + resize (CPU)
        t3 = time.perf_counter()

        t_engine.append(t2 - t1)
        t_pipeline.append(t3 - t0)
        update_confusion(conf, pred, gt_ids)

        if save_dir:
            colored = colorize_mask(pred)
            colored[gt_ids == 255] = [0, 0, 0]
            cv2.imwrite(os.path.join(save_dir, f"{prefix}_pred.png"), cv2.cvtColor(colored, cv2.COLOR_RGB2BGR))

        if i % 25 == 0 or i == len(pairs):
            print(f"  {i}/{len(pairs)} done")

    miou, per_class, pix_acc = metrics_from_confusion(conf)
    print("\n================ RESULTS ================")
    print(f"Images evaluated      : {len(t_engine)}")
    print(f"mIoU (dataset-level)  : {miou:.2f} %")
    print(f"Pixel accuracy        : {pix_acc:.2f} %")
    print(f"TensorRT engine time  : {np.mean(t_engine) * 1000:.1f} ms/img  ({1.0 / np.mean(t_engine):.1f} FPS)  [H2D + GPU + D2H]")
    print(f"End-to-end pipeline   : {np.mean(t_pipeline) * 1000:.1f} ms/img  ({1.0 / np.mean(t_pipeline):.1f} FPS)  [preprocess + engine + argmax/resize]")
    print("\nPer-class IoU (%):")
    for name, v in zip(CITYSCAPES_CLASSES, per_class):
        print(f"  {name:<14}: {'n/a' if np.isnan(v) else f'{v:.1f}'}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="DeepLabV3-ResNet18 (ONNX -> TensorRT) Cityscapes tester for Jetson")
    parser.add_argument("--model", type=str, default="/home/khubonir/Documents/cityscapes_inference/Resnet18v2/deeplabv3_resnet18_optimized.onnx", help="Path to ONNX model")
    parser.add_argument("--img_dir", type=str,
                        default="/home/khubonir/Documents/cityscapes_inference/data/leftImg8bit/leftImg8bit_blurred/val/frankfurt",
                        help="Input images folder")
    parser.add_argument("--mask_dir", type=str,
                        default="/home/khubonir/Documents/cityscapes_inference/data/gtFine_trainvaltest/gtFine/val/frankfurt",
                        help="Ground truth mask folder (needs *_gtFine_color.png)")
    parser.add_argument("--input_name", type=str, default="input_image", help="ONNX input tensor name")
    parser.add_argument("--no_fp16", action="store_true", help="Build in FP32 instead of FP16")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuilding the TensorRT engine")
    parser.add_argument("--trtexec_path", type=str, default=None, help="Explicit path to trtexec")
    parser.add_argument("--trt_cache_dir", type=str, default="trt_cache", help="Cache directory")
    parser.add_argument("--eval", action="store_true", help="Headless mode: evaluate all images, print mIoU + latency, no GUI")
    parser.add_argument("--limit", type=int, default=None, help="Only use the first N images (with --eval)")
    parser.add_argument("--save_dir", type=str, default=None, help="Save coloured predictions here (with --eval)")
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        raise SystemExit(f"ONNX model not found: {args.model}  (use --model /path/to/deeplabv3_resnet18_optimized.onnx)")

    dataset_pairs = find_dataset_pairs(args.img_dir, args.mask_dir)
    print(f"Loaded {len(dataset_pairs)} image-mask pairs.")
    if len(dataset_pairs) == 0:
        print("No image-mask pairs found. Check --img_dir and --mask_dir. Exiting.")
        return

    fp16 = not args.no_fp16
    engine_path = os.path.join(
        args.trt_cache_dir,
        os.path.splitext(os.path.basename(args.model))[0] + (".fp16.engine" if fp16 else ".fp32.engine"))
    trtexec_path = find_trtexec(args.trtexec_path)

    engine = build_or_load_engine(args.model, engine_path, fp16=fp16, trtexec_path=trtexec_path,
                                  input_name=args.input_name, rebuild=args.rebuild)
    target_h, target_w = TARGET_H, TARGET_W
    trt_infer = TRTInference(engine, target_h, target_w)

    print(f"[TensorRT] Input tensor: {trt_infer.input_name} {trt_infer.input_shape} {trt_infer.input_dtype}")
    print(f"[TensorRT] Output tensor: {trt_infer.output_name} {trt_infer.output_shape} {trt_infer.output_dtype}")

    print("Warming up (first inference calls)...")
    t_warmup = time.perf_counter()
    dummy_input = np.zeros(trt_infer.input_shape, dtype=trt_infer.input_dtype)
    for _ in range(3):
        trt_infer.infer(dummy_input)
    print(f"Warm-up complete in {time.perf_counter() - t_warmup:.2f}s")

    if args.eval:
        try:
            run_eval(dataset_pairs, trt_infer, limit=args.limit, save_dir=args.save_dir)
        finally:
            trt_infer.close()
        return

    # ---------------------------- Interactive viewer ----------------------------
    current_idx = 0
    is_busy = False

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    plt.subplots_adjust(top=0.82, bottom=0.12, left=0.03, right=0.97, wspace=0.08)

    im_orig = axes[0].imshow(np.zeros((target_h, target_w, 3), dtype=np.uint8))
    axes[0].set_title("Original Input Image", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    im_gt = axes[1].imshow(np.zeros((target_h, target_w, 3), dtype=np.uint8))
    axes[1].set_title("Ground Truth Mask (Dataset)", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    im_pred = axes[2].imshow(np.zeros((target_h, target_w, 3), dtype=np.uint8))
    axes[2].set_title("DeepLabV3-ResNet18 (ONNX/TensorRT) Prediction", fontsize=12, fontweight="bold")
    axes[2].axis("off")

    axbox = fig.add_axes([0.45, 0.02, 0.1, 0.05])
    text_box = TextBox(axbox, f'Jump to Index (1-{len(dataset_pairs)}): ')

    def submit_index(text):
        nonlocal current_idx
        if is_busy: return
        try:
            val = int(text)
            if 1 <= val <= len(dataset_pairs):
                current_idx = val - 1
                process_and_display(current_idx)
            else:
                print(f"Index out of range. Please enter a number between 1 and {len(dataset_pairs)}.")
        except ValueError:
            print("Invalid input. Please enter an integer.")
        text_box.set_val("")

    text_box.on_submit(submit_index)

    def show_loading(index: int):
        _, _, prefix = dataset_pairs[index]
        fig.suptitle(
            f"Image [{index + 1}/{len(dataset_pairs)}]: {prefix}\n"
            f"Running Inference... Please wait...\n"
            f"Use Left (<-) / Right (->) Arrows or Type Index Below to Navigate",
            fontsize=13, fontweight="bold", color="crimson"
        )
        fig.canvas.draw_idle()
        plt.pause(0.001)

    def process_and_display(index: int):
        nonlocal is_busy
        is_busy = True
        show_loading(index)

        img_path, mask_path, prefix = dataset_pairs[index]
        orig_bgr, gt_bgr = cv2.imread(img_path), cv2.imread(mask_path)
        if orig_bgr is None or gt_bgr is None:
            is_busy = False
            return

        orig_rgb, gt_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB), cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB)
        input_tensor = preprocess_image(orig_bgr, target_h, target_w)

        t_start = time.perf_counter()
        output_tensor = trt_infer.infer(input_tensor)
        inference_time_sec = time.perf_counter() - t_start

        h_orig, w_orig = orig_rgb.shape[:2]
        pred_ids_resized = logits_to_pred(output_tensor, h_orig, w_orig)

        gt_ids = rgb_to_class_ids(gt_rgb)
        miou_pct = calculate_miou(pred_ids_resized, gt_ids)

        pred_colored = colorize_mask(pred_ids_resized)
        pred_colored[gt_ids == 255] = [0, 0, 0]

        im_orig.set_data(orig_rgb)
        im_gt.set_data(gt_rgb)
        im_pred.set_data(pred_colored)

        fig.suptitle(
            f"Image [{index + 1}/{len(dataset_pairs)}]: {prefix}\n"
            f"Engine Time: {inference_time_sec * 1000:.1f} ms ({1.0 / inference_time_sec:.1f} FPS)  |  mIoU: {miou_pct:.2f}%\n"
            f"Use Left (<-) / Right (->) Arrows or Type Index Below to Navigate",
            fontsize=13, fontweight="bold", color="darkblue"
        )
        fig.canvas.draw_idle()
        is_busy = False

    def on_key(event):
        nonlocal current_idx
        if is_busy: return
        if getattr(text_box, 'capturekeystrokes', False) or event.inaxes == axbox:
            return
        if event.key in ['right', 'down', ' ']:
            if current_idx < len(dataset_pairs) - 1:
                current_idx += 1
                process_and_display(current_idx)
        elif event.key in ['left', 'up']:
            if current_idx > 0:
                current_idx -= 1
                process_and_display(current_idx)

    fig.canvas.mpl_connect('key_press_event', on_key)
    process_and_display(current_idx)

    try:
        plt.show()
    finally:
        trt_infer.close()


if __name__ == "__main__":
    main()
