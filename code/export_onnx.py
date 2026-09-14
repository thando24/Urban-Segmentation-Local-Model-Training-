import torch
from torchvision.models.segmentation import deeplabv3_resnet50

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 19
ONNX_PATH = "best_mIoU_epoch78A.onnx"

print(f"Using device: {DEVICE}")
print("Initializing model architecture...")

model = deeplabv3_resnet50(num_classes=NUM_CLASSES).to(DEVICE)

# Fixes the PyTorch warning by explicitly setting weights_only=True
print("Loading saved model weights from 'best_mIoU_epoch78.pt'...")
model.load_state_dict(torch.load('best_mIoU_epoch78.pt', weights_only=True, map_location=DEVICE))
model.eval()

print("Exporting to ONNX for TensorRT deployment on Jetson Orin Nano...")
# Note: Input resolution must match the 256x512 resolution we used during training
dummy_input = torch.randn(1, 3, 512, 1024).to(DEVICE)

# Export the model
torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

print(f"Success! ONNX model saved to {ONNX_PATH}.")
print("Next step on your Jetson: run `trtexec --onnx=best_mIoU_epoch78A.onnx --saveEngine=model.engine --fp16`")