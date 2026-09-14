import os
# ADD THIS: Forces CUDA to report errors synchronously rather than crashing silently
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import Cityscapes
from torchvision.models.segmentation import deeplabv3_resnet50
from tqdm import tqdm

# ==========================================
# 1. Hyperparameters & Configuration
# ==========================================
# Pointing to the root 'cityscapes' folder as seen in your VS Code workspace
DATA_PATH = './cityscapes'
BATCH_SIZE = 2  
EPOCHS = 60
LEARNING_RATE = 0.001
NUM_CLASSES = 19
IGNORE_INDEX = 255

# Check for CUDA availability to handle both GPU and CPU local setups gracefully
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

# ADD THIS: Disable AMP locally. Windows drivers often silently crash on mixed precision convolutions.
USE_AMP = False

# ==========================================
# 2. Dataset & Transforms
# ==========================================
valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
class_map = dict(zip(valid_classes, range(NUM_CLASSES)))

def map_classes(mask):
    mask_np = np.array(mask)
    mapped_mask = np.full(mask_np.shape, IGNORE_INDEX, dtype=np.uint8)
    for raw_c, eval_c in class_map.items():
        mapped_mask[mask_np == raw_c] = eval_c
    return torch.tensor(mapped_mask, dtype=torch.long)

class TransformContext:
    # REDUCED RESOLUTION: Changed from 512x1024 to 256x512 to prevent silent memory crashes
    def __init__(self, crop_size=(256, 512), is_train=False):
        self.crop_size = crop_size
        self.is_train = is_train

    def __call__(self, image, target):
        image = T.functional.resize(image, self.crop_size)
        target = T.functional.resize(target, self.crop_size, interpolation=T.InterpolationMode.NEAREST)
        
        if self.is_train:
            if torch.rand(1).item() > 0.5:
                image = T.functional.hflip(image)
                target = T.functional.hflip(target)
            
            if torch.rand(1).item() > 0.5:
                brightness = 1.0 + torch.empty(1).uniform_(-0.2, 0.2).item()
                image = T.functional.adjust_brightness(image, brightness)
                
        image = T.functional.to_tensor(image)
        image = T.functional.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        target = map_classes(target)
        return image, target

# ==========================================
# 3. Metric Tracking
# ==========================================
class SegmentationMetrics:
    def __init__(self, num_classes, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.hist = np.zeros((num_classes, num_classes))

    def update(self, pred, target):
        pred = pred.argmax(dim=1).flatten().cpu().numpy()
        target = target.flatten().cpu().numpy()
        
        valid_idx = target != self.ignore_index
        pred = pred[valid_idx]
        target = target[valid_idx]
        
        self.hist += np.bincount(target * self.num_classes + pred, 
                                 minlength=self.num_classes**2).reshape(self.num_classes, self.num_classes)
                                 
    def compute(self):
        acc = np.diag(self.hist).sum() / self.hist.sum()
        intersection = np.diag(self.hist)
        union = self.hist.sum(axis=1) + self.hist.sum(axis=0) - intersection
        iou = intersection / (union + 1e-10) 
        miou = np.nanmean(iou)
        return acc, miou, iou

    def reset(self):
        self.hist = np.zeros((self.num_classes, self.num_classes))

# ==========================================
# 4. Main Training Loop
# ==========================================
def train_model():
    print(f"Using device: {DEVICE}")

    if not os.path.exists(DATA_PATH):
        print(f"Error: Dataset not found at {DATA_PATH}")
        print("Please ensure you have a folder named 'cityscapes' in your root directory.")
        return None

    try:
        train_dataset = Cityscapes(DATA_PATH, split='train', mode='fine', target_type='semantic', transforms=TransformContext(is_train=True))
        val_dataset = Cityscapes(DATA_PATH, split='val', mode='fine', target_type='semantic', transforms=TransformContext(is_train=False))
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("CRITICAL CHECK: Ensure your images folder is named exactly 'leftImg8bit' and NOT 'leftImg8bit_blurred'.")
        return None

    if len(train_dataset) == 0:
        print("Error: Dataset found, but it contains 0 images.")
        print("Please ensure your images are inside city subfolders (e.g., cityscapes/leftImg8bit/train/aachen/...)")
        return None

    # FIX: Set num_workers=0 to prevent silent multiprocessing crashes on Windows
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = deeplabv3_resnet50(num_classes=NUM_CLASSES).to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.PolynomialLR(optimizer, total_iters=EPOCHS, power=0.9)
    metrics = SegmentationMetrics(num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX)
    
    # Initialize PyTorch AMP Scaler with the new USE_AMP flag
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)

    print("Starting Training...")
    best_miou = 0.0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        start_time = time.time()

        # Wrap train_loader with tqdm for a dynamic progress bar
        train_pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}] Train")
        
        for images, targets in train_pbar:
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()

            # Execute forward pass with Mixed Precision
            with torch.amp.autocast('cuda', enabled=USE_AMP):
                outputs = model(images)['out']
                loss = criterion(outputs, targets)

            # Scale gradients and execute backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
            # Update progress bar with the current loss
            train_pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        scheduler.step()
        
        # Clear CUDA cache before validation to free up memory
        if USE_CUDA:
            torch.cuda.empty_cache()

        model.eval()
        val_loss = 0.0
        metrics.reset()

        with torch.no_grad():
            # Wrap val_loader with tqdm for a dynamic progress bar
            val_pbar = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}] Val  ")
            
            for images, targets in val_pbar:
                images, targets = images.to(DEVICE), targets.to(DEVICE)
                
                # Validation with Mixed Precision
                with torch.amp.autocast('cuda', enabled=USE_AMP):
                    outputs = model(images)['out']
                    loss = criterion(outputs, targets)
                    
                val_loss += loss.item()
                metrics.update(outputs, targets)

        pixel_acc, miou, class_iou = metrics.compute()
        epoch_time = time.time() - start_time

        print(f"\nEpoch [{epoch+1}/{EPOCHS}] Summary - Time: {epoch_time:.2f}s")
        print(f"Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")
        print(f"Val Pixel Acc: {pixel_acc:.4f} | Val mIoU: {miou:.4f}")

        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), 'best_cityscapes_model.pth')
            print(f"--> Saved new best model with mIoU: {best_miou:.4f}")
        print("-" * 50)
        
        # Clear CUDA cache after epoch
        if USE_CUDA:
            torch.cuda.empty_cache()

    return model

if __name__ == '__main__':
    trained_model = train_model()

    if trained_model is not None:
        # ==========================================
        # 5. Export for NVIDIA Jetson Orin Nano
        # ==========================================
        print("Training Complete. Exporting to ONNX for TensorRT deployment on Jetson Orin Nano...")
        trained_model.load_state_dict(torch.load('best_cityscapes_model.pth'))
        trained_model.eval()

        dummy_input = torch.randn(1, 3, 512, 1024).to(DEVICE)
        onnx_path = "cityscapes_deeplabv3.onnx"

        # Explicitly turn off AMP for ONNX export to avoid precision-related export bugs
        with torch.amp.autocast('cuda', enabled=False):
            torch.onnx.export(
                trained_model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            )
        print(f"ONNX model saved to {onnx_path}.")
        print("Next step on your Jetson: run `trtexec --onnx=cityscapes_deeplabv3.onnx --saveEngine=model.engine --fp16`")
    else:
        print("Model training failed or dataset not found. Skipping ONNX export.")