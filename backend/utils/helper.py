import monai
from monai.networks.nets import EfficientNetBN
from monai.networks.nets import densenet121
from monai.visualize import GradCAM
from monai.transforms import (
    Compose,
    LoadImageD,
    EnsureChannelFirstD,
    ScaleIntensityRangeD,
    ResizeD,
    EnsureTypeD,
    MapTransform,
)
from monai.networks.nets import EfficientNetBN
from monai.visualize import GradCAM
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import os
import shutil
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class ToGrayscaleD(MapTransform):
    """
    Ensures image is 1-channel. If loaded as RGB/multi-channel, converts by mean over channels.
    Expects channel-first [C,H,W].
    """

    def __call__(self, data):
        d = dict(data)
        x = d["image"]
        # x can be numpy or torch
        c = x.shape[0]
        if c > 1:
            x = x.mean(axis=0, keepdims=True)
        d["image"] = x
        return d 


def save_heatmap(img: np.ndarray, cam: np.ndarray, output_path: str, alpha: float = 0.4):
    """
    Overlays heatmap on original image using matplotlib and saves as image file.
    img: [H, W] normalized to [0, 1]
    cam: [h, w] raw CAM values
    """

    # Interpolate cam -> same shape as img
    cam_t = torch.tensor(cam)[None, None, ...]
    cam_interp = torch.nn.functional.interpolate(
        cam_t,
        size=img.shape,  # (H, W)
        mode="bilinear",
        align_corners=False,
    )[0, 0].numpy()

    # ----------------------------
    # Flip + rotate 90° right
    # ----------------------------
    # choose ONE flip type depending on what you mean by "flip":
    # vertical flip:
    img = np.flipud(img)
    cam_interp = np.flipud(cam_interp)

    # rotate 90° to the right (clockwise)
    img = np.rot90(img, k=-1)
    cam_interp = np.rot90(cam_interp, k=-1)

    # (Optional) ensure contiguous arrays for matplotlib safety
    img = np.ascontiguousarray(img)
    cam_interp = np.ascontiguousarray(cam_interp)

    # Create figure with exact dimensions
    height, width = img.shape
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    ax.axis("off")

    # Background image
    ax.imshow(img, cmap="gray", origin="upper")

    # Heatmap overlay
    ax.imshow(cam_interp, cmap="jet", alpha=alpha, vmin=0, vmax=1.0, origin="upper")

    plt.savefig(output_path, bbox_inches="tight", pad_inches=0, transparent=False)
    plt.close(fig)

# def save_heatmap(img: np.ndarray, cam: np.ndarray, output_path: str, alpha: float = 0.4):
#     """
#     Overlays heatmap on original image using matplotlib and saves as image file.
#     img: [H, W] normalized to [0, 1]
#     cam: [h, w] raw CAM values
#     """
#     # Ensure cam is interpolated to the exact same shape as img
#     # Explicitly map (h, w) to (H, W)
#     cam_t = torch.tensor(cam)[None, None, ...]
#     cam_interp = torch.nn.functional.interpolate(
#         cam_t,
#         size=img.shape,  # (H, W)
#         mode="bilinear",
#         align_corners=False,
#     )[0, 0].numpy()

#     # Create figure with exact dimensions to avoid cropping or padding issues
#     height, width = img.shape
#     fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
#     ax.axis('off')
    
#     # Background image
#     ax.imshow(img, cmap="gray", origin='upper')
    
#     # Heatmap overlay
#     # Ensure colormap Jet is applied to normalized CAM
#     ax.imshow(cam_interp, cmap="jet", alpha=alpha, vmin=0, vmax=1.0, origin='upper')
    
#     # Save image with minimal whitespace
#     plt.savefig(output_path, bbox_inches='tight', pad_inches=0, transparent=False)
#     plt.close(fig)


def save_image(img: np.ndarray, output_path: str):
    """
    Saves the preprocessed image as an image file.
    img: [H, W] normalized to [0, 1]
    """
    fig, ax = plt.subplots(figsize=(img.shape[1] / 100, img.shape[0] / 100), dpi=100)
    ax.axis('off')
    ax.imshow(img, cmap="gray")
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def clear_temp_storage(temp_dir: str):
    """
    Clears the temporary directory for heatmaps.
    """
    if os.path.exists(temp_dir):
        for filename in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    else:
        os.makedirs(temp_dir, exist_ok=True)

# -----------------------------
# 6) Inference + Grad-CAM + overlay plotting
# -----------------------------

def disable_inplace_relu(m):
    for mod in m.modules():
        if isinstance(mod, nn.ReLU):
            mod.inplace = False

def find_last_conv_name(model: nn.Module) -> str:
    last = None
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            last = name
    if last is None:
        raise RuntimeError("No Conv2d layer found in model.")
    return last 

def load_checkpoint(ckpt_path: str, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)
    # model = densenet121(spatial_dims=2, in_channels=1, out_channels=NUM_CLASSES).to(
    #     device
    # )
    NUM_CLASSES = len(ckpt["labels"])
    model = EfficientNetBN(
            model_name="efficientnet-b0",
            spatial_dims=2,
            in_channels=1,
            pretrained=False,
            num_classes=NUM_CLASSES).to(
                device
            )
    model.load_state_dict(ckpt["model"])
    disable_inplace_relu(model)
    model.eval()
    img_size = tuple(ckpt.get("img_size", (320, 320)))
    return model, ckpt["labels"], img_size, device


def build_single_image_preprocess(img_size=(320, 320)):
    # Explicitly use PILReader to avoid orientation issues common with ITK/nibabel
    pre_tfms = Compose(
        [
            LoadImageD(keys="image", image_only=True, reader="PILReader"),
            EnsureChannelFirstD(keys="image"),
            ToGrayscaleD(keys="image"),
            ScaleIntensityRangeD(
                keys="image", a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True
            ),
            ResizeD(keys="image", spatial_size=img_size),
            EnsureTypeD(keys="image", track_meta=False),
        ]
    )
    return pre_tfms


def infer_with_gradcam(
    model: torch.nn.Module,
    image_path: str = None,
    image_tensor: torch.Tensor = None,
    class_idx: int = 0,
    img_size=(320, 320),
    target_layers: str = "features",
):
    """
    Returns:
      probs: [C] float numpy
      cam:   [h,w] float numpy (0..1), typically lower-res than input
      img:   [H,W] float numpy (0..1) preprocessed image
    """
    device = next(model.parameters()).device
    
    if image_tensor is not None:
        # User provided pre-loaded and preprocessed image from CacheDataset
        x = image_tensor.to(device)
        if x.ndim == 3: # [C, H, W]
            x = x.unsqueeze(0)
    elif image_path is not None:
        # Load and preprocess from path
        pre = build_single_image_preprocess(img_size=img_size)
        sample = {"image": image_path}
        x = pre(sample)["image"]  # [1,H,W] torch tensor
        x = x.unsqueeze(0).to(device)  # [1,1,H,W]
    else:
        raise ValueError("Either image_path or image_tensor must be provided.")

    # Probabilities (no-grad)
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits)[0].detach().cpu().numpy()
    
    target_layers = find_last_conv_name(model)
    # Grad-CAM needs gradients (do NOT wrap in torch.no_grad)
    cam_gen = GradCAM(nn_module=model, target_layers=target_layers)
    cam = cam_gen(x, class_idx=class_idx)  # [1,1,h,w]
    cam = cam[0, 0].detach().cpu().numpy()
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    img = x[0, 0].detach().cpu().numpy()  # [H,W] 0..1
    return probs, cam, img


def overlay_cam(img: np.ndarray, cam: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    """
    img: [H,W] float 0..1
    cam: [h,w] float 0..1
    output: [H,W] float 0..1 (simple blend; for nicer visuals use colormap overlay)
    """
    if cam.shape != img.shape:
        cam_t = torch.tensor(cam)[None, None, ...]
        cam_rs = torch.nn.functional.interpolate(
            cam_t, size=img.shape, mode="bilinear", align_corners=False
        )[0, 0].numpy()
    else:
        cam_rs = cam

    out = (1 - alpha) * img + alpha * cam_rs
    return np.clip(out, 0.0, 1.0)


def visualize_gradcam(image_path: str, ckpt_path: str, class_name: str):
    import matplotlib.pyplot as plt

    model, labels, img_size, device = load_checkpoint(ckpt_path)
    if class_name not in label_to_idx:
        raise ValueError(f"class_name must be one of: {labels}")

    class_idx = label_to_idx[class_name]
    probs, cam, img = infer_with_gradcam(
        model=model,
        image_path=image_path,
        class_idx=class_idx,
        img_size=img_size,
        target_layers="features",
    )

    blended = overlay_cam(img, cam, alpha=0.35)
    # probs = np.where(probs > 0.6, 1, 0)
    topk = np.argsort(-probs)[:5]
    print("Top-5 predicted labels:")
    for i in topk:
        print(f"  {labels[i]:>20s}: {probs[i]:.4f}")

    plt.figure()
    plt.title(f"Grad-CAM blend for: {class_name} (p={probs[class_idx]:.3f})")
    plt.imshow(blended, cmap="gray")
    plt.axis("off")
    plt.show()

    # Optional: show CAM heatmap itself
    plt.figure()
    plt.title("Grad-CAM heatmap")
    plt.imshow(img, cmap="gray")
    plt.imshow(
        torch.nn.functional.interpolate(
            torch.tensor(cam)[None, None, ...],
            size=img.shape,
            mode="bilinear",
            align_corners=False,
        )[0, 0].numpy(),
        cmap="jet",
        alpha=0.4,
    )
    plt.axis("off")
    plt.show()

