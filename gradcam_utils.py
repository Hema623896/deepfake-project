import numpy as np
import torch
import cv2
from PIL import Image


def compute_saliency_overlay(model, input_tensor_cpu, target_class, orig_size=None, colormap=cv2.COLORMAP_JET):
    # input_tensor_cpu: CPU tensor (C,H,W) normalized (the tensor passed to model preprocessing)
    # orig_size: (width, height) to scale the overlay to (optional)
    device = next(model.parameters()).device
    inp = input_tensor_cpu.clone().unsqueeze(0).to(device)
    inp.requires_grad = True

    model.zero_grad()
    out = model(inp)
    score = out[0, target_class]
    score.backward(retain_graph=True)

    if inp.grad is None:
        sal = np.zeros((inp.shape[2], inp.shape[3]), dtype=np.float32)
    else:
        grad = inp.grad.data.squeeze().cpu().numpy()
        sal = np.mean(np.abs(grad), axis=0)

    # normalize saliency
    sal = sal - sal.min()
    if sal.max() > 0:
        sal = sal / sal.max()

    h = input_tensor_cpu.shape[1]
    w = input_tensor_cpu.shape[2]
    if orig_size is not None:
        ow, oh = orig_size
        sal = cv2.resize(sal, (ow, oh))
    else:
        sal = cv2.resize(sal, (w, h))

    # recreate image from normalized tensor (approx) for overlay and resize to orig_size if provided
    img_np = input_tensor_cpu.numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img_np * std + mean)
    img = np.clip(img, 0, 1)
    if orig_size is not None:
        img = cv2.resize(img, (ow, oh))

    heatmap = np.uint8(255 * sal)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = 0.6 * heatmap.astype(np.float32) / 255.0 + 0.4 * img.astype(np.float32)
    overlay = np.clip(overlay, 0, 1)
    overlay_img = Image.fromarray((overlay * 255).astype(np.uint8))
    return overlay_img
