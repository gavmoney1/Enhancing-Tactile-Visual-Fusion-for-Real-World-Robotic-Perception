
import os
import math
import cv2
import numpy as np  # needed for mask array

# --- Config ---
input_root = r"dir matched_frames"
output_root = r"dir masked_frames"
img_size = 224
visible_frac = 0.10  # 10% visible in center

def mask_image_center(img, visible_frac=0.10, size=224):
    """Keep only center square, black out rest."""
    # Resize image
    img_resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    H, W = size, size

    # Center square
    side = int(round(math.sqrt(visible_frac) * min(H, W)))
    cy, cx = H // 2, W // 2
    y0, x0 = cy - side // 2, cx - side // 2
    y1, x1 = y0 + side, x0 + side

    # Binary mask: 255 = visible, 0 = masked
    mask = np.zeros((H, W), dtype="uint8")
    mask[y0:y1, x0:x1] = 255

    # Apply mask
    visible_part = cv2.bitwise_and(img_resized, img_resized, mask=mask)
    return visible_part

# Walk through dataset
for root, _, files in os.walk(input_root):
    for fname in files:
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        in_path = os.path.join(root, fname)
        rel_path = os.path.relpath(in_path, input_root)
        out_path = os.path.join(output_root, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        img = cv2.imread(in_path)
        if img is None:
            print(f"⚠️ Could not open {in_path}")
            continue

        masked = mask_image_center(img, visible_frac, img_size)
        cv2.imwrite(out_path, masked)
        print(f"Saved masked image: {out_path}")
