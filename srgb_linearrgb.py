import cv2
import numpy as np


# ============================================================
# CONVERSION FUNCTIONS
# ============================================================

def srgb_to_linear(img):

    img = img.astype(np.float32) / 255.0

    return np.power(img, 2.2)


def linear_to_srgb(img):

    img = np.clip(img, 0, 1)

    img = np.power(img, 1.0 / 2.2)

    return (img * 255).astype(np.uint8)


# ============================================================
# LOAD IMAGE
# ============================================================

img = cv2.imread(
    "cherry.png"
)

if img is None:
    raise ValueError("Could not load image")


h, w = img.shape[:2]


# ============================================================
# CREATE SOFT RED CIRCLE
# ============================================================

overlay = np.zeros_like(img)

mask = np.zeros((h, w), dtype=np.uint8)

center = (w // 2, h // 2)

radius = 180

cv2.circle(
    overlay,
    center,
    radius,
    (0, 0, 255),
    -1
)

cv2.circle(
    mask,
    center,
    radius,
    255,
    -1
)

# soft alpha edges
mask = cv2.GaussianBlur(
    mask,
    (101, 101),
    0
)

alpha = mask.astype(np.float32) / 255.0

alpha_3 = cv2.merge([
    alpha,
    alpha,
    alpha
])


# ============================================================
# WRONG: sRGB BLENDING
# ============================================================

img_srgb = img.astype(np.float32) / 255.0

overlay_srgb = overlay.astype(np.float32) / 255.0

blend_srgb = (

    alpha_3 * overlay_srgb +

    (1.0 - alpha_3) * img_srgb
)

blend_srgb = (
    blend_srgb * 255
).astype(np.uint8)


# ============================================================
# CORRECT: LINEAR RGB BLENDING
# ============================================================

img_linear = srgb_to_linear(img)

overlay_linear = srgb_to_linear(overlay)

blend_linear = (

    alpha_3 * overlay_linear +

    (1.0 - alpha_3) * img_linear
)

blend_linear = linear_to_srgb(
    blend_linear
)


# ============================================================
# SIDE-BY-SIDE COMPARISON
# ============================================================

comparison = np.hstack([
    blend_srgb,
    blend_linear
])

# labels
cv2.putText(
    comparison,
    "sRGB Blend (WRONG)",
    (50, 60),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.2,
    (255,255,255),
    3
)

cv2.putText(
    comparison,
    "Linear RGB Blend (CORRECT)",
    (w + 50, 60),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.2,
    (255,255,255),
    3
)


# ============================================================
# SAVE RESULTS
# ============================================================

cv2.imwrite(
    "blend_srgb_wrong.png",
    blend_srgb
)

cv2.imwrite(
    "blend_linear_correct.png",
    blend_linear
)

cv2.imwrite(
    "blend_comparison.png",
    comparison
)

print("Saved:")
print(" - blend_srgb_wrong.png")
print(" - blend_linear_correct.png")
print(" - blend_comparison.png")