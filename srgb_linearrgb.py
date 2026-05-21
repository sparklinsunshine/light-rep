import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# LOAD IMAGE
# ============================================================

img = cv2.imread(
    "cherry.png"
)

if img is None:
    raise ValueError("Could not load image")

img = cv2.cvtColor(
    img,
    cv2.COLOR_BGR2RGB
)

img = img.astype(np.float32) / 255.0


# ============================================================
# SRGB <-> LINEAR FUNCTIONS
# ============================================================

def srgb_to_linear(img):

    return np.power(img, 2.2)


def linear_to_srgb(img):

    img = np.clip(img, 0, 1)

    return np.power(img, 1.0 / 2.2)


# ============================================================
# CONVERT TO LINEAR
# ============================================================

linear_img = srgb_to_linear(img)


# ============================================================
# BRIGHTNESS SCALING
# ============================================================

# simulate doubling light intensity

linear_scaled = np.clip(
    linear_img * 2.0,
    0,
    1
)

# convert back for display
linear_scaled_display = linear_to_srgb(
    linear_scaled
)

# WRONG WAY:
# scaling directly in srgb

srgb_scaled = np.clip(
    img * 2.0,
    0,
    1
)


# ============================================================
# PICK SAMPLE PIXEL
# ============================================================

y = img.shape[0] // 2
x = img.shape[1] // 2

srgb_pixel = img[y, x]

linear_pixel = linear_img[y, x]

scaled_srgb_pixel = srgb_scaled[y, x]

scaled_linear_pixel = linear_scaled_display[y, x]


print("\n================ PIXEL ANALYSIS ================\n")

print(f"Pixel Location: ({x}, {y})\n")

print("Original sRGB pixel:")
print(srgb_pixel)

print("\nConverted Linear RGB pixel:")
print(linear_pixel)

print("\nScaled directly in sRGB:")
print(scaled_srgb_pixel)

print("\nScaled in Linear RGB then converted back:")
print(scaled_linear_pixel)


# ============================================================
# HISTOGRAMS
# ============================================================

srgb_gray = np.mean(img, axis=2)

linear_gray = np.mean(linear_img, axis=2)

plt.figure(figsize=(12,6))

plt.hist(
    srgb_gray.flatten(),
    bins=100,
    alpha=0.6,
    label="sRGB"
)

plt.hist(
    linear_gray.flatten(),
    bins=100,
    alpha=0.6,
    label="Linear RGB"
)

plt.title("Intensity Distribution")

plt.xlabel("Intensity")

plt.ylabel("Pixel Count")

plt.legend()

plt.savefig(
    "histogram_comparison.png"
)

print("\nSaved: histogram_comparison.png")


# ============================================================
# INTENSITY CURVE VISUALIZATION
# ============================================================

x_vals = np.linspace(0, 1, 1000)

srgb_curve = np.power(
    x_vals,
    1/2.2
)

linear_curve = x_vals

plt.figure(figsize=(10,6))

plt.plot(
    x_vals,
    linear_curve,
    label="Linear RGB"
)

plt.plot(
    x_vals,
    srgb_curve,
    label="sRGB Gamma Curve"
)

plt.title("Linear RGB vs sRGB")

plt.xlabel("Physical Light Intensity")

plt.ylabel("Stored Pixel Value")

plt.legend()

plt.grid(True)

plt.savefig(
    "gamma_curve.png"
)

print("Saved: gamma_curve.png")


# ============================================================
# IMAGE COMPARISONS
# ============================================================

comparison = np.hstack([

    (img * 255).astype(np.uint8),

    (srgb_scaled * 255).astype(np.uint8),

    (linear_scaled_display * 255).astype(np.uint8)

])

comparison = cv2.cvtColor(
    comparison,
    cv2.COLOR_RGB2BGR
)

cv2.imwrite(
    "brightness_scaling_comparison.png",
    comparison
)

print("Saved: brightness_scaling_comparison.png")


# ============================================================
# VISUALIZATION WITH LABELS
# ============================================================

fig, axs = plt.subplots(1,3, figsize=(18,6))

axs[0].imshow(img)
axs[0].set_title("Original sRGB")

axs[1].imshow(srgb_scaled)
axs[1].set_title("Scaled Directly in sRGB")

axs[2].imshow(linear_scaled_display)
axs[2].set_title("Scaled in Linear RGB")

for ax in axs:
    ax.axis("off")

plt.tight_layout()

plt.savefig(
    "visual_comparison.png"
)

print("Saved: visual_comparison.png")