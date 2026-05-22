import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# LOAD IMAGE
# ============================================================

img_bgr = cv2.imread(
    "cherry.png"
)

if img_bgr is None:
    raise ValueError("Could not load image")

img = cv2.cvtColor(
    img_bgr,
    cv2.COLOR_BGR2RGB
).astype(np.float32) / 255.0


# ============================================================
# SRGB <-> LINEAR
# ============================================================

def srgb_to_linear(img):

    return np.power(img, 2.2)


def linear_to_srgb(img):

    img = np.clip(img, 0, 1)

    return np.power(img, 1.0 / 2.2)


# ============================================================
# INITIAL PARAMETERS
# ============================================================

use_linear_hsv = False

sat_scale = 1.0

val_scale = 1.0

hue_shift = 0


# ============================================================
# PIXEL INSPECTION
# ============================================================

clicked_x = 0
clicked_y = 0


def mouse_callback(event, x, y, flags, param):

    global clicked_x, clicked_y

    if event == cv2.EVENT_LBUTTONDOWN:

        clicked_x = x
        clicked_y = y


cv2.namedWindow("Comparison")

cv2.setMouseCallback(
    "Comparison",
    mouse_callback
)


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    # --------------------------------------------------------
    # SELECT RGB SPACE
    # --------------------------------------------------------

    if use_linear_hsv:

        working_rgb = srgb_to_linear(img)

    else:

        working_rgb = img.copy()


    # --------------------------------------------------------
    # RGB -> HSV
    # --------------------------------------------------------

    hsv = cv2.cvtColor(
        (working_rgb * 255).astype(np.uint8),
        cv2.COLOR_RGB2HSV
    ).astype(np.float32)


    # --------------------------------------------------------
    # MODIFY HSV
    # --------------------------------------------------------

    hsv[:,:,0] = (
        hsv[:,:,0] + hue_shift
    ) % 180

    hsv[:,:,1] *= sat_scale

    hsv[:,:,2] *= val_scale

    hsv[:,:,1] = np.clip(
        hsv[:,:,1],
        0,
        255
    )

    hsv[:,:,2] = np.clip(
        hsv[:,:,2],
        0,
        255
    )


    # --------------------------------------------------------
    # HSV -> RGB
    # --------------------------------------------------------

    modified_rgb = cv2.cvtColor(
        hsv.astype(np.uint8),
        cv2.COLOR_HSV2RGB
    ).astype(np.float32) / 255.0


    # --------------------------------------------------------
    # CONVERT BACK IF LINEAR
    # --------------------------------------------------------

    if use_linear_hsv:

        display_img = linear_to_srgb(
            modified_rgb
        )

    else:

        display_img = modified_rgb


    # --------------------------------------------------------
    # SIDE-BY-SIDE
    # --------------------------------------------------------

    comparison = np.hstack([
        img,
        display_img
    ])

    comparison_bgr = cv2.cvtColor(
        (comparison * 255).astype(np.uint8),
        cv2.COLOR_RGB2BGR
    )


    # --------------------------------------------------------
    # PIXEL INFO
    # --------------------------------------------------------

    px = min(clicked_x, img.shape[1]-1)
    py = min(clicked_y, img.shape[0]-1)

    srgb_pixel = img[py, px]

    linear_pixel = srgb_to_linear(
        srgb_pixel
    )

    hsv_srgb = cv2.cvtColor(
        np.uint8([[srgb_pixel * 255]]),
        cv2.COLOR_RGB2HSV
    )[0,0]

    hsv_linear = cv2.cvtColor(
        np.uint8([[linear_pixel * 255]]),
        cv2.COLOR_RGB2HSV
    )[0,0]


    # --------------------------------------------------------
    # DRAW PIXEL
    # --------------------------------------------------------

    cv2.circle(
        comparison_bgr,
        (px, py),
        5,
        (0,0,255),
        -1
    )


    # --------------------------------------------------------
    # TEXT
    # --------------------------------------------------------

    mode_text = (
        "LINEAR RGB -> HSV"
        if use_linear_hsv
        else
        "sRGB -> HSV"
    )

    y0 = 40

    lines = [

        mode_text,

        f"Hue Shift: {hue_shift}",

        f"Saturation Scale: {sat_scale:.2f}",

        f"Value Scale: {val_scale:.2f}",

        f"Pixel: ({px},{py})",

        f"sRGB: {srgb_pixel}",

        f"Linear RGB: {linear_pixel}",

        f"HSV(sRGB): {hsv_srgb}",

        f"HSV(Linear): {hsv_linear}"
    ]

    for i, line in enumerate(lines):

        cv2.putText(
            comparison_bgr,
            line,
            (20, y0 + i*35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,0),
            2
        )


    # --------------------------------------------------------
    # SHOW
    # --------------------------------------------------------

    cv2.imshow(
        "Comparison",
        comparison_bgr
    )


    # --------------------------------------------------------
    # KEYBOARD CONTROLS
    # --------------------------------------------------------

    key = cv2.waitKey(30) & 0xFF


    # saturation
    if key == ord('s'):
        sat_scale += 0.1

    elif key == ord('a'):
        sat_scale -= 0.1


    # value
    elif key == ord('v'):
        val_scale += 0.1

    elif key == ord('c'):
        val_scale -= 0.1


    # hue
    elif key == ord('h'):
        hue_shift += 5

    elif key == ord('j'):
        hue_shift -= 5


    # toggle mode
    elif key == ord('t'):

        use_linear_hsv = not use_linear_hsv


    # reset
    elif key == ord('r'):

        sat_scale = 1.0

        val_scale = 1.0

        hue_shift = 0


    # quit
    elif key == ord('q'):
        break


cv2.destroyAllWindows()