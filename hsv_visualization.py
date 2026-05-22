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


# ============================================================
# INITIAL HSV CONTROLS
# ============================================================

hue_shift = 0
sat_scale = 1.0
val_scale = 1.0


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    # --------------------------------------------------------
    # RGB -> HSV
    # --------------------------------------------------------

    hsv = cv2.cvtColor(
        img,
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

    modified = cv2.cvtColor(
        hsv.astype(np.uint8),
        cv2.COLOR_HSV2RGB
    )

    # --------------------------------------------------------
    # CREATE SIDE-BY-SIDE VIEW
    # --------------------------------------------------------

    comparison = np.hstack([
        img,
        modified
    ])

    comparison_bgr = cv2.cvtColor(
        comparison,
        cv2.COLOR_RGB2BGR
    )

    # --------------------------------------------------------
    # DISPLAY TEXT
    # --------------------------------------------------------

    cv2.putText(
        comparison_bgr,
        f"Hue Shift: {hue_shift}",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.putText(
        comparison_bgr,
        f"Saturation Scale: {sat_scale:.2f}",
        (20,90),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.putText(
        comparison_bgr,
        f"Value Scale: {val_scale:.2f}",
        (20,140),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    # --------------------------------------------------------
    # SHOW
    # --------------------------------------------------------

    cv2.imshow(
        "HSV Visualizer",
        comparison_bgr
    )

    key = cv2.waitKey(30) & 0xFF

    # --------------------------------------------------------
    # CONTROLS
    # --------------------------------------------------------

    # hue
    if key == ord('h'):
        hue_shift += 5

    elif key == ord('j'):
        hue_shift -= 5

    # saturation
    elif key == ord('s'):
        sat_scale += 0.1

    elif key == ord('a'):
        sat_scale -= 0.1

    # value
    elif key == ord('v'):
        val_scale += 0.1

    elif key == ord('c'):
        val_scale -= 0.1

    # reset
    elif key == ord('r'):

        hue_shift = 0
        sat_scale = 1.0
        val_scale = 1.0

    # quit
    elif key == ord('q'):
        break


cv2.destroyAllWindows()