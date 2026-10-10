"""
Colour-based shadow rejection for segmented detections.

A cast shadow is the ground around it, dimmed: compared with a ring of pixels just outside the
mask it is darker, (almost) never brighter, keeps the ground's colour balance (only turning a bit
bluer from skylight), and is uniform inside. A real object -- even a dark car -- usually breaks at
least one of those (paint colour, windscreen highlights, body edges).

What colour alone can NOT separate is a shadow from a backlit person / silhouette or a black car
with no visible highlights. On footage from VIAT's UAV test videos those always came with high
detector confidence (> 0.9), while shadows mis-detected as objects scored 0.3-0.55. So callers
should only apply this to low-confidence detections (see `max_score` in looks_like_shadow).

Thresholds were chosen on ~500 SAM3-segmented masks from 12 UAV videos (shadows vs cars/people),
checked leave-one-video-out: the rule flags ~70% of shadows (7 of 9 shadows that SAM3 had labelled
'person'/'car') and ~10% of objects -- all of which scored > 0.9, i.e. are excluded by the gate.
"""
import cv2
import numpy as np

MAX_BRIGHTNESS_RATIO = 0.9   # median L inside / median L of surroundings: shadows are darker
MAX_BRIGHTER_FRACTION = 0.15  # share of mask pixels brighter than the surroundings' median
MAX_RG_SHIFT = 0.1           # change in red/(red+green) balance vs the surroundings
MAX_SPREAD = 0.6             # interquartile range of L inside, relative to its median: uniformity
MIN_MASK_AREA = 150          # px; below this there is too little evidence either way
NOISE_FLOOR = 10.0           # L units added to the median when normalising the spread
MIN_LIT_LEVEL = 30           # mean 0-255 channel value below which a pixel has no usable colour


def shadow_features(frame_bgr, polygon):
    """
    Colour statistics of the masked region vs. a ring around it, or None if the mask is too small
    or too close to the frame edge to have usable surroundings.
    """
    height, width = frame_bgr.shape[:2]
    pts = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    x1, y1 = np.floor(pts.min(0)).astype(int)
    x2, y2 = np.ceil(pts.max(0)).astype(int)
    size = max(x2 - x1, y2 - y1, 1)
    # Work on a crop around the mask: much cheaper than full-frame masks for small objects
    pad = size // 2 + 4
    cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
    cx2, cy2 = min(width, x2 + pad + 1), min(height, y2 + pad + 1)
    crop = frame_bgr[cy1:cy2, cx1:cx2]
    mask = np.zeros(crop.shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.round(pts - [cx1, cy1]).astype(np.int32)], 1)
    area = int(mask.sum())
    if area < MIN_MASK_AREA:
        return None

    kernel = np.ones((3, 3), np.uint8)
    ring_width = max(2, int(0.25 * np.sqrt(area)))
    # Skip the soft penumbra at the mask edge: it mixes shadow and ground pixels
    inner = cv2.erode(mask, kernel, iterations=max(1, ring_width // 3))
    if inner.sum() < 50:
        inner = mask
    ring = (cv2.dilate(mask, kernel, iterations=ring_width)
            - cv2.dilate(mask, kernel, iterations=max(1, ring_width // 3)))
    inside, around = inner > 0, ring > 0
    if around.sum() < 50:
        return None

    lightness = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    rgb = crop.astype(np.float32) + 1.0
    l_in, l_around = lightness[inside], lightness[around]
    med_in, med_around = np.median(l_in), np.median(l_around)

    def spread(values, median):
        # Interquartile range relative to brightness, with a noise floor so very dark regions
        # (sensor/JPEG noise dominates) don't look artificially "varied"
        return (np.percentile(values, 75) - np.percentile(values, 25)) / (median + NOISE_FLOOR)

    # Colour ratios of near-black pixels are noise: judge colour only where there is some light
    lit_in = rgb[inside][rgb[inside].sum(1) > 3 * MIN_LIT_LEVEL]
    lit_around = rgb[around][rgb[around].sum(1) > 3 * MIN_LIT_LEVEL]
    if len(lit_in) >= 30 and len(lit_around) >= 30:
        # Red-vs-green balance only (BGR order): shadows lit by blue sky legitimately turn bluer,
        # but keep the ground's red/green balance, whereas painted objects usually change it
        rg = lambda px: np.median(px[:, 2] / (px[:, 1] + px[:, 2]))
        rg_shift = float(abs(rg(lit_in) - rg(lit_around)))
    else:
        rg_shift = 0.0  # too dark to have a measurable colour: consistent with a shadow

    return {
        'brightness_ratio': float(med_in / max(1.0, med_around)),
        'brighter_fraction': float(np.mean(l_in > med_around)),
        'rg_shift': rg_shift,
        'spread': float(spread(l_in, med_in)),
    }


def looks_like_shadow(frame_bgr, polygon, score=None, max_score=0.6):
    """
    True if the segmented region looks like a cast shadow rather than an object.
    Detections with `score >= max_score` are always trusted (returns False), because colour
    statistics can't tell a shadow from a backlit person or a black car.
    """
    if score is not None and score >= max_score:
        return False
    if polygon is None or len(polygon) < 3:
        return False
    feats = shadow_features(frame_bgr, polygon)
    if feats is None:
        return False
    return (feats['brightness_ratio'] < MAX_BRIGHTNESS_RATIO
            and feats['brighter_fraction'] < MAX_BRIGHTER_FRACTION
            and feats['rg_shift'] < MAX_RG_SHIFT
            and feats['spread'] < MAX_SPREAD)
