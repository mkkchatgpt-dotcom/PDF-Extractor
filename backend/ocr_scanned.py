"""
OCR support for review comments that are not real text.

Three kinds of comment are handled here:

1. Scanned pages (a page that is one big picture)
   - typed comment sheets ("Point No.1 ...") are read in full, one record
     per numbered point;
   - drawings and forms: only the coloured ink (red, orange, gold, green,
     blue) is cut out and read.
2. Callout boxes drawn as plain shapes (a filled box with a coloured border).
   The text is usually outlines, not text, so the box is rendered and read.
3. Markup on pages that are not scans:
   - flattened highlight / cloud regions (the text underneath is read);
   - native Ink annotations (pen strokes) are rendered and read.

Orientation
-----------
Review pages are often rotated, so the text inside a box can be turned a
quarter turn either way. The OCR engine silently turns every upside-down
line the right way up, which means a wrong turn still produces correct
words but in reverse line order. To avoid that, each region is tested in
every plausible orientation with the engine's own up/down classifier, and
the orientation is chosen by that vote (with a page-wide majority used when
a single box is unclear).

Engine: RapidOCR (pip install rapidocr-onnxruntime==1.2.3). No Tesseract
install is needed, which keeps PyInstaller packaging simple.
"""

import re
import unicodedata

import numpy as np
import pymupdf

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

try:
    import wordninja
except Exception:  # optional: improves spacing of typed pages
    wordninja = None


# ============================================================
# CONFIGURATION
# ============================================================

# A page is treated as a scan when raster images cover at least this
# fraction of the page area.
SCAN_IMAGE_COVERAGE = 0.50

# Render resolution. Capped so huge drawings (A0/A1) don't exhaust memory.
RENDER_DPI = 300
MAX_RENDER_SIDE_PX = 7000

# HSV ranges (OpenCV: H 0-179, S/V 0-255) for review ink on scanned pages.
# Each entry: (name, hue_low, hue_high). Red wraps around 0, so it is two
# entries. Remove an entry to stop reading that colour.
MIN_SATURATION = 70
MIN_VALUE = 80
#
# Every hue is read: red, orange, yellow, green, blue, violet, pink.
# Black, grey and white ink have no saturation and are never read.
INK_COLOURS = [
    ("all-colours", 0, 179),
]

# Flattened highlighter / cloud colours (pastel: moderate saturation, bright).
# Each entry: (name, hue_low, hue_high).
HIGHLIGHT_COLOURS = [
    ("yellow", 18, 35),
    ("green", 40, 85),
    ("blue", 95, 125),
    ("pink", 140, 175),
]
HIGHLIGHT_MIN_SAT = 35
HIGHLIGHT_MAX_SAT = 120
HIGHLIGHT_MIN_VAL = 140
HIGHLIGHT_MAX_VAL = 245
HIGHLIGHT_DPI = 250
HIGHLIGHT_MIN_INK_PIXELS = 600
HIGHLIGHT_MIN_FILL_RATIO = 0.25
HIGHLIGHT_MIN_SIZE_PT = 5
# Text under a highlighter: grey level below this, covering at least this
# share of the highlighted region.
HIGHLIGHT_DARK_TEXT_LEVEL = 110
HIGHLIGHT_MIN_DARK_TEXT_RATIO = 0.02

# Blob filtering
MIN_BLOB_INK_PIXELS = 40
MAX_INK_FILL_RATIO = 0.55  # solid fills are not text
MIN_INK_FILL_RATIO = 0.04  # sparse strokes (arrows, X marks) are not text
MAX_BLOB_PAGE_FRACTION = 0.45  # a "comment" bigger than this is a drawn shape
MIN_OCR_CONFIDENCE = 0.40

# Orientation voting: a single box is trusted when its best orientation
# beats the runner-up by at least this much (pixels of confident text).
ORIENTATION_CONFIDENT_MARGIN = 40.0

_engine = None
_engine_failed = False
_engine_error = None


# ============================================================
# ENGINE
# ============================================================


def ocr_available():
    return cv2 is not None and _get_engine() is not None


def ocr_error():
    """Why OCR is unavailable (None when it works). Shown in the UI/logs."""

    if cv2 is None:
        return "OpenCV (opencv-python-headless) is not installed or failed to load."

    if _get_engine() is None:
        return _engine_error or "RapidOCR engine could not be started."

    return None


def _get_engine():
    """Create the OCR engine once (loading takes a couple of seconds)."""

    global _engine, _engine_failed, _engine_error

    if _engine is not None:
        return _engine

    if _engine_failed:
        return None

    try:
        from rapidocr_onnxruntime import RapidOCR

        _engine = RapidOCR()
        return _engine

    except Exception as exc:
        # Most common cause in the EXE: RapidOCR's .onnx models / config.yaml
        # were not bundled. Keep the reason so it can be shown to the user.
        _engine_failed = True
        _engine_error = f"RapidOCR could not be started: {type(exc).__name__}: {exc}"
        return None


# ============================================================
# TEXT CLEAN-UP
# ============================================================


def _clean_ocr_text(text):
    """Normalise full-width punctuation and strip stray trailing marks."""

    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"[\s,;:|]+$", "", text)

    # An all-capitals code ending in a lower-case "l" is an "I" ("BR-NI").
    text = re.sub(r"\b([A-Z][A-Z\-]*[A-Z])l\b", r"\1I", text)

    return text


RESPACE_MIN_RUN = 10  # only letter runs at least this long are re-split


def _respace(text):
    """
    OCR sometimes drops spaces on long typed lines
    ("shallbeshown", "PointNo.3:RETAININGRING"). Re-split long glued
    letter runs using an English word list, and fix common patterns.
    """

    text = re.sub(
        r"(?i)\b(point|note)\s*(no)\b\.?\s*",
        lambda m: f"{m.group(1).title()} No.",
        text,
    )
    text = re.sub(r"(\d)(of)(\d)", r"\1 of \3", text)
    text = re.sub(
        r"(?i)\b(sheet|page)\s*(\d+)\s*of\s*(\d+)",
        lambda m: f"{m.group(1).title()} {m.group(2)} of {m.group(3)}",
        text,
    )
    text = re.sub(r"(No\.\d+):(?=\S)", r"\1: ", text)

    if wordninja is None:
        return text

    def split_run(match):
        run = match.group(0)

        if len(run) < RESPACE_MIN_RUN:
            return run

        parts = wordninja.split(run)

        if len(parts) > 1 and all(len(part) >= 2 for part in parts):
            return " ".join(parts)

        return run

    text = re.sub(r"[A-Za-z]+", split_run, text)

    text = re.sub(
        r"(?i)(sheet|page)\s*(\d+)\s*of\s*(\d+)",
        lambda m: f"{m.group(1).title()} {m.group(2)} of {m.group(3)}",
        text,
    )

    return re.sub(r"\s+", " ", text).strip()


# ============================================================
# ORIENTATION
# ============================================================

_ROTATIONS = {
    "upright": None,
    "cw": cv2.ROTATE_90_CLOCKWISE if cv2 is not None else None,
    "ccw": cv2.ROTATE_90_COUNTERCLOCKWISE if cv2 is not None else None,
    "flip": cv2.ROTATE_180 if cv2 is not None else None,
}


def _pad(img, border=15):
    return cv2.copyMakeBorder(
        img,
        border,
        border,
        border,
        border,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )


def _rotate(img, name):
    code = _ROTATIONS[name]

    if code is None:
        return np.ascontiguousarray(img)

    return np.ascontiguousarray(cv2.rotate(img, code))


def _orientation_score(engine, img):
    """
    How "upright and horizontal" the text in img is. Higher is better.

    Every detected line votes with its width: lines the engine's up/down
    classifier calls upright count for, upside-down lines count against, and
    vertical lines (a wrong quarter turn) count against. Returns None when
    the engine does not expose the pieces needed (different library version).
    """

    try:
        padded = _pad(img)

        boxes, _ = engine.text_detector(padded)

        if boxes is None or len(boxes) == 0:
            return 0.0

        boxes = engine.sorted_boxes(boxes)
        crops = engine.get_crop_img_list(padded, boxes)
        _, cls_result, _ = engine.text_cls([crop.copy() for crop in crops])

    except Exception:
        return None

    score = 0.0

    for box, (label, confidence) in zip(boxes, cls_result):

        xs = box[:, 0]
        ys = box[:, 1]

        width = float(xs.max() - xs.min())
        height = float(ys.max() - ys.min())

        if height > 1.5 * width:
            # Vertical line: this quarter turn is wrong.
            score -= 0.5 * height
            continue

        vote = 1.0 if str(label) == "0" else -1.0
        score += vote * width * float(confidence)

    return score


def _candidate_names(img):
    height, width = img.shape[:2]

    if height > 1.15 * width:
        return ["cw", "ccw"]

    if width > 1.15 * height:
        return ["upright", "flip"]

    return ["upright", "cw", "ccw", "flip"]


def _group_of(img):
    height, width = img.shape[:2]

    if height > 1.15 * width:
        return "tall"

    if width > 1.15 * height:
        return "wide"

    return "square"


def _score_orientations(engine, img):
    """{orientation name: score} for the plausible orientations, or None."""

    scores = {}

    for name in _candidate_names(img):
        value = _orientation_score(engine, _rotate(img, name))

        if value is None:
            return None

        scores[name] = value

    # Nothing readable in the likely orientations: try the others too.
    if max(scores.values()) <= 0:
        for name in _ROTATIONS:
            if name not in scores:
                value = _orientation_score(engine, _rotate(img, name))

                if value is None:
                    return None

                scores[name] = value

    return scores


def _read_regions(engine, images):
    """
    Read a list of region images (RGB arrays) with orientation voting.

    Returns [(text, orientation_name), ...] in the same order. Regions of
    one page share a page rotation, so a region whose own vote is weak
    follows the majority of its group.
    """

    all_scores = []

    for img in images:
        all_scores.append(_score_orientations(engine, img))

    # Page-wide majority per group (tall / wide / square).
    totals = {}

    for img, scores in zip(images, all_scores):

        if not scores or len(scores) < 2:
            continue

        group = _group_of(img)
        mean = sum(scores.values()) / len(scores)

        for name, value in scores.items():
            totals[(group, name)] = totals.get((group, name), 0.0) + (value - mean)

    outcomes = []

    for img, scores in zip(images, all_scores):

        if scores is None:
            outcomes.append(_read_without_classifier(engine, img))
            continue

        ranked = sorted(scores, key=scores.get, reverse=True)
        best = ranked[0]
        margin = scores[best] - (scores[ranked[1]] if len(ranked) > 1 else 0.0)

        if margin < ORIENTATION_CONFIDENT_MARGIN:
            group = _group_of(img)
            majority = [name for name in scores if (group, name) in totals]

            if majority:
                best = max(majority, key=lambda name: totals[(group, name)])

        outcomes.append((_read_image_text(engine, _rotate(img, best)), best))

    return outcomes


def _read_without_classifier(engine, img):
    """Fallback when the engine has no classifier access: best text quality."""

    best_text, best_quality, best_name = "", -1.0, "upright"

    for name in _candidate_names(img):
        text, quality = _ocr_plain_image(engine, _rotate(img, name))

        if quality > best_quality:
            best_text, best_quality, best_name = text, quality, name

    return best_text, best_name


# ============================================================
# LINE READING
# ============================================================


# Tighter detection boxes than the default (1.6): neighbouring words then
# overlap less, which means fewer letters are read twice.
DETECTION_UNCLIP_RATIO = 1.3


def _run_engine(engine, image):
    """Run the OCR engine; returns its result list (or None)."""

    try:
        result, _ = engine(image, unclip_ratio=DETECTION_UNCLIP_RATIO)
        return result
    except TypeError:
        pass
    except Exception:
        return None

    try:
        result, _ = engine(image)
        return result
    except Exception:
        return None


def _fix_neighbours(ordered):
    """
    Repair the seams between two boxes that sit next to each other on one
    line. The detector's boxes overlap a little, so a letter at the seam is
    sometimes read twice or cut in half:

        "please" + "e update"   ->  please update
        "pers"   + "seal"       ->  per seal
        "Ex-"    + "-Certification" -> Ex-Certification
    """

    texts = [item["text"] for item in ordered]

    for index in range(len(ordered) - 1):

        left, right = ordered[index], ordered[index + 1]
        overlap = left["x1"] - right["x0"]

        left_tokens = texts[index].split(" ")
        right_tokens = texts[index + 1].split(" ")

        # Hyphenated word split across two boxes.
        if re.search(r"-\(?$", texts[index]) and texts[index + 1].startswith("-"):
            texts[index] = re.sub(r"\(?$", "", texts[index])
            texts[index + 1] = "\0" + texts[index + 1][1:]
            continue

        if overlap <= -3:
            continue

        # Dangling single letter at the end of the left box.
        if len(left_tokens) > 1 and len(left_tokens[-1]) == 1 and overlap > 2:
            texts[index] = " ".join(left_tokens[:-1])
            left_tokens = texts[index].split(" ")

        # Left box ends with the letter the right box starts with
        # ("pers" + "seal"): the first one is a half-cut copy.
        last = left_tokens[-1]
        first = right_tokens[0]

        if (
            overlap > 2
            and len(last) >= 3
            and len(first) >= 2
            and last[-1].isalpha()
            and last[-1].lower() == first[0].lower()
        ):
            left_tokens[-1] = last[:-1]
            texts[index] = " ".join(left_tokens)

        # Right box starts with a lone copy of the left box's last letter
        # ("please" + "e update").
        if (
            len(right_tokens) > 1
            and len(first) == 1
            and first.isalpha()
            and texts[index]
            and texts[index][-1].lower() == first.lower()
        ):
            texts[index + 1] = " ".join(right_tokens[1:])

    return texts


def _ocr_plain_image(engine, rgb):
    """OCR an upright image. Returns (text, quality_score)."""

    canvas = _pad(rgb)

    result = _run_engine(engine, canvas)

    if not result:
        return "", 0.0

    items = []

    for box, text, conf in result:

        text = str(text).strip()

        try:
            conf = float(conf)
        except Exception:
            conf = 0.5

        if not text or conf < MIN_OCR_CONFIDENCE:
            continue

        xs = [point[0] for point in box]
        ys = [point[1] for point in box]

        items.append(
            {
                "text": text,
                "conf": conf,
                "x0": min(xs),
                "x1": max(xs),
                "y0": min(ys),
                "y1": max(ys),
            }
        )

    # Drop tiny fragments that sit inside a larger detection
    # (these produce stray letters such as "t these" / "c check").
    kept = []

    for item in items:

        area = max(1.0, (item["x1"] - item["x0"]) * (item["y1"] - item["y0"]))
        fragment = False

        if len(item["text"]) <= 2:
            for other in items:

                if other is item or len(other["text"]) <= 2:
                    continue

                ix = min(item["x1"], other["x1"]) - max(item["x0"], other["x0"])
                iy = min(item["y1"], other["y1"]) - max(item["y0"], other["y0"])

                if ix > 0 and iy > 0 and (ix * iy) / area > 0.05:
                    fragment = True
                    break

        if not fragment:
            kept.append(item)

    if not kept:
        return "", 0.0

    # Group into rows by vertical centre, then read each row left to right.
    heights = sorted(i["y1"] - i["y0"] for i in kept)
    line_height = max(10.0, heights[len(heights) // 2])

    kept.sort(key=lambda i: (i["y0"] + i["y1"]) / 2)

    rows = []

    for item in kept:

        centre = (item["y0"] + item["y1"]) / 2

        if rows and abs(centre - rows[-1]["centre"]) <= 0.6 * line_height:
            rows[-1]["items"].append(item)
            count = len(rows[-1]["items"])
            rows[-1]["centre"] += (centre - rows[-1]["centre"]) / count
        else:
            rows.append({"centre": centre, "items": [item]})

    parts = []
    score = 0.0

    for row in rows:

        ordered = sorted(row["items"], key=lambda i: i["x0"])

        for text, item in zip(_fix_neighbours(ordered), ordered):

            if text == "\0":
                continue

            if text.startswith("\0"):
                # Continuation of a hyphenated word: no space before it.
                if parts:
                    parts[-1] = parts[-1] + text[1:]
                else:
                    parts.append(text[1:])
            elif text:
                parts.append(text)

            score += sum(ch.isalnum() for ch in text) * item["conf"]

    return _clean_ocr_text(" ".join(parts)), score


# ============================================================
# LINE-BY-LINE READING
# ============================================================
#
# The detector sometimes cuts a line into boxes in the middle of a word, which
# leaves stray or doubled letters at the seams. For clean, upright text (a
# callout box, a rendered pen stroke) it is more reliable to cut the image
# into text lines first and give each whole line straight to the recognizer.

LINE_MIN_HEIGHT_PX = 8
LINE_MIN_ASPECT = 1.3  # a text line is clearly wider than it is tall
LINE_PAD_PX = 5
LINE_MIN_CONFIDENCE = 0.30


def _segment_lines(img):
    """Cut an upright image into text-line crops (top to bottom), or None."""

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = (gray < threshold).astype(np.uint8)

    profile = mask.sum(axis=1)

    if profile.max() <= 0:
        return None

    active = profile > max(1, 0.02 * profile.max())

    runs = []
    start = None

    for index, value in enumerate(active):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append([start, index])
            start = None

    if start is not None:
        runs.append([start, len(active)])

    # Join runs separated by a hairline gap (an accent or a descender).
    merged = []

    for run in runs:
        if merged and run[0] - merged[-1][1] <= 3:
            merged[-1][1] = run[1]
        else:
            merged.append(run)

    merged = [run for run in merged if run[1] - run[0] >= LINE_MIN_HEIGHT_PX]

    if not merged:
        return None

    # Tight leading can fuse two lines into one tall run: split at the
    # thinnest row near its middle.
    heights = sorted(run[1] - run[0] for run in merged)
    typical = heights[len(heights) // 2]

    split = []

    for run in merged:

        height = run[1] - run[0]

        if len(merged) > 1 and height > 1.7 * typical:

            low = run[0] + int(height * 0.3)
            high = run[0] + int(height * 0.7)
            cut = low + int(np.argmin(profile[low:high]))

            split.append([run[0], cut])
            split.append([cut, run[1]])

        else:
            split.append(run)

    crops = []

    for top, bottom in split:

        columns = np.where(mask[top:bottom].sum(axis=0) > 0)[0]

        if len(columns) == 0:
            continue

        x0 = max(0, int(columns[0]) - LINE_PAD_PX)
        x1 = min(img.shape[1], int(columns[-1]) + LINE_PAD_PX + 1)
        y0 = max(0, top - LINE_PAD_PX)
        y1 = min(img.shape[0], bottom + LINE_PAD_PX)

        crop = np.ascontiguousarray(img[y0:y1, x0:x1])

        if crop.shape[1] < LINE_MIN_ASPECT * crop.shape[0]:
            return None

        crops.append(crop)

    return crops or None


def _read_lines(engine, img):
    """Read an upright image line by line. Returns text, or None on failure."""

    try:
        crops = _segment_lines(img)

        if not crops:
            return None

        recognised, _elapsed = engine.text_recognizer(crops)

    except Exception:
        return None

    lines = []

    for text, confidence in recognised:

        try:
            if float(confidence) < LINE_MIN_CONFIDENCE:
                continue
        except Exception:
            pass

        text = str(text).strip()

        if text:
            lines.append(text)

    if not lines:
        return None

    joined = lines[0]

    for line in lines[1:]:
        # A hyphen at a line end is a word split across lines ("LGG-" "WT").
        if joined.endswith("-") and line[:1].isalnum():
            joined += line
        else:
            joined += " " + line

    return _clean_ocr_text(joined)


def _read_image_text(engine, img):
    """Upright image -> text: line by line, else with the box detector."""

    text = _read_lines(engine, img)

    if text:
        return text

    text, _quality = _ocr_plain_image(engine, img)

    return text


def _has_content(text, minimum_alnum=2):
    return sum(ch.isalnum() for ch in text) >= minimum_alnum


# ============================================================
# SCAN DETECTION AND RENDERING
# ============================================================


def is_scanned_page(page):
    """
    True when raster images cover most of the page.

    The area covered by ALL images is measured (on a coarse grid, so
    overlapping images are not counted twice). This also catches scanners
    that store one page as several strips or tiles.
    """

    try:
        page_rect = page.rect
        page_area = abs(page_rect.width * page_rect.height)

        if page_area <= 0:
            return False

        grid = np.zeros((100, 100), dtype=bool)
        sx = 100.0 / page_rect.width
        sy = 100.0 / page_rect.height

        for info in page.get_image_info():

            # Image boxes are reported in the displayed (rotated) page
            # space, the same space as page.rect.
            rect = pymupdf.Rect(info["bbox"]) & page_rect

            if rect.is_empty:
                continue

            x0 = int((rect.x0 - page_rect.x0) * sx)
            y0 = int((rect.y0 - page_rect.y0) * sy)
            x1 = int(np.ceil((rect.x1 - page_rect.x0) * sx))
            y1 = int(np.ceil((rect.y1 - page_rect.y0) * sy))

            grid[max(0, y0) : min(100, y1), max(0, x0) : min(100, x1)] = True

        return grid.mean() >= SCAN_IMAGE_COVERAGE

    except Exception:
        return False


def _render_page(page, dpi=RENDER_DPI, annots=False):
    """Render a page to an RGB numpy array. Returns (image, zoom)."""

    zoom = dpi / 72.0
    longest = max(page.rect.width, page.rect.height) * zoom

    if longest > MAX_RENDER_SIDE_PX:
        zoom *= MAX_RENDER_SIDE_PX / longest

    pix = page.get_pixmap(
        matrix=pymupdf.Matrix(zoom, zoom),
        colorspace=pymupdf.csRGB,
        alpha=False,
        annots=annots,
    )

    img = np.frombuffer(pix.samples, dtype=np.uint8)
    img = img.reshape(pix.height, pix.width, 3)

    return img.copy(), zoom


def _pixel_rect_to_page(rect_px, zoom, page):
    """Pixel rectangle of a rendered page -> unrotated page coordinates."""

    rect = pymupdf.Rect(
        rect_px[0] / zoom, rect_px[1] / zoom, rect_px[2] / zoom, rect_px[3] / zoom
    )

    try:
        rect = (rect * page.derotation_matrix).normalize()
    except Exception:
        pass

    return rect


def _page_rect_to_pixels(rect, zoom, page, shape):
    """Unrotated page rectangle -> clamped pixel box (x0, y0, x1, y1)."""

    try:
        rect = (pymupdf.Rect(rect) * page.rotation_matrix).normalize()
    except Exception:
        rect = pymupdf.Rect(rect)

    height, width = shape[:2]

    x0 = max(0, int(rect.x0 * zoom))
    y0 = max(0, int(rect.y0 * zoom))
    x1 = min(width, int(rect.x1 * zoom) + 1)
    y1 = min(height, int(rect.y1 * zoom) + 1)

    return x0, y0, x1, y1


def _ink_canvas(mask):
    """Black ink on white paper: what OCR models are trained on."""

    ink = cv2.dilate(mask, np.ones((2, 2), np.uint8))
    canvas = np.where(ink[..., None] > 0, 0, 255).astype(np.uint8)
    canvas = np.repeat(canvas, 3, axis=2) if canvas.shape[2] == 1 else canvas

    if canvas.shape[0] < 64 and canvas.shape[1] < 64:
        factor = 64 / max(1, min(canvas.shape[:2]))
        canvas = cv2.resize(
            canvas, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC
        )

    return np.ascontiguousarray(canvas)


# ============================================================
# COLOURED INK ON SCANNED PAGES
# ============================================================


def _review_ink_mask(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    strong = (s >= MIN_SATURATION) & (v >= MIN_VALUE)

    wanted = np.zeros(h.shape, dtype=bool)

    for _name, low, high in INK_COLOURS:
        wanted |= (h >= low) & (h <= high)

    mask = (strong & wanted).astype(np.uint8) * 255

    # Remove isolated speckle noise from the scanner.
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def _fallback_read(engine, image):
    """Plain whole-image OCR, used when orientation voting reads nothing."""

    try:
        text, _score = _ocr_plain_image(engine, np.ascontiguousarray(image))
    except Exception:
        return ""

    return text or ""


def extract_scanned_comments(page):
    """
    Return [{"text": str, "rect": pymupdf.Rect}] for coloured review text
    found on a scanned page. Rect is in unrotated page space.
    """

    engine = _get_engine()

    if engine is None or cv2 is None:
        return []

    rgb, zoom = _render_page(page)
    mask = _review_ink_mask(rgb)

    if cv2.countNonZero(mask) < MIN_BLOB_INK_PIXELS:
        return []

    # Join letters -> words -> lines -> comment blobs.
    scale = zoom / (300 / 72.0)
    close_w = max(15, int(55 * scale))
    close_h = max(10, int(40 * scale))

    joined = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (close_w, close_h)),
    )

    count, _, stats, _ = cv2.connectedComponentsWithStats(joined, 8)

    blobs = []
    images = []

    for index in range(1, count):
        x, y, w, h, _ = stats[index]

        blob_mask = mask[y : y + h, x : x + w]
        ink_pixels = cv2.countNonZero(blob_mask)

        if ink_pixels < MIN_BLOB_INK_PIXELS:
            continue

        fill_ratio = ink_pixels / float(w * h)

        if fill_ratio > MAX_INK_FILL_RATIO or fill_ratio < MIN_INK_FILL_RATIO:
            continue

        if (
            w > rgb.shape[1] * MAX_BLOB_PAGE_FRACTION
            or h > rgb.shape[0] * MAX_BLOB_PAGE_FRACTION
        ):
            continue

        blobs.append((x, y, w, h))
        images.append(_ink_canvas(blob_mask))

    results = []

    for (x, y, w, h), image in zip(blobs, images):

        # Handwriting on a scan is upright: the plain reader copes with it
        # better than orientation voting, which is the fallback.
        text = _fallback_read(engine, image)

        # Sideways comment (rotated page / reviewer wrote along the margin):
        # read each plausible quarter turn and keep the clearest reading.
        # The up/down classifier is unreliable on a single thin line.
        if not _has_content(text, 3):
            text = _read_without_classifier(engine, image)[0]

        if not _has_content(text, 3):
            text = _read_regions(engine, [image])[0][0]

        if not _has_content(text, 3):
            continue

        results.append(
            {
                "text": text,
                "rect": _pixel_rect_to_page((x, y, x + w, y + h), zoom, page),
            }
        )

    # Top to bottom, left to right.
    results.sort(key=lambda item: (round(item["rect"].y0 / 10), item["rect"].x0))

    return results


# ============================================================
# TYPED COMMENT SHEETS (read the whole page)
# ============================================================
#
# A scanned *typed* comment sheet ("Point No.1 ... Point No.2 ...") has no
# coloured ink to pick out. Instead the whole page is read and split into
# one record per numbered point.

TEXT_SHEET_DPI = 220
TEXT_SHEET_MAX_LONG_LINES = 3  # page edges / one rule are tolerated
TEXT_SHEET_MIN_ROWS = 3

_POINT_RE = re.compile(
    r"^\s*(?:point\s*no\.?\s*\d+|\(?\d{1,3}\s*[.)]|\(?[a-z]\s*\))",
    re.IGNORECASE,
)
_HEADING_RE = re.compile(r"(:\s*-?\s*$)|^\s*comments?\b", re.IGNORECASE)


def count_long_lines(page):
    """Number of long straight rules on the page (tables, borders, title blocks)."""

    img, _ = _render_page(page, dpi=72)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    height, width = binary.shape

    horizontal = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, int(width * 0.2)), 1)),
    )

    vertical = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, int(height * 0.2)))),
    )

    total = 0

    for part in (horizontal, vertical):
        count, _, _stats, _ = cv2.connectedComponentsWithStats(part, 8)
        total += max(0, count - 1)

    return total


def is_text_sheet(page):
    """Cheap check: scanned page with no tables / drawing lines."""

    try:
        return count_long_lines(page) <= TEXT_SHEET_MAX_LONG_LINES
    except Exception:
        return False


def _ocr_rows(engine, rgb):
    """Whole-page OCR -> list of rows [{text, x0, y0, x1, y1, h}] top to bottom."""

    result = _run_engine(engine, rgb)

    items = []

    for box, text, conf in result or []:

        text = str(text).strip()

        try:
            conf = float(conf)
        except Exception:
            conf = 0.5

        if not text or conf < MIN_OCR_CONFIDENCE:
            continue

        xs = [p[0] for p in box]
        ys = [p[1] for p in box]

        items.append(
            {
                "text": text,
                "x0": min(xs),
                "x1": max(xs),
                "y0": min(ys),
                "y1": max(ys),
            }
        )

    if not items:
        return []

    heights = sorted(i["y1"] - i["y0"] for i in items)
    line_height = max(10.0, heights[len(heights) // 2])

    items.sort(key=lambda i: (i["y0"] + i["y1"]) / 2)

    rows = []

    for item in items:

        centre = (item["y0"] + item["y1"]) / 2

        if rows and abs(centre - rows[-1]["centre"]) <= 0.6 * line_height:
            rows[-1]["items"].append(item)
        else:
            rows.append({"centre": centre, "items": [item]})

    merged = []

    for row in rows:

        parts = sorted(row["items"], key=lambda i: i["x0"])

        merged.append(
            {
                "text": " ".join(p["text"] for p in parts),
                "x0": min(p["x0"] for p in parts),
                "x1": max(p["x1"] for p in parts),
                "y0": min(p["y0"] for p in parts),
                "y1": max(p["y1"] for p in parts),
                "h": line_height,
            }
        )

    return merged


def _best_orientation_rows(engine, rgb):
    """Read the page in the orientation the engine's classifier prefers."""

    def quality(rows):
        return sum(ch.isalnum() for row in rows for ch in row["text"])

    small = rgb
    longest = max(rgb.shape[:2])

    if longest > 1800:
        factor = 1800.0 / longest
        small = cv2.resize(
            rgb, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA
        )

    scores = {}

    for name in _ROTATIONS:
        value = _orientation_score(engine, _rotate(small, name))

        if value is None:
            scores = None
            break

        scores[name] = value

    if scores:
        best = max(scores, key=scores.get)
        turned = _rotate(rgb, best)
        return _ocr_rows(engine, turned), turned

    # Fallback: upright first, other turns only when it reads poorly.
    rows = _ocr_rows(engine, rgb)
    best_rows, best_q, best_img = rows, quality(rows), rgb

    if best_q >= 60:
        return best_rows, best_img

    for name in ("cw", "ccw", "flip"):

        turned = _rotate(rgb, name)
        candidate = _ocr_rows(engine, turned)
        q = quality(candidate)

        if q > best_q:
            best_rows, best_q, best_img = candidate, q, turned

    return best_rows, best_img


def _clean_heading(text):
    text = re.sub(r"[\s:\-]+$", "", text.strip())
    return _respace(re.sub(r"\s+", " ", text))


def _group_rows_into_points(rows):
    """Split OCR rows into [{text, x0, y0, x1, y1}] - one per numbered point."""

    if not rows:
        return []

    line_height = rows[0]["h"]

    point_rows = [r for r in rows if _POINT_RE.match(r["text"])]

    groups = []
    heading = ""

    if point_rows:

        left_margin = min(r["x0"] for r in point_rows)
        current = None
        previous = None

        for row in rows:

            is_point = bool(_POINT_RE.match(row["text"]))
            indented = row["x0"] > left_margin + line_height * 0.8
            gap = (row["y0"] - previous["y1"]) if previous else 0

            if is_point:
                current = {"heading": heading, "rows": [row]}
                groups.append(current)

            elif current is None:
                # Text before the first point: a heading or a title.
                if _HEADING_RE.search(row["text"]):
                    heading = _clean_heading(row["text"])

            elif _HEADING_RE.search(row["text"]) and not indented:
                # New section heading ("Comments on Sheet 2 of 2: -").
                heading = _clean_heading(row["text"])
                current = None

            elif gap > line_height * 2.2:
                # Big gap and not a heading: signature block / footer. Stop.
                current = None

            elif indented or gap <= line_height * 1.6:
                # Wrapped / continuation line of the current point.
                current["rows"].append(row)

            else:
                current = None

            previous = row

    else:
        # No numbering: one record per paragraph.
        current = None
        previous = None

        for row in rows:

            gap = (row["y0"] - previous["y1"]) if previous else 0

            if current is None or gap > line_height * 1.3:
                current = {"heading": "", "rows": [row]}
                groups.append(current)
            else:
                current["rows"].append(row)

            previous = row

    points = []

    for group in groups:

        text = " ".join(r["text"] for r in group["rows"])
        text = _respace(_clean_ocr_text(text))

        if sum(ch.isalnum() for ch in text) < 4:
            continue

        if group["heading"]:
            text = f"[{group['heading']}] {text}"

        points.append(
            {
                "text": text,
                "x0": min(r["x0"] for r in group["rows"]),
                "y0": min(r["y0"] for r in group["rows"]),
                "x1": max(r["x1"] for r in group["rows"]),
                "y1": max(r["y1"] for r in group["rows"]),
            }
        )

    return points


def extract_text_sheet_comments(page, require_points=False):
    """
    OCR a whole typed page and return [{"text", "rect"}] - one per point.

    require_points=True is the conservative mode: the page is accepted only
    when it really contains numbered points ("Point No.1", "1.").
    """

    engine = _get_engine()

    if engine is None or cv2 is None:
        return []

    rgb, zoom = _render_page(page, dpi=TEXT_SHEET_DPI)

    rows, _used = _best_orientation_rows(engine, np.ascontiguousarray(rgb))

    if len(rows) < TEXT_SHEET_MIN_ROWS:
        return []

    if require_points:
        numbered = sum(1 for row in rows if _POINT_RE.match(row["text"]))

        if numbered < 2:
            return []

    results = []

    for point in _group_rows_into_points(rows):

        rect = pymupdf.Rect(
            point["x0"] / zoom,
            point["y0"] / zoom,
            point["x1"] / zoom,
            point["y1"] / zoom,
        )

        results.append({"text": point["text"], "rect": rect})

    return results


# ============================================================
# VECTOR CALLOUT COMMENTS
# ============================================================
#
# Many review tools (and "Print to PDF" of reviewed files) flatten
# comments into plain drawing shapes:
#
#     a FILLED rectangle  +  a COLOURED border on the same rectangle
#
# There is no annotation object and often no text layer either (the
# letters are converted to vector outlines), so neither the annotation pass
# nor the coloured-text pass can see them.

CALLOUT_MIN_SIZE_PT = 8  # smaller shapes are bullets / checkboxes
CALLOUT_MIN_LONG_SIDE_PT = 25
CALLOUT_MAX_PAGE_FRACTION = 0.5
CALLOUT_RENDER_DPI = 250


def _is_greyish(color, tolerance=0.12):
    """True for black / grey / white (no real hue)."""

    r, g, b = color[:3]
    return (max(r, g, b) - min(r, g, b)) < tolerance


def find_callout_rects(page):
    """Return rectangles that look like flattened comment boxes."""

    try:
        drawings = page.get_drawings()
    except Exception:
        return []

    page_area = abs(page.rect.width * page.rect.height)

    filled = []
    borders = []

    for item in drawings:

        rect = item.get("rect")

        if rect is None or rect.is_empty:
            continue

        fill = item.get("fill")
        stroke = item.get("color")

        if fill is not None and len(fill) >= 3:
            # Ignore white / near-white fills (blank cells).
            if min(fill[:3]) < 0.97:
                filled.append((rect, fill))

        if stroke is not None and len(stroke) >= 3 and not _is_greyish(stroke):
            borders.append(rect)

    callouts = []

    for rect, fill in filled:

        width, height = rect.width, rect.height

        if min(width, height) < CALLOUT_MIN_SIZE_PT:
            continue

        if max(width, height) < CALLOUT_MIN_LONG_SIDE_PT:
            continue

        if (width * height) > page_area * CALLOUT_MAX_PAGE_FRACTION:
            continue

        # Black / dark fills are ink, not callout boxes.
        if max(fill[:3]) < 0.5:
            continue

        has_border = any(
            abs(b.x0 - rect.x0) < 3
            and abs(b.y0 - rect.y0) < 3
            and abs(b.x1 - rect.x1) < 3
            and abs(b.y1 - rect.y1) < 3
            for b in borders
        )

        if not has_border:
            continue

        # Skip duplicates (same box drawn twice).
        if any(
            abs(rect.x0 - c.x0) < 2
            and abs(rect.y0 - c.y0) < 2
            and abs(rect.x1 - c.x1) < 2
            and abs(rect.y1 - c.y1) < 2
            for c in callouts
        ):
            continue

        callouts.append(rect)

    return callouts


def _callout_image(page, rect):
    inner = pymupdf.Rect(rect.x0 + 1.5, rect.y0 + 1.5, rect.x1 - 1.5, rect.y1 - 1.5)

    zoom = CALLOUT_RENDER_DPI / 72.0

    pix = page.get_pixmap(
        matrix=pymupdf.Matrix(zoom, zoom),
        clip=inner,
        colorspace=pymupdf.csRGB,
        alpha=False,
        annots=False,
    )

    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)

    return np.ascontiguousarray(img)


def extract_callout_comments(page, state=None):
    """
    Returns [{"text", "rect", "method"}] for flattened callout boxes, in
    logical reading order. `state` is accepted for compatibility and unused.
    """

    rects = find_callout_rects(page)

    if not rects:
        return []

    engine = _get_engine()

    results = []
    pending = []

    for rect in rects:

        # A real text layer inside the box is always better than OCR.
        try:
            direct = re.sub(
                r"\s+", " ", page.get_text("text", clip=rect, sort=True)
            ).strip()
        except Exception:
            direct = ""

        if len(direct) >= 2:
            results.append({"text": direct, "rect": rect, "method": "text-layer"})
            continue

        if engine is None or cv2 is None:
            continue

        pending.append((rect, _callout_image(page, rect)))

    if pending:

        readings = _read_regions(engine, [image for _rect, image in pending])

        for (rect, _image), (text, name) in zip(pending, readings):

            text = text.strip()

            if len(text) < 2 or not re.search(r"[A-Za-z0-9]", text):
                continue

            results.append({"text": text, "rect": rect, "method": name})

    if not results:
        return results

    # Reading order. If text had to be turned clockwise, the logical page
    # top is the physical left edge (and the reverse for counter-clockwise).
    cw = sum(1 for item in results if item["method"] == "cw")
    ccw = sum(1 for item in results if item["method"] == "ccw")

    if cw > ccw:
        results.sort(key=lambda i: (round(i["rect"].x0 / 12), -i["rect"].y0))
    elif ccw > cw:
        results.sort(key=lambda i: (-round(i["rect"].x0 / 12), i["rect"].y0))
    else:
        results.sort(key=lambda i: (round(i["rect"].y0 / 12), i["rect"].x0))

    return results


# ============================================================
# FLATTENED HIGHLIGHT / CLOUD REGIONS (pages that are not scans)
# ============================================================
#
# A reviewer's highlighter or "cloud" around a value is often flattened into
# the page as a pale coloured shape with the original text showing through.
# There is no annotation object, so the words under it are read from the
# rendered page, the same way a native Highlight annotation is read.


def _highlight_mask(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    h = hsv[..., 0].astype(np.int32)
    s = hsv[..., 1].astype(np.int32)
    v = hsv[..., 2].astype(np.int32)

    pastel = (
        (s >= HIGHLIGHT_MIN_SAT)
        & (s <= HIGHLIGHT_MAX_SAT)
        & (v >= HIGHLIGHT_MIN_VAL)
        & (v <= HIGHLIGHT_MAX_VAL)
    )

    wanted = np.zeros(h.shape, dtype=bool)

    for _name, low, high in HIGHLIGHT_COLOURS:
        wanted |= (h >= low) & (h <= high)

    mask = (pastel & wanted).astype(np.uint8) * 255

    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def extract_highlight_comments(page, exclude_rects=None):
    """
    Return [{"text", "rect"}] for the text under flattened highlight / cloud
    shapes. `exclude_rects` (page coordinates) are blanked out first - pass
    the callout boxes so they are not read twice.
    """

    engine = _get_engine()

    if engine is None or cv2 is None:
        return []

    rgb, zoom = _render_page(page, dpi=HIGHLIGHT_DPI)
    mask = _highlight_mask(rgb)

    for rect in exclude_rects or []:

        x0, y0, x1, y1 = _page_rect_to_pixels(
            pymupdf.Rect(rect.x0 - 4, rect.y0 - 4, rect.x1 + 4, rect.y1 + 4),
            zoom,
            page,
            mask.shape,
        )
        mask[y0:y1, x0:x1] = 0

    if cv2.countNonZero(mask) < HIGHLIGHT_MIN_INK_PIXELS:
        return []

    joined = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)),
    )

    count, _, stats, _ = cv2.connectedComponentsWithStats(joined, 8)

    minimum_side = HIGHLIGHT_MIN_SIZE_PT * zoom

    regions = []
    images = []

    for index in range(1, count):
        x, y, w, h, _ = stats[index]

        if min(w, h) < minimum_side:
            continue

        ink_pixels = cv2.countNonZero(mask[y : y + h, x : x + w])

        if ink_pixels < HIGHLIGHT_MIN_INK_PIXELS:
            continue

        if ink_pixels / float(w * h) < HIGHLIGHT_MIN_FILL_RATIO:
            continue

        # A highlighter is a pale fill with DARK text showing through.
        # Pale coloured letters on their own (a green "Formerly NPCC" in a
        # logo) have no dark text inside and are not a highlight.
        region_gray = cv2.cvtColor(rgb[y : y + h, x : x + w], cv2.COLOR_RGB2GRAY)

        if np.count_nonzero(region_gray < HIGHLIGHT_DARK_TEXT_LEVEL) < (
            HIGHLIGHT_MIN_DARK_TEXT_RATIO * w * h
        ):
            continue

        pad = 4
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(rgb.shape[1], x + w + pad), min(rgb.shape[0], y + h + pad)

        regions.append((x, y, x + w, y + h))
        images.append(np.ascontiguousarray(rgb[y0:y1, x0:x1]))

    results = []

    for rect_px, (text, name) in zip(regions, _read_regions(engine, images)):

        if not _has_content(text, 2):
            continue

        results.append(
            {
                "text": text,
                "rect": _pixel_rect_to_page(rect_px, zoom, page),
                "method": name,
            }
        )

    _sort_reading_order(results)

    return results


def _sort_reading_order(results):
    """Sort in place into the logical reading order of a rotated page."""

    cw = sum(1 for item in results if item.get("method") == "cw")
    ccw = sum(1 for item in results if item.get("method") == "ccw")

    if cw > ccw:
        results.sort(key=lambda i: (round(i["rect"].x0 / 12), -i["rect"].y0))
    elif ccw > cw:
        results.sort(key=lambda i: (-round(i["rect"].x0 / 12), i["rect"].y0))
    else:
        results.sort(key=lambda i: (round(i["rect"].y0 / 12), i["rect"].x0))


# ============================================================
# NATIVE INK (PEN) ANNOTATIONS
# ============================================================
#
# A pen stroke is stored as an Ink annotation: just coordinates, no text.
# The strokes are rendered, separated from the page underneath, and read.

INK_DPI = 250
INK_CLUSTER_GAP_PT = 14.0
INK_MIN_DIFFERENCE = 40
INK_UNREADABLE_TEXT = "[Pen markup - handwriting or mark could not be read as text]"


def _cluster_rects(rects, gap):
    """Merge rectangles that are within `gap` points of each other."""

    clusters = [pymupdf.Rect(r) for r in rects]

    merged = True

    while merged:

        merged = False
        result = []

        while clusters:

            current = clusters.pop()
            inflated = pymupdf.Rect(
                current.x0 - gap, current.y0 - gap, current.x1 + gap, current.y1 + gap
            )

            index = 0

            while index < len(clusters):

                if inflated.intersects(clusters[index]):
                    current = current | clusters.pop(index)
                    inflated = pymupdf.Rect(
                        current.x0 - gap,
                        current.y0 - gap,
                        current.x1 + gap,
                        current.y1 + gap,
                    )
                    merged = True
                else:
                    index += 1

            result.append(current)

        clusters = result

    return clusters


def extract_ink_comments(page):
    """Return [{"text", "rect", "author"}] for pen (Ink) annotations."""

    engine = _get_engine()

    if engine is None or cv2 is None:
        return []

    ink = []

    try:
        for annotation in page.annots() or []:
            if annotation.type[1] == "Ink":
                info = annotation.info or {}
                ink.append(
                    (
                        pymupdf.Rect(annotation.rect),
                        info.get("title") or info.get("author"),
                    )
                )
    except Exception:
        return []

    if not ink:
        return []

    clusters = _cluster_rects([rect for rect, _author in ink], INK_CLUSTER_GAP_PT)

    with_ink, zoom = _render_page(page, dpi=INK_DPI, annots=True)
    without_ink, _zoom = _render_page(page, dpi=INK_DPI, annots=False)

    regions = []
    images = []

    for cluster in clusters:

        padded = pymupdf.Rect(
            cluster.x0 - 6, cluster.y0 - 6, cluster.x1 + 6, cluster.y1 + 6
        )

        x0, y0, x1, y1 = _page_rect_to_pixels(padded, zoom, page, with_ink.shape)

        if x1 - x0 < 4 or y1 - y0 < 4:
            continue

        difference = cv2.absdiff(with_ink[y0:y1, x0:x1], without_ink[y0:y1, x0:x1]).max(
            axis=2
        )

        mask = (difference >= INK_MIN_DIFFERENCE).astype(np.uint8) * 255

        if cv2.countNonZero(mask) < MIN_BLOB_INK_PIXELS:
            continue

        regions.append(cluster)
        images.append(_ink_canvas(mask))

    results = []

    for cluster, image, (text, _name) in zip(
        regions, images, _read_regions(engine, images)
    ):

        if not _has_content(text, 3):
            text = _fallback_read(engine, image)

        if not _has_content(text, 3):
            text = INK_UNREADABLE_TEXT

        author = None

        for rect, name in ink:
            if name and cluster.intersects(rect):
                author = name
                break

        results.append({"text": text, "rect": cluster, "author": author})

    return results