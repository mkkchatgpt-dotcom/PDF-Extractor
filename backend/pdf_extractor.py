from datetime import datetime
import re
import math

import pymupdf


# ============================================================
# CONFIGURATION
# ============================================================

# Native annotation types that we want.
ALLOWED_ANNOTATION_TYPES = {
    "FreeText",
    "Highlight",
    "Stamp",
}

# Used for detecting coloured / flattened review text.
MIN_COLOR_SATURATION = 0.25
MIN_COLOR_BRIGHTNESS = 0.18

# Used when grouping coloured page text.
TEXT_LINE_VERTICAL_TOLERANCE = 5.0
TEXT_GROUP_DISTANCE = 25.0


# ============================================================
# PDF DATE PARSER
# ============================================================

def parse_pdf_datetime(value):
    """
    Convert PDF date:

        D:YYYYMMDDHHMMSS

    Into:

        DD-MM-YYYY
        HH:MM:SS
    """

    if not value:
        return None, None

    try:
        value = str(value)

        if value.startswith("D:"):
            value = value[2:]

        match = re.match(
            r"(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})",
            value
        )

        if not match:
            return None, None

        year, month, day, hour, minute, second = match.groups()

        dt = datetime(
            int(year),
            int(month),
            int(day),
            int(hour),
            int(minute),
            int(second)
        )

        return (
            dt.strftime("%d-%m-%Y"),
            dt.strftime("%H:%M:%S")
        )

    except Exception:
        return None, None


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(value):
    """
    Normalize text for duplicate comparison.
    """

    if not value:
        return ""

    value = str(value)

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    return value.strip().lower()


# ============================================================
# COLOR HELPERS
# ============================================================

def int_to_rgb(color_value):
    """
    Convert PyMuPDF integer colour to RGB 0-1 values.
    """

    try:
        color_value = int(color_value)

        red = (color_value >> 16) & 255
        green = (color_value >> 8) & 255
        blue = color_value & 255

        return (
            red / 255.0,
            green / 255.0,
            blue / 255.0
        )

    except Exception:
        return 0.0, 0.0, 0.0


def get_color_properties(rgb):
    """
    Return brightness and saturation.
    """

    r, g, b = rgb

    maximum = max(r, g, b)
    minimum = min(r, g, b)

    brightness = maximum

    if maximum == 0:
        saturation = 0
    else:
        saturation = (
            maximum - minimum
        ) / maximum

    return brightness, saturation


def is_review_color(color_value):
    """
    Detect red / orange / yellow coloured review text.

    Black and grey drawing text is ignored.
    """

    try:
        r, g, b = int_to_rgb(
            color_value
        )

        brightness, saturation = (
            get_color_properties(
                (r, g, b)
            )
        )

        if brightness < MIN_COLOR_BRIGHTNESS:
            return False

        if saturation < MIN_COLOR_SATURATION:
            return False

        # Red
        red_like = (
            r >= 0.45
            and r > g * 1.15
            and r > b * 1.20
        )

        # Orange
        orange_like = (
            r >= 0.55
            and g >= 0.15
            and g <= r
            and b <= 0.55
            and r > b * 1.20
        )

        # Yellow / gold
        yellow_like = (
            r >= 0.55
            and g >= 0.45
            and b <= 0.45
            and r > b * 1.30
            and g > b * 1.20
        )

        return (
            red_like
            or orange_like
            or yellow_like
        )

    except Exception:
        return False


# ============================================================
# RECTANGLE HELPERS
# ============================================================

def bbox_to_rect(bbox):

    try:
        if not bbox or len(bbox) < 4:
            return None

        return pymupdf.Rect(
            float(bbox[0]),
            float(bbox[1]),
            float(bbox[2]),
            float(bbox[3])
        )

    except Exception:
        return None


def rect_center(rect):

    if rect is None:
        return None

    return (
        (rect.x0 + rect.x1) / 2,
        (rect.y0 + rect.y1) / 2
    )


def rect_distance(rect1, rect2):

    if rect1 is None or rect2 is None:
        return float("inf")

    center1 = rect_center(rect1)
    center2 = rect_center(rect2)

    if center1 is None or center2 is None:
        return float("inf")

    return math.sqrt(
        (center1[0] - center2[0]) ** 2
        +
        (center1[1] - center2[1]) ** 2
    )


def rectangles_overlap(rect1, rect2):

    if rect1 is None or rect2 is None:
        return False

    try:

        intersection = rect1 & rect2

        return (
            intersection.width > 0
            and intersection.height > 0
        )

    except Exception:
        return False


# ============================================================
# INTERSECTION AREA
# ============================================================

def rectangle_intersection_ratio(
    word_rect,
    highlight_rect
):
    """
    Calculate how much of a word rectangle intersects
    with a highlight rectangle.

    This is much safer than using only get_textbox().
    """

    try:

        intersection = (
            word_rect & highlight_rect
        )

        if (
            intersection.width <= 0
            or intersection.height <= 0
        ):
            return 0.0

        word_area = (
            word_rect.width
            *
            word_rect.height
        )

        if word_area <= 0:
            return 0.0

        intersection_area = (
            intersection.width
            *
            intersection.height
        )

        return (
            intersection_area
            /
            word_area
        )

    except Exception:
        return 0.0


# ============================================================
# DUPLICATE CHECK
# ============================================================

def is_duplicate_record(
    existing_records,
    page_number,
    text,
    rect=None
):

    normalized_new_text = (
        normalize_text(text)
    )

    if not normalized_new_text:
        return False

    for existing in existing_records:

        if (
            existing.get("page_number")
            != page_number
        ):
            continue

        existing_text = normalize_text(
            existing.get("text")
        )

        if (
            existing_text
            != normalized_new_text
        ):
            continue

        existing_rect = (
            existing.get("_rect")
        )

        # If position isn't available,
        # same page + same text = duplicate.
        if (
            existing_rect is None
            or rect is None
        ):
            return True

        if rectangles_overlap(
            existing_rect,
            rect
        ):
            return True

        distance = rect_distance(
            existing_rect,
            rect
        )

        if distance <= 15:
            return True

    return False


# ============================================================
# HIGHLIGHT RECTANGLES
# ============================================================

def get_highlight_rectangles(annotation):
    """
    Convert Highlight vertices into rectangles.

    PDF highlights normally contain four points per quad.
    """

    rectangles = []

    try:

        vertices = (
            annotation.vertices
        )

        if not vertices:
            return rectangles

        for i in range(
            0,
            len(vertices),
            4
        ):

            points = vertices[
                i:i + 4
            ]

            if len(points) < 4:
                continue

            xs = [
                point.x
                for point in points
            ]

            ys = [
                point.y
                for point in points
            ]

            rect = pymupdf.Rect(
                min(xs),
                min(ys),
                max(xs),
                max(ys)
            )

            rectangles.append(
                rect
            )

    except Exception as exc:

        print(
            "Could not read highlight "
            f"vertices: {exc}"
        )

    return rectangles


# ============================================================
# EXTRACT WORDS UNDER HIGHLIGHT
# ============================================================

# def extract_highlight_text(
#     page,
#     annotation
# ):

def extract_highlight_text(page, annotation):
    """
    Extract the actual text underneath a Highlight annotation.

    Supports annotation.vertices returned as:
      - pymupdf.Point objects
      - (x, y) tuples / lists
    """

    try:
        vertices = annotation.vertices

        if not vertices:
            print(
                f"Highlight XREF {annotation.xref}: "
                "No vertices found"
            )
            return ""

        # ====================================================
        # HELPER: GET X / Y FROM VERTEX
        # ====================================================

        def get_xy(point):
            """
            PyMuPDF versions may return vertices either as:

                Point(x, y)

            or:

                (x, y)
            """

            if hasattr(point, "x") and hasattr(point, "y"):
                return float(point.x), float(point.y)

            if (
                isinstance(point, (tuple, list))
                and len(point) >= 2
            ):
                return float(point[0]), float(point[1])

            raise ValueError(
                f"Unsupported vertex format: {point}"
            )

        # ====================================================
        # CREATE HIGHLIGHT RECTANGLES
        # ====================================================

        highlight_rects = []

        for i in range(0, len(vertices), 4):

            points = vertices[i:i + 4]

            if len(points) < 4:
                continue

            xy_points = [
                get_xy(point)
                for point in points
            ]

            xs = [
                point[0]
                for point in xy_points
            ]

            ys = [
                point[1]
                for point in xy_points
            ]

            rect = pymupdf.Rect(
                min(xs),
                min(ys),
                max(xs),
                max(ys)
            )

            # Slight expansion helps with text whose bounding
            # box extends slightly outside the highlight.
            expanded_rect = pymupdf.Rect(
                rect.x0 - 2,
                rect.y0 - 2,
                rect.x1 + 2,
                rect.y1 + 2
            )

            highlight_rects.append(
                expanded_rect
            )

            print(
                f"Highlight XREF {annotation.xref} "
                f"| Rect: {expanded_rect}"
            )

        if not highlight_rects:

            print(
                f"Highlight XREF {annotation.xref}: "
                "No valid highlight rectangles"
            )

            return ""

        # ====================================================
        # GET PAGE WORDS
        # ====================================================

        words = page.get_text(
            "words",
            sort=True
        )

        matched_words = []

        # ====================================================
        # MATCH WORDS TO HIGHLIGHT
        # ====================================================

        for word in words:

            if len(word) < 5:
                continue

            word_text = str(
                word[4]
            ).strip()

            if not word_text:
                continue

            word_rect = pymupdf.Rect(
                float(word[0]),
                float(word[1]),
                float(word[2]),
                float(word[3])
            )

            # -----------------------------------------------
            # Calculate word area
            # -----------------------------------------------

            word_area = (
                word_rect.width
                *
                word_rect.height
            )

            if word_area <= 0:
                continue

            matched = False

            for highlight_rect in highlight_rects:

                intersection = (
                    word_rect
                    &
                    highlight_rect
                )

                if (
                    intersection.width <= 0
                    or
                    intersection.height <= 0
                ):
                    continue

                intersection_area = (
                    intersection.width
                    *
                    intersection.height
                )

                overlap_ratio = (
                    intersection_area
                    /
                    word_area
                )

                # -------------------------------------------
                # Require 35% of the word to overlap.
                #
                # Engineering drawing highlights are often
                # slightly smaller than the text bounding box.
                # -------------------------------------------

                if overlap_ratio >= 0.35:

                    matched = True
                    break

            if not matched:
                continue

            matched_words.append({

                "text":
                    word_text,

                "x0":
                    float(word[0]),

                "y0":
                    float(word[1]),

                "x1":
                    float(word[2]),

                "y1":
                    float(word[3]),

                "block":
                    word[5]
                    if len(word) > 5
                    else 0,

                "line":
                    word[6]
                    if len(word) > 6
                    else 0,

                "word":
                    word[7]
                    if len(word) > 7
                    else 0
            })

        # ====================================================
        # NO MATCHES -> CLIP FALLBACK
        # ====================================================

        if not matched_words:

            fallback_parts = []

            for rect in highlight_rects:

                try:

                    clipped_text = page.get_text(
                        "text",
                        clip=rect,
                        sort=True
                    )

                    clipped_text = re.sub(
                        r"\s+",
                        " ",
                        clipped_text
                    ).strip()

                    if clipped_text:
                        fallback_parts.append(
                            clipped_text
                        )

                except Exception as exc:

                    print(
                        f"Highlight XREF "
                        f"{annotation.xref} "
                        f"fallback failed: {exc}"
                    )

            result = " ".join(
                fallback_parts
            ).strip()

            print(
                f"HIGHLIGHT RESULT -> "
                f"XREF: {annotation.xref} | "
                f"Text: [{result}]"
            )

            return result

        # ====================================================
        # SORT MATCHED WORDS
        # ====================================================

        matched_words.sort(
            key=lambda item: (
                item["block"],
                item["line"],
                item["word"],
                item["y0"],
                item["x0"]
            )
        )

        # ====================================================
        # REMOVE DUPLICATE WORDS
        # ====================================================

        unique_words = []

        seen = set()

        for item in matched_words:

            key = (
                round(item["x0"], 2),
                round(item["y0"], 2),
                round(item["x1"], 2),
                round(item["y1"], 2),
                item["text"]
            )

            if key in seen:
                continue

            seen.add(key)

            unique_words.append(
                item["text"]
            )

        # ====================================================
        # FINAL TEXT
        # ====================================================

        result = " ".join(
            unique_words
        )

        result = re.sub(
            r"\s+",
            " ",
            result
        ).strip()

        print(
            f"HIGHLIGHT RESULT -> "
            f"XREF: {annotation.xref} | "
            f"Text: [{result}]"
        )

        return result

    except Exception as exc:

        print(
            f"Highlight extraction failed "
            f"for XREF {annotation.xref}: "
            f"{exc}"
        )

        return ""

# ============================================================
# GET COLOURED PAGE TEXT
# ============================================================

def get_colored_text_spans(page):
    """
    Find red/orange/yellow text that exists as normal
    PDF page content.
    """

    spans = []

    try:

        page_dict = page.get_text(
            "dict"
        )

        blocks = page_dict.get(
            "blocks",
            []
        )


        for block in blocks:

            if block.get("type") != 0:
                continue


            for line in block.get(
                "lines",
                []
            ):

                for span in line.get(
                    "spans",
                    []
                ):

                    text = (
                        span.get(
                            "text",
                            ""
                        )
                        or ""
                    ).strip()


                    if not text:
                        continue


                    color = span.get(
                        "color",
                        0
                    )


                    if not is_review_color(
                        color
                    ):
                        continue


                    rect = bbox_to_rect(
                        span.get(
                            "bbox"
                        )
                    )


                    if rect is None:
                        continue


                    spans.append({

                        "text":
                            text,

                        "rect":
                            rect,

                        "color":
                            color,

                        "size":
                            span.get(
                                "size"
                            ),

                        "font":
                            span.get(
                                "font"
                            )
                    })


    except Exception as exc:

        print(
            "Could not inspect coloured "
            f"page text: {exc}"
        )


    return spans


# ============================================================
# GROUP COLOURED TEXT
# ============================================================

def group_colored_spans(spans):
    """
    Combine neighbouring coloured spans into comments.
    """

    if not spans:
        return []


    spans = sorted(
        spans,
        key=lambda item: (
            round(
                item["rect"].y0,
                1
            ),
            item["rect"].x0
        )
    )


    groups = []


    for span in spans:

        added = False


        for group in groups:

            group_rect = (
                group["rect"]
            )

            span_rect = (
                span["rect"]
            )


            same_line = (
                abs(
                    span_rect.y0
                    -
                    group_rect.y0
                )
                <=
                TEXT_LINE_VERTICAL_TOLERANCE
            )


            next_line = (
                abs(
                    span_rect.y0
                    -
                    group_rect.y1
                )
                <=
                TEXT_GROUP_DISTANCE
            )


            horizontal_relation = not (
                span_rect.x1
                <
                group_rect.x0 - 20
                or
                span_rect.x0
                >
                group_rect.x1 + 100
            )


            if (
                horizontal_relation
                and
                (
                    same_line
                    or
                    next_line
                )
            ):

                group[
                    "spans"
                ].append(
                    span
                )

                group["rect"] = (
                    group_rect
                    |
                    span_rect
                )

                added = True

                break


        if not added:

            groups.append({

                "spans": [
                    span
                ],

                "rect":
                    pymupdf.Rect(
                        span["rect"]
                    )
            })


    results = []


    for group in groups:

        group_spans = sorted(
            group["spans"],
            key=lambda item: (
                round(
                    item["rect"].y0,
                    1
                ),
                item["rect"].x0
            )
        )


        text_parts = []


        for span in group_spans:

            value = (
                span["text"]
                .strip()
            )

            if value:
                text_parts.append(
                    value
                )


        combined_text = " ".join(
            text_parts
        )


        combined_text = re.sub(
            r"\s+",
            " ",
            combined_text
        ).strip()


        if combined_text:

            results.append({

                "text":
                    combined_text,

                "rect":
                    group["rect"]
            })


    return results


# ============================================================
# EXTRACT NATIVE ANNOTATIONS
# ============================================================

def extract_native_annotations(
    page,
    page_number,
    records,
    record_id
):

    annotation_xrefs = []


    try:

        for annotation in (
            page.annots() or []
        ):

            annotation_xrefs.append(
                annotation.xref
            )


    except Exception as exc:

        print(
            f"Could not enumerate annotations "
            f"on page {page_number}: "
            f"{exc}"
        )

        return record_id


    print(
        f"Page {page_number} "
        f"| Total annotation objects found: "
        f"{len(annotation_xrefs)}"
    )


    for xref in annotation_xrefs:

        try:

            annotation = page.load_annot(
                xref
            )


            if annotation is None:
                continue


            info = (
                annotation.info
                or {}
            )


            # =================================================
            # ANNOTATION TYPE
            # =================================================

            annotation_type = (
                annotation.type
            )


            if (
                isinstance(
                    annotation_type,
                    (tuple, list)
                )
                and
                len(annotation_type) > 1
            ):

                annotation_type_name = (
                    annotation_type[1]
                )

            else:

                annotation_type_name = (
                    str(annotation_type)
                )


            print(
                f"Page {page_number} "
                f"| XREF: {xref} "
                f"| Type: "
                f"{annotation_type_name}"
            )


            if (
                annotation_type_name
                not in
                ALLOWED_ANNOTATION_TYPES
            ):

                continue


            # =================================================
            # AUTHOR
            # =================================================

            author = (
                info.get("title")
                or
                info.get("author")
                or
                None
            )


            # =================================================
            # NORMAL COMMENT CONTENT
            # =================================================

            content = (
                info.get("content")
                or ""
            ).strip()


            # =================================================
            # FREETEXT
            # =================================================

            if (
                annotation_type_name
                == "FreeText"
            ):

                # If content is empty, read visible text.
                if not content:

                    try:

                        annotation_text = (
                            annotation.get_text(
                                "text"
                            )
                        )


                        if annotation_text:

                            content = (
                                annotation_text
                                .strip()
                            )


                    except Exception as exc:

                        print(
                            "Could not extract "
                            "FreeText text "
                            f"xref {xref}: "
                            f"{exc}"
                        )


            # =================================================
            # HIGHLIGHT
            # =================================================
            #
            # IMPORTANT:
            #
            # For Highlight we want the text underneath the
            # highlight, not only info["content"].
            #
            # Therefore always attempt geometric extraction.
            # =================================================

            elif (
                annotation_type_name
                == "Highlight"
            ):

                highlighted_text = (
                    extract_highlight_text(
                        page,
                        annotation
                    )
                )


                if highlighted_text:

                    content = (
                        highlighted_text
                    )


            # =================================================
            # STAMP
            # =================================================

            elif (
                annotation_type_name
                == "Stamp"
            ):

                if not content:

                    stamp_name = (
                        info.get("name")
                    )

                    if stamp_name:

                        content = (
                            str(
                                stamp_name
                            )
                            .strip()
                        )


            # =================================================
            # ANNOTATION RECT
            # =================================================

            try:

                annotation_rect = (
                    pymupdf.Rect(
                        annotation.rect
                    )
                )

            except Exception:

                annotation_rect = None


            # =================================================
            # DATES
            # =================================================

            (
                created_date,
                created_time

            ) = parse_pdf_datetime(

                info.get(
                    "creationDate"
                )
            )


            (
                modified_date,
                modified_time

            ) = parse_pdf_datetime(

                info.get(
                    "modDate"
                )
            )


            # =================================================
            # DUPLICATE CHECK
            # =================================================

            if is_duplicate_record(
                records,
                page_number,
                content,
                annotation_rect
            ):

                print(
                    "Duplicate native annotation "
                    f"ignored: "
                    f"{content[:100]}"
                )

                continue


            # =================================================
            # ADD RECORD
            # =================================================

            records.append({

                "record_id":
                    record_id,

                "page_number":
                    page_number,

                "object_type":
                    "Annotation",

                "annotation_type":
                    annotation_type_name,

                "author":
                    author,

                "created_date":
                    created_date,

                "created_time":
                    created_time,

                "modified_date":
                    modified_date,

                "modified_time":
                    modified_time,

                "text":
                    content,

                "_rect":
                    annotation_rect
            })


            print(
                f"EXTRACTED NATIVE -> "
                f"Page: {page_number} | "
                f"Type: "
                f"{annotation_type_name} | "
                f"Text: "
                f"{content[:150]}"
            )


            record_id += 1


        except Exception as exc:

            print(
                f"Error processing annotation "
                f"xref {xref} "
                f"on page {page_number}: "
                f"{exc}"
            )


    return record_id


# ============================================================
# EXTRACT COLOURED / FLATTENED COMMENTS
# ============================================================

def extract_colored_page_comments(
    page,
    page_number,
    records,
    record_id
):
    """
    Extract genuinely flattened coloured review comments.

    Priority:
        1. Native FreeText / Highlight / Stamp are already extracted
           in PASS 1.
        2. PASS 2 should capture only coloured page-content comments
           that are NOT already represented by native annotations.
        3. Prevent a coloured text group from being exported when it
           contains or overlaps an existing native FreeText comment.
    """

    # ============================================================
    # GET COLOURED TEXT SPANS
    # ============================================================

    colored_spans = get_colored_text_spans(
        page
    )

    print(
        f"Page {page_number} "
        f"| Coloured text spans found: "
        f"{len(colored_spans)}"
    )

    # ============================================================
    # GROUP COLOURED TEXT
    # ============================================================

    grouped_comments = group_colored_spans(
        colored_spans
    )

    print(
        f"Page {page_number} "
        f"| Coloured comment groups: "
        f"{len(grouped_comments)}"
    )

    # ============================================================
    # PROCESS EACH COLOURED GROUP
    # ============================================================

    for comment in grouped_comments:

        content = (
            comment.get(
                "text",
                ""
            )
            .strip()
        )

        rect = comment.get(
            "rect"
        )

        if not content:
            continue

        if len(content) < 2:
            continue

        normalized_colored = normalize_text(
            content
        )

        if not normalized_colored:
            continue

        # ========================================================
        # IMPORTANT:
        # CHECK AGAINST NATIVE FREETEXT COMMENTS
        # ========================================================

        belongs_to_native_freetext = False

        for existing in records:

            # Same page only.
            if (
                existing.get("page_number")
                != page_number
            ):
                continue

            # We specifically want to protect native FreeText.
            if (
                existing.get("object_type")
                != "Annotation"
            ):
                continue

            if (
                existing.get("annotation_type")
                != "FreeText"
            ):
                continue

            existing_text = normalize_text(
                existing.get(
                    "text",
                    ""
                )
            )

            if not existing_text:
                continue

            # ----------------------------------------------------
            # CASE 1:
            # Exact same text
            # ----------------------------------------------------

            if (
                normalized_colored
                == existing_text
            ):

                belongs_to_native_freetext = True

                print(
                    "Coloured text ignored - "
                    "exact native FreeText match: "
                    f"{content[:100]}"
                )

                break

            # ----------------------------------------------------
            # CASE 2:
            # Coloured group contains an entire native comment.
            #
            # Example:
            #
            # coloured group:
            # "Provide sunshade... Typical comment..."
            #
            # native:
            # "Provide sunshade..."
            #
            # This means the grouping joined native comments.
            # Do NOT export the combined flattened record.
            # ----------------------------------------------------

            if (
                len(existing_text) >= 8
                and
                existing_text
                in normalized_colored
            ):

                belongs_to_native_freetext = True

                print(
                    "Coloured group ignored - "
                    "contains native FreeText: "
                    f"{content[:100]}"
                )

                break

            # ----------------------------------------------------
            # CASE 3:
            # Native FreeText contains this coloured text.
            # ----------------------------------------------------

            if (
                len(normalized_colored) >= 8
                and
                normalized_colored
                in existing_text
            ):

                belongs_to_native_freetext = True

                print(
                    "Coloured text ignored - "
                    "contained inside native FreeText: "
                    f"{content[:100]}"
                )

                break

            # ----------------------------------------------------
            # CASE 4:
            # POSITIONAL CHECK
            #
            # Native FreeText can sometimes differ slightly from
            # page-content text because of spaces / line wrapping.
            # If their rectangles overlap strongly, treat the
            # coloured text as belonging to native FreeText.
            # ----------------------------------------------------

            existing_rect = existing.get(
                "_rect"
            )

            if (
                rect is not None
                and
                existing_rect is not None
            ):

                try:

                    intersection = (
                        rect
                        &
                        existing_rect
                    )

                    if (
                        intersection.width > 0
                        and
                        intersection.height > 0
                    ):

                        colored_area = (
                            rect.width
                            *
                            rect.height
                        )

                        intersection_area = (
                            intersection.width
                            *
                            intersection.height
                        )

                        if colored_area > 0:

                            overlap_ratio = (
                                intersection_area
                                /
                                colored_area
                            )

                            # 50% of coloured group falls inside
                            # the native FreeText annotation.
                            if overlap_ratio >= 0.50:

                                belongs_to_native_freetext = True

                                print(
                                    "Coloured text ignored - "
                                    "overlaps native FreeText: "
                                    f"{content[:100]}"
                                )

                                break

                except Exception:
                    pass

        # ========================================================
        # DON'T EXPORT NATIVE FREETEXT AGAIN
        # ========================================================

        if belongs_to_native_freetext:
            continue

        # ========================================================
        # NORMAL DUPLICATE CHECK
        # ========================================================

        if is_duplicate_record(
            records,
            page_number,
            content,
            rect
        ):

            print(
                "Duplicate coloured page "
                f"text ignored: "
                f"{content[:100]}"
            )

            continue

        # ========================================================
        # THIS IS GENUINE FLATTENED PAGE CONTENT
        # ========================================================

        records.append({

            "record_id":
                record_id,

            "page_number":
                page_number,

            "object_type":
                "Page Content",

            "annotation_type":
                "FlattenedText",

            "author":
                None,

            "created_date":
                None,

            "created_time":
                None,

            "modified_date":
                None,

            "modified_time":
                None,

            "text":
                content,

            "_rect":
                rect
        })

        print(
            f"EXTRACTED FLATTENED -> "
            f"Page: {page_number} | "
            f"Text: "
            f"{content[:150]}"
        )

        record_id += 1

    return record_id

# ============================================================
# FINAL DUPLICATE REMOVAL
# ============================================================

def remove_final_duplicates(records):

    unique_records = []


    for record in records:

        page_number = (
            record.get(
                "page_number"
            )
        )

        text = (
            record.get(
                "text",
                ""
            )
        )

        rect = (
            record.get(
                "_rect"
            )
        )


        if is_duplicate_record(
            unique_records,
            page_number,
            text,
            rect
        ):

            print(
                "FINAL DUPLICATE REMOVED -> "
                f"Page: {page_number} | "
                f"Text: "
                f"{text[:100]}"
            )

            continue


        unique_records.append(
            record
        )


    # Re-number Record IDs.
    for index, record in enumerate(
        unique_records,
        start=1
    ):

        record[
            "record_id"
        ] = index


    return unique_records


# ============================================================
# REMOVE INTERNAL VALUES
# ============================================================

def clean_internal_fields(records):

    cleaned_records = []


    for record in records:

        clean_record = dict(
            record
        )


        clean_record.pop(
            "_rect",
            None
        )


        cleaned_records.append(
            clean_record
        )


    return cleaned_records


# ============================================================
# MAIN PDF EXTRACTION
# ============================================================

def extract_pdf(pdf_path):
    """
    Complete PDF review extraction.

    PASS 1
    ------
    Native annotations:

        FreeText
        Highlight
        Stamp

    Highlight:
        extracts the actual words underneath the highlight.

    PASS 2
    ------
    Coloured page content:

        red
        orange
        yellow/gold

    Used for flattened review comments.

    PASS 3
    ------
    Duplicate removal.
    """


    records = []


    with pymupdf.open(
        pdf_path
    ) as document:


        summary = {

            "pages":
                len(document),

            "annotations":
                0
        }


        record_id = 1


        # ====================================================
        # PAGE LOOP
        # ====================================================

        for page_number in range(
            1,
            len(document) + 1
        ):


            page = document.load_page(
                page_number - 1
            )


            page_rotation = (
                page.rotation
            )


            print()
            print(
                "=" * 70
            )

            print(
                f"Processing Page "
                f"{page_number} "
                f"| Rotation: "
                f"{page_rotation} degrees"
            )

            print(
                "=" * 70
            )


            # =================================================
            # PASS 1
            # NATIVE ANNOTATIONS
            # =================================================

            print()
            print(
                "PASS 1: "
                "Native PDF annotations"
            )


            record_id = (
                extract_native_annotations(
                    page,
                    page_number,
                    records,
                    record_id
                )
            )


            # =================================================
            # PASS 2
            # FLATTENED COLOURED TEXT
            # =================================================

            print()
            print(
                "PASS 2: "
                "Coloured page-content comments"
            )


            record_id = (
                extract_colored_page_comments(
                    page,
                    page_number,
                    records,
                    record_id
                )
            )


        # ====================================================
        # PASS 3
        # DUPLICATES
        # ====================================================

        print()
        print(
            "=" * 70
        )

        print(
            "PASS 3: "
            "Removing duplicate records"
        )

        print(
            "=" * 70
        )


        records = (
            remove_final_duplicates(
                records
            )
        )


        summary[
            "annotations"
        ] = len(
            records
        )


        # Remove _rect before sending data to frontend/Excel.
        records = (
            clean_internal_fields(
                records
            )
        )


    # ========================================================
    # FINAL DEBUG
    # ========================================================

    print()
    print(
        "=" * 70
    )

    print(
        "EXTRACTION COMPLETE"
    )

    print(
        f"Pages: "
        f"{summary['pages']}"
    )

    print(
        f"Records: "
        f"{summary['annotations']}"
    )

    print(
        "=" * 70
    )


    return {

        "summary":
            summary,

        "records":
            records
    }