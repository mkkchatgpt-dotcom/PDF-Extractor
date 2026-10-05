from datetime import datetime
import re
import math

import pymupdf


# ============================================================
# CONFIGURATION
# ============================================================

# Native PDF annotation types that we want to extract.
ALLOWED_ANNOTATION_TYPES = {
    "FreeText",
    "Highlight",
    "Stamp",
}


# Minimum amount of colour required before ordinary page text
# is considered potentially review/comment text.
#
# This is intentionally conservative so that normal black/grey
# engineering drawing text is ignored.
MIN_COLOR_SATURATION = 0.25


# Very light colours are normally background / anti-aliasing.
MIN_COLOR_BRIGHTNESS = 0.18


# Maximum vertical gap used when combining neighbouring coloured
# text spans into one review comment.
TEXT_LINE_VERTICAL_TOLERANCE = 5.0


# Maximum distance between coloured text spans before they are
# considered separate comments.
TEXT_GROUP_DISTANCE = 25.0


# ============================================================
# PDF DATE PARSER
# ============================================================

def parse_pdf_datetime(value):
    """
    Convert PDF date format:

        D:YYYYMMDDHHMMSS

    Example:

        D:20260929143025

    Into:

        Date: 29-09-2026
        Time: 14:30:25
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

    Example:

        " Please   Update Drawing "
                    ↓
        "please update drawing"
    """

    if not value:
        return ""

    value = str(value)

    # Replace multiple whitespace characters with one space.
    value = re.sub(r"\s+", " ", value)

    return value.strip().lower()


# ============================================================
# COLOR CONVERSION
# ============================================================

def int_to_rgb(color_value):
    """
    Convert PyMuPDF integer colour into RGB values from 0.0 - 1.0.

    Example:

        16711680 -> red
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


# ============================================================
# RGB -> HSV STYLE VALUES
# ============================================================

def get_color_properties(rgb):
    """
    Calculate simple brightness and saturation.

    We use saturation rather than checking for one exact orange
    colour because different PDF applications can generate
    slightly different orange / red / yellow RGB values.
    """

    r, g, b = rgb

    maximum = max(r, g, b)
    minimum = min(r, g, b)

    brightness = maximum

    if maximum == 0:
        saturation = 0
    else:
        saturation = (maximum - minimum) / maximum

    return brightness, saturation


# ============================================================
# DETECT REVIEW COLOUR
# ============================================================

def is_review_color(color_value):
    """
    Determine whether page text looks like coloured review markup.

    Designed mainly for:

        orange
        red
        reddish-orange
        yellow/orange

    while rejecting:

        black
        dark grey
        normal drawing text

    IMPORTANT:
    This intentionally does not depend on one exact RGB value.
    """

    try:
        r, g, b = int_to_rgb(color_value)

        brightness, saturation = get_color_properties(
            (r, g, b)
        )

        # ----------------------------------------------------
        # Reject black / grey text
        # ----------------------------------------------------

        if brightness < MIN_COLOR_BRIGHTNESS:
            return False

        if saturation < MIN_COLOR_SATURATION:
            return False

        # ----------------------------------------------------
        # RED
        # ----------------------------------------------------
        #
        # Red component clearly stronger than green and blue.
        #

        red_like = (
            r >= 0.45
            and r > g * 1.15
            and r > b * 1.20
        )

        # ----------------------------------------------------
        # ORANGE
        # ----------------------------------------------------

        orange_like = (
            r >= 0.55
            and g >= 0.15
            and g <= r
            and b <= 0.55
            and r > b * 1.20
        )

        # ----------------------------------------------------
        # YELLOW / GOLD
        # ----------------------------------------------------

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
    """
    Safely convert a bbox into a PyMuPDF Rect.
    """

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
    """
    Return center point of a rectangle.
    """

    if rect is None:
        return None

    return (
        (rect.x0 + rect.x1) / 2,
        (rect.y0 + rect.y1) / 2
    )


def rect_distance(rect1, rect2):
    """
    Calculate approximate distance between two rectangle centres.
    """

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
    """
    Check whether two rectangles overlap.
    """

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
# DUPLICATE CHECK
# ============================================================

def is_duplicate_record(
    existing_records,
    page_number,
    text,
    rect=None
):
    """
    Determine whether a newly found comment already exists.

    Primary comparison:

        page
        +
        normalized text

    If position information exists, it is also considered.

    This prevents:

        FreeText annotation
              +
        same visible page text

    from becoming two Excel records.
    """

    normalized_new_text = normalize_text(text)

    if not normalized_new_text:
        return True

    for existing in existing_records:

        if existing.get("page_number") != page_number:
            continue

        existing_text = normalize_text(
            existing.get("text")
        )

        if existing_text != normalized_new_text:
            continue

        # ----------------------------------------------------
        # Same page + exactly same normalized text
        # ----------------------------------------------------

        existing_rect = existing.get("_rect")

        # If position isn't available, text equality on the same
        # page is the safest fallback.
        if existing_rect is None or rect is None:
            return True

        # Exact/overlapping physical location.
        if rectangles_overlap(
            existing_rect,
            rect
        ):
            return True

        # Very close location.
        distance = rect_distance(
            existing_rect,
            rect
        )

        if distance <= 15:
            return True

        # Same text but genuinely different location:
        # keep it because the same comment could legitimately
        # appear twice on a drawing.

    return False


# ============================================================
# HIGHLIGHT TEXT EXTRACTION
# ============================================================

def extract_highlight_text(
    page,
    annotation
):
    """
    Extract the actual text underneath a Highlight annotation.
    """

    highlighted_text = []

    try:
        vertices = annotation.vertices

        if not vertices:
            return ""

        # Highlight vertices normally come in groups of four.
        for i in range(
            0,
            len(vertices),
            4
        ):

            quad_points = vertices[
                i:i + 4
            ]

            if len(quad_points) < 4:
                continue

            xs = [
                point.x
                for point in quad_points
            ]

            ys = [
                point.y
                for point in quad_points
            ]

            rect = pymupdf.Rect(
                min(xs),
                min(ys),
                max(xs),
                max(ys)
            )

            text = page.get_textbox(
                rect
            )

            if text:
                cleaned = text.strip()

                if cleaned:
                    highlighted_text.append(
                        cleaned
                    )

        return " ".join(
            highlighted_text
        ).strip()

    except Exception as exc:

        print(
            "Could not extract highlighted text: "
            f"{exc}"
        )

        return ""


# ============================================================
# EXTRACT NATIVE PDF ANNOTATIONS
# ============================================================

def extract_native_annotations(
    page,
    page_number,
    records,
    record_id
):
    """
    Extract:

        FreeText
        Highlight
        Stamp

    from real PDF annotation objects.
    """

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

            # ------------------------------------------------
            # LOAD ANNOTATION
            # ------------------------------------------------

            annotation = page.load_annot(
                xref
            )

            if annotation is None:
                continue


            # ------------------------------------------------
            # ANNOTATION INFO
            # ------------------------------------------------

            info = (
                annotation.info
                or {}
            )


            # ------------------------------------------------
            # ANNOTATION TYPE
            # ------------------------------------------------

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


            # ------------------------------------------------
            # FILTER
            # ------------------------------------------------

            if (
                annotation_type_name
                not in
                ALLOWED_ANNOTATION_TYPES
            ):
                continue


            # ------------------------------------------------
            # AUTHOR
            # ------------------------------------------------

            author = (
                info.get("title")
                or
                info.get("author")
                or
                None
            )


            # ------------------------------------------------
            # CONTENT
            # ------------------------------------------------

            content = (
                info.get("content")
                or ""
            ).strip()


            # =================================================
            # FREETEXT FALLBACK
            # =================================================

            if (
                not content
                and
                annotation_type_name
                == "FreeText"
            ):

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
                        "FreeText annotation "
                        f"xref {xref}: {exc}"
                    )


            # =================================================
            # STAMP FALLBACK
            # =================================================

            if (
                not content
                and
                annotation_type_name
                == "Stamp"
            ):

                stamp_name = (
                    info.get("name")
                )

                if stamp_name:

                    content = (
                        str(stamp_name)
                        .strip()
                    )


            # =================================================
            # HIGHLIGHT FALLBACK
            # =================================================

            if (
                not content
                and
                annotation_type_name
                == "Highlight"
            ):

                content = (
                    extract_highlight_text(
                        page,
                        annotation
                    )
                )


            # ------------------------------------------------
            # RECTANGLE
            # ------------------------------------------------

            try:
                annotation_rect = pymupdf.Rect(
                    annotation.rect
                )

            except Exception:
                annotation_rect = None


            # ------------------------------------------------
            # CREATED DATE
            # ------------------------------------------------

            (
                created_date,
                created_time

            ) = parse_pdf_datetime(

                info.get(
                    "creationDate"
                )
            )


            # ------------------------------------------------
            # MODIFIED DATE
            # ------------------------------------------------

            (
                modified_date,
                modified_time

            ) = parse_pdf_datetime(

                info.get(
                    "modDate"
                )
            )


            # ------------------------------------------------
            # DUPLICATE CHECK
            # ------------------------------------------------

            if is_duplicate_record(
                records,
                page_number,
                content,
                annotation_rect
            ):

                print(
                    "Duplicate native annotation "
                    f"ignored: {content[:100]}"
                )

                continue


            # ------------------------------------------------
            # ADD RECORD
            # ------------------------------------------------

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

                # Internal value.
                # Excel exporter does not need to use this.
                "_rect":
                    annotation_rect
            })


            print(
                f"EXTRACTED NATIVE -> "
                f"Page: {page_number} | "
                f"Type: "
                f"{annotation_type_name} | "
                f"Text: "
                f"{content[:100]}"
            )


            record_id += 1


        except Exception as exc:

            print(
                f"Error processing annotation "
                f"xref {xref} "
                f"on page {page_number}: "
                f"{exc}"
            )

            continue


    return record_id


# ============================================================
# GET COLOURED TEXT SPANS
# ============================================================

def get_colored_text_spans(page):
    """
    Extract individual coloured text spans from normal PDF
    page content.

    This is how we detect review text that visually looks like
    an annotation but has been flattened / converted into page
    content.
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

            # Type 0 = text block
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
                        span.get("bbox")
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
    Combine neighbouring coloured text spans.

    A single review comment can be stored as several PDF text
    spans or several lines.

    Example:

        "Mention CPC tag number"
        "9002A on drawing"
        "and in file name as well"

    should become one comment.
    """

    if not spans:
        return []


    # Sort top-to-bottom, then left-to-right.
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

            group_rect = group["rect"]
            span_rect = span["rect"]


            # ---------------------------------------------
            # SAME / NEARBY LINE
            # ---------------------------------------------

            vertical_difference = abs(
                span_rect.y0
                -
                group_rect.y1
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
                vertical_difference
                <=
                TEXT_GROUP_DISTANCE
            )


            # ---------------------------------------------
            # HORIZONTAL RELATIONSHIP
            # ---------------------------------------------

            horizontal_overlap = not (
                span_rect.x1
                <
                group_rect.x0 - 20
                or
                span_rect.x0
                >
                group_rect.x1 + 100
            )


            if (
                horizontal_overlap
                and
                (
                    same_line
                    or
                    next_line
                )
            ):

                group["spans"].append(
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


    # --------------------------------------------------------
    # BUILD FINAL TEXT
    # --------------------------------------------------------

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
# EXTRACT FLATTENED / PAGE CONTENT COMMENTS
# ============================================================

def extract_colored_page_comments(
    page,
    page_number,
    records,
    record_id
):
    """
    Extract orange/red/yellow review text that exists as
    ordinary PDF page content instead of an annotation object.
    """

    colored_spans = (
        get_colored_text_spans(
            page
        )
    )


    print(
        f"Page {page_number} "
        f"| Coloured text spans found: "
        f"{len(colored_spans)}"
    )


    grouped_comments = (
        group_colored_spans(
            colored_spans
        )
    )


    print(
        f"Page {page_number} "
        f"| Coloured comment groups: "
        f"{len(grouped_comments)}"
    )


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


        # ----------------------------------------------------
        # Ignore extremely small fragments
        # ----------------------------------------------------

        if len(content) < 2:
            continue


        # ----------------------------------------------------
        # DUPLICATE CHECK
        # ----------------------------------------------------

        if is_duplicate_record(
            records,
            page_number,
            content,
            rect
        ):

            print(
                "Duplicate coloured page text "
                f"ignored: {content[:100]}"
            )

            continue


        # ----------------------------------------------------
        # ADD RECORD
        # ----------------------------------------------------

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

            # Internal position for deduplication.
            "_rect":
                rect
        })


        print(
            f"EXTRACTED FLATTENED -> "
            f"Page: {page_number} | "
            f"Text: {content[:150]}"
        )


        record_id += 1


    return record_id


# ============================================================
# FINAL DUPLICATE CLEANUP
# ============================================================

def remove_final_duplicates(records):
    """
    Final safety-net duplicate removal.

    Keeps comments at different physical positions.

    Removes duplicates when:

        same page
        +
        same normalized text
        +
        same / overlapping location

    If no rectangle exists, same page + same text is used.
    """

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
                f"Text: {text[:100]}"
            )

            continue


        unique_records.append(
            record
        )


    # Re-number Record IDs after duplicate removal.
    for index, record in enumerate(
        unique_records,
        start=1
    ):

        record["record_id"] = index


    return unique_records


# ============================================================
# REMOVE INTERNAL FIELDS
# ============================================================

def clean_internal_fields(records):
    """
    Remove internal helper values before returning records.

    _rect is only needed for duplicate checking and should not
    appear in Excel / API output.
    """

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
    Extract review comments from PDF.

    PASS 1
    ------
    Native PDF annotations:

        FreeText
        Highlight
        Stamp

    PASS 2
    ------
    Coloured page-content review text:

        Orange
        Red
        Yellow / Gold

    PASS 3
    ------
    Remove duplicates.

    This allows the extractor to handle both:

        normal editable PDF annotations

    and

        review comments that have been flattened / converted
        into ordinary page content.
    """


    # ========================================================
    # RESULT RECORDS
    # ========================================================

    records = []


    # ========================================================
    # OPEN PDF
    # ========================================================

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
        # LOOP THROUGH EVERY PAGE
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

            print(
                "\nPASS 1: "
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
            # COLOURED / FLATTENED PAGE TEXT
            # =================================================

            print(
                "\nPASS 2: "
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
        # FINAL DUPLICATE CLEANUP
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


        # ====================================================
        # SUMMARY
        # ====================================================

        summary[
            "annotations"
        ] = len(
            records
        )


        # ====================================================
        # REMOVE INTERNAL RECT VALUES
        # ====================================================

        records = (
            clean_internal_fields(
                records
            )
        )


    # ========================================================
    # DEBUG SUMMARY
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


    # ========================================================
    # RETURN
    # ========================================================

    return {

        "summary":
            summary,

        "records":
            records
    }