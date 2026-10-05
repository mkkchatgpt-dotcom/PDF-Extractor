from datetime import datetime
import re
import pymupdf


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

        # Remove PDF date prefix
        if value.startswith("D:"):
            value = value[2:]


        # Extract:
        # YYYY MM DD HH MM SS

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
# PDF EXTRACTION
# ============================================================

def extract_pdf(pdf_path):

    """
    Extract supported PDF annotations.

    Currently extracted annotation types:

    1. FreeText
    2. Highlight
    3. Stamp

    Other graphical annotations such as:

    - Line
    - Square
    - Circle
    - Polygon
    - PolyLine
    - Ink
    - StrikeOut
    - Underline

    are ignored.
    """


    # ========================================================
    # RESULT RECORDS
    # ========================================================

    records = []


    # ========================================================
    # ALLOWED ANNOTATION TYPES
    # ========================================================

    allowed_annotation_types = {
        "FreeText",
        "Highlight",
        "Stamp"
    }


    # ========================================================
    # OPEN PDF
    # ========================================================

    with pymupdf.open(pdf_path) as document:


        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        summary = {
            "pages": len(document),
            "annotations": 0
        }


        # ----------------------------------------------------
        # RECORD ID
        # ----------------------------------------------------

        record_id = 1


        # ====================================================
        # LOOP THROUGH EVERY PAGE
        # ====================================================

        for page_number in range(
            1,
            len(document) + 1
        ):


            # Load page
            page = document.load_page(
                page_number - 1
            )


            # Get page rotation
            page_rotation = page.rotation


            print(
                f"Processing Page {page_number} "
                f"| Rotation: {page_rotation} degrees"
            )


            # =================================================
            # COLLECT ANNOTATION XREFS
            # =================================================
            #
            # Instead of keeping annotation objects directly,
            # first collect their XREF IDs.
            #
            # This is more robust when working with PDFs that
            # contain many annotations.
            # =================================================

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

                continue


            print(
                f"Page {page_number} "
                f"| Total annotation objects found: "
                f"{len(annotation_xrefs)}"
            )


            # =================================================
            # PROCESS EACH ANNOTATION
            # =================================================

            for xref in annotation_xrefs:


                try:

                    # -----------------------------------------
                    # LOAD ANNOTATION
                    # -----------------------------------------

                    annotation = page.load_annot(
                        xref
                    )


                    if annotation is None:

                        print(
                            f"Could not load annotation "
                            f"xref {xref}"
                        )

                        continue


                    # -----------------------------------------
                    # ANNOTATION INFORMATION
                    # -----------------------------------------

                    info = (
                        annotation.info
                        or {}
                    )


                    # =========================================
                    # GET ANNOTATION TYPE
                    # =========================================

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


                    # =========================================
                    # FILTER ANNOTATION TYPES
                    # =========================================
                    #
                    # ONLY:
                    #
                    # FreeText
                    # Highlight
                    # Stamp
                    #
                    # Everything else is ignored.
                    # =========================================

                    if (
                        annotation_type_name
                        not in
                        allowed_annotation_types
                    ):

                        continue


                    # =========================================
                    # AUTHOR
                    # =========================================

                    author = (
                        info.get("title")
                        or
                        info.get("author")
                        or
                        None
                    )


                    # =========================================
                    # COMMENT / CONTENT
                    # =========================================

                    content = (
                        info.get("content")
                        or ""
                    ).strip()


                    # =========================================
                    # FREETEXT FALLBACK
                    # =========================================
                    #
                    # Sometimes visible FreeText text is not
                    # available inside:
                    #
                    # info["content"]
                    #
                    # In that situation try reading the text
                    # directly from the annotation.
                    # =========================================

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


                    # =========================================
                    # STAMP FALLBACK
                    # =========================================
                    #
                    # Some stamp annotations don't contain
                    # normal comment content.
                    #
                    # In those cases use the stamp name.
                    # =========================================

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


                    # =========================================
                    # HIGHLIGHT FALLBACK
                    # =========================================
                    #
                    # A Highlight annotation may not have a
                    # comment.
                    #
                    # If the comment is empty, try extracting
                    # the actual highlighted text from the PDF.
                    # =========================================

                    if (
                        not content
                        and
                        annotation_type_name
                        == "Highlight"
                    ):

                        try:

                            vertices = (
                                annotation.vertices
                            )


                            if vertices:

                                highlighted_text = []


                                # ---------------------------------
                                # Highlight vertices normally come
                                # in groups of four points.
                                # ---------------------------------

                                for i in range(
                                    0,
                                    len(vertices),
                                    4
                                ):


                                    quad_points = (
                                        vertices[
                                            i:i + 4
                                        ]
                                    )


                                    if (
                                        len(quad_points)
                                        < 4
                                    ):

                                        continue


                                    # -----------------------------
                                    # X coordinates
                                    # -----------------------------

                                    xs = [

                                        point.x

                                        for point
                                        in quad_points

                                    ]


                                    # -----------------------------
                                    # Y coordinates
                                    # -----------------------------

                                    ys = [

                                        point.y

                                        for point
                                        in quad_points

                                    ]


                                    # -----------------------------
                                    # CREATE RECTANGLE
                                    # -----------------------------

                                    rect = (
                                        pymupdf.Rect(
                                            min(xs),
                                            min(ys),
                                            max(xs),
                                            max(ys)
                                        )
                                    )


                                    # -----------------------------
                                    # EXTRACT TEXT
                                    # -----------------------------

                                    text = (
                                        page.get_textbox(
                                            rect
                                        )
                                    )


                                    if text:

                                        highlighted_text.append(
                                            text.strip()
                                        )


                                # ---------------------------------
                                # COMBINE HIGHLIGHTED TEXT
                                # ---------------------------------

                                if highlighted_text:

                                    content = " ".join(
                                        highlighted_text
                                    ).strip()


                        except Exception as exc:

                            print(
                                f"Could not extract "
                                f"highlighted text "
                                f"on page "
                                f"{page_number}: "
                                f"{exc}"
                            )


                    # =========================================
                    # CREATED DATE / TIME
                    # =========================================

                    (
                        created_date,
                        created_time

                    ) = parse_pdf_datetime(

                        info.get(
                            "creationDate"
                        )

                    )


                    # =========================================
                    # MODIFIED DATE / TIME
                    # =========================================

                    (
                        modified_date,
                        modified_time

                    ) = parse_pdf_datetime(

                        info.get(
                            "modDate"
                        )

                    )


                    # =========================================
                    # ADD RECORD
                    # =========================================

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
                            content

                    })


                    # =========================================
                    # DEBUG INFORMATION
                    # =========================================

                    print(
                        f"EXTRACTED -> "
                        f"Page: {page_number} | "
                        f"Rotation: "
                        f"{page_rotation} | "
                        f"Type: "
                        f"{annotation_type_name} | "
                        f"Text: "
                        f"{content[:100]}"
                    )


                    # =========================================
                    # UPDATE COUNTERS
                    # =========================================

                    record_id += 1

                    summary[
                        "annotations"
                    ] += 1


                # =============================================
                # ONE BAD ANNOTATION SHOULD NOT STOP THE PDF
                # =============================================

                except Exception as exc:

                    print(
                        f"Error processing annotation "
                        f"xref {xref} "
                        f"on page {page_number}: "
                        f"{exc}"
                    )

                    continue


    # ========================================================
    # RETURN RESULT
    # ========================================================

    return {

        "summary":
            summary,

        "records":
            records

    }




