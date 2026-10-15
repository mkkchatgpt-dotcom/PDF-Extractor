from pathlib import Path

import shutil

import uuid

import sys

import tempfile

import threading

import webbrowser

import time

import zipfile

import re

from urllib.parse import quote


from fastapi import FastAPI, UploadFile, File, HTTPException

from fastapi.responses import FileResponse

from fastapi.staticfiles import StaticFiles


from pdf_extractor import extract_pdf

from excel_exporter import generate_excel

import ocr_scanned

# ============================================================

# PATHS

# Works with normal Python AND PyInstaller EXE

# ============================================================


if getattr(sys, "frozen", False):

    # Running as PyInstaller EXE

    RESOURCE_DIR = Path(sys._MEIPASS)

    # React production build bundled inside EXE

    FRONTEND_DIST = RESOURCE_DIR / "frontend" / "dist"

    # Writable temporary directory

    DATA_DIR = Path(tempfile.gettempdir()) / "PDFReviewExtractor"

    UPLOAD_DIR = DATA_DIR / "uploads"

    EXPORT_DIR = DATA_DIR / "exports"


else:

    # Running normally using Python

    BASE_DIR = Path(__file__).resolve().parent

    PROJECT_DIR = BASE_DIR.parent

    FRONTEND_DIST = PROJECT_DIR / "frontend" / "dist"

    UPLOAD_DIR = BASE_DIR / "uploads"

    EXPORT_DIR = BASE_DIR / "exports"


# Create folders if they do not exist

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


EXPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================

# FASTAPI APPLICATION

# ============================================================


app = FastAPI(title="PDF Review Extractor", version="2.1")


# Each extraction batch gets its own folder inside EXPORT_DIR, so every
# Excel file can keep exactly the PDF's name (QA-6922-1.pdf -> QA-6922-1.xlsx)
# without a prefix, even when several users / batches use the same names.

JOB_ID_RE = re.compile(r"^[0-9a-f]{32}$")

ZIP_FILENAME = "PDF_Annotations.zip"

# Batch folders older than this are deleted on the next extraction.
EXPORT_MAX_AGE_SECONDS = 24 * 60 * 60


def cleanup_old_exports():

    cutoff = time.time() - EXPORT_MAX_AGE_SECONDS

    for folder in EXPORT_DIR.iterdir():

        try:

            if folder.is_dir() and folder.stat().st_mtime < cutoff:

                shutil.rmtree(folder, ignore_errors=True)

        except OSError:

            pass


def excel_name_for(pdf_filename, used_names):
    """QA-6922-1.pdf -> QA-6922-1.xlsx (QA-6922-1 (2).xlsx if repeated)."""

    stem = Path(pdf_filename).stem.strip() or "annotations"

    candidate = f"{stem}.xlsx"

    counter = 2

    while candidate.lower() in used_names:

        candidate = f"{stem} ({counter}).xlsx"

        counter += 1

    used_names.add(candidate.lower())

    return candidate


def job_file(job_id, filename):
    """Resolve a download path safely (no directory traversal)."""

    if not JOB_ID_RE.match(job_id):

        raise HTTPException(status_code=404, detail="File not found.")

    safe_filename = Path(filename).name

    path = EXPORT_DIR / job_id / safe_filename

    if not safe_filename or not path.is_file():

        raise HTTPException(status_code=404, detail="File not found.")

    return path


# ============================================================

# STARTUP INFORMATION

# ============================================================


def report_ocr_status():

    error = ocr_scanned.ocr_error()

    if error:

        print(f"[PDF Review Extractor] OCR NOT AVAILABLE - {error}")

    else:

        print("[PDF Review Extractor] OCR engine (RapidOCR) loaded.")


# ============================================================

# HEALTH CHECK

# ============================================================


@app.get("/health")
def health():

    return {
        "status": "ok",
        "application": "PDF Review Extractor",
        "frontend_found": FRONTEND_DIST.exists(),
        "ocr_available": ocr_scanned.ocr_available(),
        "ocr_error": ocr_scanned.ocr_error(),
    }


# ============================================================

# MULTIPLE PDF EXTRACTION API

# ============================================================


@app.post("/api/extract")
async def extract(
    files: list[UploadFile] = File(...),
):

    # Scanned pages are always read in "auto" mode: typed comment
    # sheets are read in full, coloured review ink elsewhere.

    scanned_text_mode = "auto"

    # --------------------------------------------------------

    # CHECK FILES

    # --------------------------------------------------------

    if not files:

        raise HTTPException(
            status_code=400, detail="Please upload at least one PDF file."
        )

    # --------------------------------------------------------

    # VALIDATE ALL FILES

    # --------------------------------------------------------

    for file in files:

        if not file.filename or not file.filename.lower().endswith(".pdf"):

            raise HTTPException(
                status_code=400,
                detail=(
                    f"{file.filename or 'Unknown file'} " f"is not a valid PDF file."
                ),
            )

    # --------------------------------------------------------

    # CREATE UNIQUE BATCH ID

    # --------------------------------------------------------

    cleanup_old_exports()

    job_id = uuid.uuid4().hex

    job_dir = EXPORT_DIR / job_id

    job_dir.mkdir(parents=True, exist_ok=True)

    # Excel names already used in this batch (case-insensitive,
    # because Windows treats A.xlsx and a.xlsx as the same file).

    used_excel_names = set()

    # --------------------------------------------------------

    # ZIP FILE

    # --------------------------------------------------------

    zip_filename = ZIP_FILENAME

    zip_path = job_dir / zip_filename

    # Results returned to React

    results = []

    # Excel files that will be added to ZIP

    created_excel_files = []

    try:

        # ====================================================

        # PROCESS EACH PDF

        # ====================================================

        for file_index, file in enumerate(files, start=1):

            # ------------------------------------------------

            # UNIQUE TEMP PDF PATH

            # ------------------------------------------------

            pdf_id = uuid.uuid4().hex

            pdf_path = UPLOAD_DIR / f"{pdf_id}.pdf"

            # ------------------------------------------------

            # EXCEL FILE NAME

            # ------------------------------------------------

            # Same name as the PDF, no prefix or suffix:
            # QA-6922-1.pdf -> QA-6922-1.xlsx

            excel_filename = excel_name_for(file.filename, used_excel_names)

            # The batch folder (job_dir) keeps batches apart,
            # so the file itself needs no prefix.

            excel_path = job_dir / excel_filename

            try:

                # ============================================

                # SAVE PDF TEMPORARILY

                # ============================================

                with pdf_path.open("wb") as buffer:

                    shutil.copyfileobj(file.file, buffer)

                # ============================================

                # EXTRACT PDF ANNOTATIONS

                # ============================================

                extraction_result = extract_pdf(
                    pdf_path, scanned_text_mode=scanned_text_mode
                )

                # ============================================

                # GENERATE EXCEL

                # ============================================

                generate_excel(
                    extraction_result["records"],
                    excel_path,
                    filename=file.filename,
                )

                # ============================================

                # STORE EXCEL FOR ZIP CREATION

                # ============================================

                created_excel_files.append((excel_path, excel_filename))

                # ============================================

                # RESULT FOR REACT

                # ============================================

                results.append(
                    {
                        "filename": file.filename,
                        "summary": extraction_result["summary"],
                        "records": extraction_result["records"],
                        "download_url": (
                            f"/api/download/{job_id}/{quote(excel_filename)}"
                        ),
                        "excel_filename": excel_filename,
                        "success": True,
                        "error": None,
                    }
                )

            except Exception as exc:

                # ============================================

                # ERROR FOR THIS PARTICULAR PDF

                # ============================================

                results.append(
                    {
                        "filename": file.filename,
                        "summary": {"pages": 0, "annotations": 0},
                        "records": [],
                        "success": False,
                        "error": str(exc),
                    }
                )

            finally:

                # ============================================

                # DELETE TEMPORARY PDF

                # ============================================

                pdf_path.unlink(missing_ok=True)

        # ====================================================

        # MAKE SURE AT LEAST ONE FILE WORKED

        # ====================================================

        if not created_excel_files:

            raise HTTPException(
                status_code=500,
                detail=("None of the selected PDFs " "could be processed."),
            )

        # ====================================================

        # CREATE ZIP FILE

        # ====================================================

        # Single uploaded PDF: XLSX. Multiple uploaded PDFs: ZIP.
        download_type = "xlsx" if len(files) == 1 else "zip"
        if download_type == "xlsx":
            download_url = next(
                item["download_url"] for item in results if item["success"]
            )
        else:
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
                for excel_path, excel_filename in created_excel_files:
                    # Inside the ZIP each file keeps the PDF's name.
                    zip_file.write(excel_path, arcname=excel_filename)
            download_url = f"/api/download-zip/{job_id}/{quote(zip_filename)}"

        # Keep XLSX outputs available for individual downloads.

        successful_files = sum(1 for item in results if item["success"])

        failed_files = len(files) - successful_files

        # ====================================================

        # RETURN BATCH RESULT TO REACT

        # ====================================================

        return {
            "total_files": len(files),
            "successful_files": successful_files,
            "failed_files": failed_files,
            "results": results,
            "download_url": download_url,
            "download_type": download_type,
        }

    except HTTPException:

        raise

    except Exception as exc:

        raise HTTPException(status_code=500, detail=str(exc))


# ============================================================

# ZIP DOWNLOAD API

# ============================================================


@app.get("/api/download-zip/{job_id}/{filename}")
def download_zip(job_id: str, filename: str):

    zip_path = job_file(job_id, filename)

    return FileResponse(
        path=zip_path, media_type="application/zip", filename=zip_path.name
    )


# ============================================================

# INDIVIDUAL EXCEL DOWNLOAD API

#

# The downloaded file is named exactly like the PDF:

# QA-6922-1.pdf -> QA-6922-1.xlsx

# ============================================================


@app.get("/api/download/{job_id}/{filename}")
def download(job_id: str, filename: str):

    path = job_file(job_id, filename)

    return FileResponse(
        path=path,
        media_type=(
            "application/" "vnd.openxmlformats-officedocument." "spreadsheetml.sheet"
        ),
        filename=path.name,
    )


# ============================================================

# REACT STATIC ASSETS

# ============================================================


if FRONTEND_DIST.exists():

    ASSETS_DIR = FRONTEND_DIST / "assets"

    if ASSETS_DIR.exists():

        app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")


# ============================================================

# REACT FRONTEND

#

# IMPORTANT:

# This route MUST remain after all API routes.

# ============================================================


@app.get("/{full_path:path}")
def serve_frontend(full_path: str):

    # --------------------------------------------------------

    # DON'T RETURN REACT FOR INVALID API REQUESTS

    # --------------------------------------------------------

    if full_path.startswith("api/"):

        raise HTTPException(status_code=404, detail="API endpoint not found.")

    # --------------------------------------------------------

    # FIND INDEX.HTML

    # --------------------------------------------------------

    index_file = FRONTEND_DIST / "index.html"

    if not index_file.exists():

        raise HTTPException(
            status_code=404, detail=("Frontend build not found at " f"{FRONTEND_DIST}")
        )

    # --------------------------------------------------------

    # SERVE ACTUAL FRONTEND FILE

    # --------------------------------------------------------

    requested_file = FRONTEND_DIST / full_path

    if full_path and requested_file.is_file():

        return FileResponse(requested_file)

    # --------------------------------------------------------

    # REACT SPA FALLBACK

    # --------------------------------------------------------

    return FileResponse(index_file)


# ============================================================

# OPEN BROWSER

# ============================================================


def open_browser():

    # Give Uvicorn time to start

    time.sleep(2)

    webbrowser.open("http://127.0.0.1:8000")


# ============================================================

# APPLICATION ENTRY POINT

#

# Works with:

#

# python main.py

#

# OR

#

# PDFReviewExtractor.exe

# ============================================================


if __name__ == "__main__":

    # Quick self-test for a built EXE:  PDFReviewExtractor.exe --check-ocr

    if "--check-ocr" in sys.argv:

        error = ocr_scanned.ocr_error()

        print("OCR OK - scanned pages can be read." if not error else f"OCR FAILED - {error}")

        sys.exit(1 if error else 0)

    import uvicorn

    # Tell the user in the console whether scanned pages can be read

    report_ocr_status()

    # Open browser automatically

    threading.Thread(target=open_browser, daemon=True).start()

    # Start FastAPI

    # ws="none": the app uses no WebSockets. Without this, uvicorn tries to
    # load the "websockets" library at start-up, which PyInstaller does not
    # bundle completely ("cannot import name '__version__' from 'websockets'").

    uvicorn.run(
        app, host="127.0.0.1", port=8000, log_level="info", ws="none", http="h11"
    )
