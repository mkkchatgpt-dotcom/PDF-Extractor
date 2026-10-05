from pathlib import Path

import shutil

import uuid

import sys

import tempfile

import threading

import webbrowser

import time

import zipfile



from fastapi import FastAPI, UploadFile, File, HTTPException

from fastapi.responses import FileResponse

from fastapi.staticfiles import StaticFiles



from pdf_extractor import extract_pdf

from excel_exporter import generate_excel





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

UPLOAD_DIR.mkdir(

    parents=True,

    exist_ok=True

)



EXPORT_DIR.mkdir(

    parents=True,

    exist_ok=True

)





# ============================================================

# FASTAPI APPLICATION

# ============================================================



app = FastAPI(

    title="PDF Review Extractor",

    version="2.0"

)





# ============================================================

# STARTUP INFORMATION

# ============================================================



print("=" * 60)

print("PDF Review Extractor")

print("=" * 60)

print(f"Frontend path : {FRONTEND_DIST}")

print(f"Frontend found: {FRONTEND_DIST.exists()}")

print(f"Upload folder : {UPLOAD_DIR}")

print(f"Export folder : {EXPORT_DIR}")

print("=" * 60)





# ============================================================

# HEALTH CHECK

# ============================================================



@app.get("/health")

def health():



    return {

        "status": "ok",

        "application": "PDF Review Extractor",

        "frontend_found": FRONTEND_DIST.exists()

    }





# ============================================================

# MULTIPLE PDF EXTRACTION API

# ============================================================



@app.post("/api/extract")

async def extract(

    files: list[UploadFile] = File(...)

):



    # --------------------------------------------------------

    # CHECK FILES

    # --------------------------------------------------------



    if not files:



        raise HTTPException(

            status_code=400,

            detail="Please upload at least one PDF file."

        )





    # --------------------------------------------------------

    # VALIDATE ALL FILES

    # --------------------------------------------------------



    for file in files:



        if (

            not file.filename

            or not file.filename.lower().endswith(".pdf")

        ):



            raise HTTPException(

                status_code=400,

                detail=(

                    f"{file.filename or 'Unknown file'} "

                    f"is not a valid PDF file."

                )

            )





    # --------------------------------------------------------

    # CREATE UNIQUE BATCH ID

    # --------------------------------------------------------



    job_id = uuid.uuid4().hex





    # --------------------------------------------------------

    # ZIP FILE

    # --------------------------------------------------------



    zip_filename = (

        f"PDF_Annotations_{job_id[:8]}.zip"

    )



    zip_path = (

        EXPORT_DIR /

        zip_filename

    )





    # Results returned to React

    results = []



    # Excel files that will be added to ZIP

    created_excel_files = []





    try:



        # ====================================================

        # PROCESS EACH PDF

        # ====================================================



        for file_index, file in enumerate(

            files,

            start=1

        ):



            print("=" * 60)

            print(

                f"Processing PDF "

                f"{file_index}/{len(files)}"

            )

            print(

                f"File: {file.filename}"

            )

            print("=" * 60)





            # ------------------------------------------------

            # UNIQUE TEMP PDF PATH

            # ------------------------------------------------



            pdf_id = uuid.uuid4().hex



            pdf_path = (

                UPLOAD_DIR /

                f"{pdf_id}.pdf"

            )





            # ------------------------------------------------

            # EXCEL FILE NAME

            # ------------------------------------------------



            original_stem = Path(

                file.filename

            ).stem



            excel_filename = (

                f"{original_stem}_annotations.xlsx"

            )





            # Actual temporary Excel path.

            # job ID avoids filename collisions.

            excel_path = (

                EXPORT_DIR /

                f"{job_id[:8]}_"

                f"{file_index}_"

                f"{excel_filename}"

            )





            try:



                # ============================================

                # SAVE PDF TEMPORARILY

                # ============================================



                with pdf_path.open("wb") as buffer:



                    shutil.copyfileobj(

                        file.file,

                        buffer

                    )





                print(

                    f"Temporary PDF saved: "

                    f"{pdf_path}"

                )





                # ============================================

                # EXTRACT PDF ANNOTATIONS

                # ============================================



                extraction_result = extract_pdf(

                    pdf_path

                )





                # ============================================

                # GENERATE EXCEL

                # ============================================



                generate_excel(
                    extraction_result["records"],
                    excel_path,
                    filename=file.filename,
                )





                print(

                    f"Excel generated: "

                    f"{excel_filename}"

                )





                # ============================================

                # STORE EXCEL FOR ZIP CREATION

                # ============================================



                created_excel_files.append(

                    (

                        excel_path,

                        excel_filename

                    )

                )





                # ============================================

                # RESULT FOR REACT

                # ============================================



                results.append({



                    "filename":

                        file.filename,



                    "summary":

                        extraction_result[

                            "summary"

                        ],



                    "records":

                        extraction_result[

                            "records"

                        ],



                    "download_url": f"/api/download/{excel_path.name}",

                    "success":

                        True,



                    "error":

                        None

                })





                print(

                    f"Successfully processed: "

                    f"{file.filename}"

                )





            except Exception as exc:



                # ============================================

                # ERROR FOR THIS PARTICULAR PDF

                # ============================================



                print(

                    f"Error processing "

                    f"{file.filename}: "

                    f"{str(exc)}"

                )





                results.append({



                    "filename":

                        file.filename,



                    "summary": {

                        "pages": 0,

                        "annotations": 0

                    },



                    "records": [],



                    "success":

                        False,



                    "error":

                        str(exc)

                })





            finally:



                # ============================================

                # DELETE TEMPORARY PDF

                # ============================================



                pdf_path.unlink(

                    missing_ok=True

                )





                print(

                    f"Temporary PDF deleted: "

                    f"{pdf_path}"

                )





        # ====================================================

        # MAKE SURE AT LEAST ONE FILE WORKED

        # ====================================================



        if not created_excel_files:



            raise HTTPException(

                status_code=500,

                detail=(

                    "None of the selected PDFs "

                    "could be processed."

                )

            )





        # ====================================================

        # CREATE ZIP FILE

        # ====================================================



        # Single uploaded PDF: XLSX. Multiple uploaded PDFs: ZIP.
        download_type = "xlsx" if len(files) == 1 else "zip"
        if download_type == "xlsx":
            download_url = next(item["download_url"] for item in results if item["success"])
        else:
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
                for index, (excel_path, excel_filename) in enumerate(created_excel_files, start=1):
                    zip_file.write(excel_path, arcname=f"{index}_{excel_filename}")
            download_url = f"/api/download-zip/{zip_filename}"

        # Keep XLSX outputs available for individual downloads.

        successful_files = sum(

            1

            for item in results

            if item["success"]

        )



        failed_files = (

            len(files)

            - successful_files

        )





        # ====================================================

        # RETURN BATCH RESULT TO REACT

        # ====================================================



        return {



            "total_files":

                len(files),



            "successful_files":

                successful_files,



            "failed_files":

                failed_files,



            "results":

                results,



            "download_url": download_url,
            "download_type": download_type

        }





    except HTTPException:



        raise





    except Exception as exc:



        print(

            "Batch extraction error:",

            str(exc)

        )



        raise HTTPException(

            status_code=500,

            detail=str(exc)

        )





# ============================================================

# ZIP DOWNLOAD API

# ============================================================



@app.get(

    "/api/download-zip/{filename}"

)

def download_zip(

    filename: str

):



    # Prevent directory traversal

    safe_filename = Path(

        filename

    ).name



    zip_path = (

        EXPORT_DIR /

        safe_filename

    )





    if not zip_path.exists():



        raise HTTPException(

            status_code=404,

            detail="ZIP file not found."

        )





    return FileResponse(

        path=zip_path,

        media_type="application/zip",

        filename=safe_filename

    )





# ============================================================

# OPTIONAL OLD EXCEL DOWNLOAD API

#

# Keeping this endpoint does not cause any problem.

# It may be useful if individual Excel download is added later.

# ============================================================



@app.get(

    "/api/download/{filename}"

)

def download(

    filename: str

):



    # Prevent directory traversal

    safe_filename = Path(

        filename

    ).name



    path = (

        EXPORT_DIR /

        safe_filename

    )





    if not path.exists():



        raise HTTPException(

            status_code=404,

            detail="File not found."

        )





    return FileResponse(

        path=path,

        media_type=(

            "application/"

            "vnd.openxmlformats-officedocument."

            "spreadsheetml.sheet"

        ),

        filename=safe_filename

    )





# ============================================================

# REACT STATIC ASSETS

# ============================================================



if FRONTEND_DIST.exists():



    ASSETS_DIR = (

        FRONTEND_DIST /

        "assets"

    )



    if ASSETS_DIR.exists():



        app.mount(

            "/assets",

            StaticFiles(

                directory=str(

                    ASSETS_DIR

                )

            ),

            name="assets"

        )





# ============================================================

# REACT FRONTEND

#

# IMPORTANT:

# This route MUST remain after all API routes.

# ============================================================



@app.get("/{full_path:path}")

def serve_frontend(

    full_path: str

):



    # --------------------------------------------------------

    # DON'T RETURN REACT FOR INVALID API REQUESTS

    # --------------------------------------------------------



    if full_path.startswith(

        "api/"

    ):



        raise HTTPException(

            status_code=404,

            detail="API endpoint not found."

        )





    # --------------------------------------------------------

    # FIND INDEX.HTML

    # --------------------------------------------------------



    index_file = (

        FRONTEND_DIST /

        "index.html"

    )





    if not index_file.exists():



        raise HTTPException(

            status_code=404,

            detail=(

                "Frontend build not found at "

                f"{FRONTEND_DIST}"

            )

        )





    # --------------------------------------------------------

    # SERVE ACTUAL FRONTEND FILE

    # --------------------------------------------------------



    requested_file = (

        FRONTEND_DIST /

        full_path

    )





    if (

        full_path

        and requested_file.is_file()

    ):



        return FileResponse(

            requested_file

        )





    # --------------------------------------------------------

    # REACT SPA FALLBACK

    # --------------------------------------------------------



    return FileResponse(

        index_file

    )





# ============================================================

# OPEN BROWSER

# ============================================================



def open_browser():



    # Give Uvicorn time to start

    time.sleep(2)



    webbrowser.open(

        "http://127.0.0.1:8000"

    )





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



    import uvicorn





    # Open browser automatically

    threading.Thread(

        target=open_browser,

        daemon=True

    ).start()





    # Start FastAPI

    uvicorn.run(

        app,

        host="127.0.0.1",

        port=8000,

        log_level="info"

    )