import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from run_dpsk_ocr_pdf import pdf_to_text

app = FastAPI(
    title="DeepSeek OCR API",
    description="Extract text from PDFs using DeepSeek OCR",
)

# Load config from env or defaults (adjust as needed)
MODEL_PATH = os.getenv("MODEL_PATH", "/path/to/model")
PROMPT = os.getenv("PROMPT", "Extract text from this image.")
CROP_MODE = os.getenv("CROP_MODE", "auto")  # Adjust based on your config
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "1"))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "4"))
SKIP_REPEAT = os.getenv("SKIP_REPEAT", "false").lower() == "true"


@app.post("/ocr", summary="Extract text from PDF")
async def extract_text(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400, detail="Only PDF files are supported"
        )  # noqa: E501  # noqa: E501

    try:
        pdf_bytes = await file.read()
        result = pdf_to_text(
            input_pdf_bytes=pdf_bytes,
            model_path=MODEL_PATH,
            prompt=PROMPT,
            crop_mode=CROP_MODE,
            max_concurrency=MAX_CONCURRENCY,
            num_workers=NUM_WORKERS,
            skip_repeat=SKIP_REPEAT,
        )
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
