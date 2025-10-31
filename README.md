# DeepSeek OCR PDF API

A FastAPI-based web service for extracting text from PDF documents using the DeepSeek OCR model. The API converts PDF pages to high-quality images and performs optical character recognition (OCR) to extract text content.

## Features

- **PDF to Image Conversion**: High-quality rendering of PDF pages using PyMuPDF
- **OCR Processing**: Leverages DeepSeek OCR model via vLLM for accurate text extraction
- **RESTful API**: Simple FastAPI endpoint for PDF uploads
- **Configurable**: Environment-based configuration for model paths, prompts, and processing options
- **Batch Processing**: Supports concurrent processing of multiple pages

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd deepseek-ocr-pdf-api
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install development dependencies (optional):
   ```bash
   pip install -r requirements-dev.txt
   ```

## Usage

### Running the API Server

Start the FastAPI server:
```bash
python api.py
```

Or using uvicorn:
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`.

### API Documentation

Visit `http://localhost:8000/docs` for interactive Swagger UI documentation.

## API Endpoints

### POST /ocr

Extract text from an uploaded PDF file.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: `file` (PDF file upload)

**Response:**
- Status: `200 OK`
- Content-Type: `application/json`
- Body: JSON object with page-wise text extraction results

**Example Response:**
```json
{
  "page_0": {
    "text": "Extracted text content..."
  },
  "page_1": {
    "text": "More extracted text..."
  }
}
```

**Error Responses:**
- `400 Bad Request`: Invalid file type (only PDFs supported)
- `500 Internal Server Error`: OCR processing failed

## Configuration

Configure the API using environment variables:

- `MODEL_PATH`: Path to the DeepSeek OCR model (default: "/path/to/model")
- `PROMPT`: OCR prompt for the model (default: "Extract text from this image.")
- `CROP_MODE`: Image cropping mode (default: "auto")
- `MAX_CONCURRENCY`: Maximum concurrent requests (default: 1)
- `NUM_WORKERS`: Number of worker threads for image processing (default: 4)
- `SKIP_REPEAT`: Skip pages with repeated content (default: false)

Example:
```bash
export MODEL_PATH="/path/to/your/model"
export PROMPT="Extract all text from this document image."
python api.py
```

## Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended for model inference)
- Sufficient RAM for model loading (depends on model size)

## Development

Run code formatting and linting:
```bash
make tidy
```

This runs:
- `black .` for code formatting
- `isort .` for import sorting
- `flake8 .` for linting