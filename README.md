# AI-Powered Medical Report Simplifier

## Overview

This Flask-based backend takes medical reports (text or scanned images) and outputs:

* Extracted tests (`tests_raw`)
* Normalized values with reference ranges
* Patient-friendly explanations
* JSON summary via Gemini API

Supports **OCR** (pytesseract) and local extraction fallback.

## Features

* OCR for image reports (png, jpg, jpeg, bmp, tiff)
* Test normalization with default reference ranges
* AI explanations using Gemini API
* Hallucination guardrail
* Web interface for uploads and JSON output

## Architecture

```
[Report (Text/Image)] --> [Flask API] --> [OCR/Extraction] --> [Normalization] --> [Gemini AI] --> [JSON Output]
```

## Requirements

* Python 3.9+
* Flask
* google-generativeai (`pip install google-generativeai`)
* Pillow + pytesseract (optional, for OCR)
* ngrok (for local demo)
* Gemini API key

## Installation

```bash
git clone https://github.com/<username>/ai-medical-report-simplifier.git
cd ai-medical-report-simplifier
python -m venv venv
# Activate venv:
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
pip install -r requirements.txt
```

Set your Gemini API key in `app.py`:

```python
GEMINI_API_KEY = "YOUR_KEY"
```

(Optional) Install Tesseract OCR engine:

* Windows: [https://github.com/tesseract-ocr/tesseract](https://github.com/tesseract-ocr/tesseract)
* Linux: `sudo apt install tesseract-ocr`

## Run Locally

```bash
python app.py
```

Open browser: `http://localhost:8000`

Expose via ngrok:

```bash
ngrok http 8000
```

Copy the ngrok URL for remote demo.

## API Usage

### Endpoint: `/` (POST)

Upload a medical report file (`txt` or image).

**Curl Example:**

```bash
curl -X POST http://localhost:8000/ -F "report=@sample_reports/sample_report.txt"
```

### Sample JSON Response

```json
{
  "confidence": 0.8,
  "explanations": [
    "An elevated white blood cell (WBC) count can suggest that your body is fighting an infection or experiencing inflammation."
  ],
  "normalization_confidence": 0.9,
  "status": "ok",
  "summary": "Your lab results indicate that your white blood cell count is elevated. Your platelet count is within the expected range.",
  "tests": [
    {
      "name": "WBC",
      "ref_range": {"high": 11000, "low": 4000},
      "status": "high",
      "unit": "/uL",
      "value": 11200.0
    },
    {
      "name": "Platelets",
      "ref_range": {"high": 450000, "low": 150000},
      "status": "normal",
      "unit": "/uL",
      "value": 200000.0
    }
  ],
  "tests_raw": [
    "CBC: Hemoglobin 10.2 g/dL (Low)",
    "WBC 11200 /uL (High)",
    "Platelets 200000 /uL (Normal)"
  ]
}
```

## Postman Testing

1. Create a **POST request** to `http://localhost:8000/`
2. Key: `report` → attach file
3. Send → receive JSON response

## Sample Reports

Place reports in `sample_reports/`:

```
sample_reports/
├── sample_report.txt
├── sample_report2.png
```

* Text example:

```
CBC: Hemoglobin 10.2 g/dL (Low)
WBC 11200 /uL (High)
Platelets 200000 /uL (Normal)
```

* Image reports can be screenshots or scanned PDFs converted to images.

<<<<<<< HEAD

=======
>>>>>>> dc89a1cc56f09cbc01ff27e15e92e6717336ad0d

## Folder Structure

```
ai-medical-report-simplifier/
├── app.py
├── templates/index.html
├── sample_reports/sample_report.txt
├── requirements.txt
├── README.md
```


