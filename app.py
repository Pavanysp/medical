import os
import re
import json
import tempfile
from flask import Flask, request, render_template, send_from_directory
import pytesseract
from PIL import Image
import google.generativeai as genai

# ==========================
# Flask App Setup
# ==========================
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'sample_reports'

# Gemini API Key

GEMINI_API_KEY = "Your_Api_Key"
genai.configure(api_key=GEMINI_API_KEY)

# OCR Function

def ocr_from_image_file(file_stream):
    """Extract text from image using Tesseract OCR with preprocessing."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
        temp_file.write(file_stream.read())
        temp_path = temp_file.name

    try:
        image = Image.open(temp_path).convert("L")
        threshold = 140
        image = image.point(lambda p: 255 if p > threshold else 0)
        image = image.resize((image.width * 2, image.height * 2))
        text = pytesseract.image_to_string(image, config="--psm 6")
    except Exception as e:
        print("OCR error:", e)
        text = ""
    finally:
        os.remove(temp_path)

    return text

# Safe JSON Parsing

def safe_json_loads(text):
    """Extract and safely load JSON from LLM output."""
    try:
        matches = re.findall(r"\{.*\}", text, re.DOTALL)
        if not matches:
            return {"status": "unprocessed", "reason": "No valid JSON generated"}
        raw = max(matches, key=len)
        raw = re.sub(r",\s*([\]}])", r"\1", raw)  # remove trailing commas
        return json.loads(raw)
    except Exception as e:
        print("JSON parse error:", e)
        return {"status": "unprocessed", "reason": "Invalid JSON from model"}


# Gemini Processing

def generate_summary_gemini(extracted_text, temperature=0.3):
    """
    Feed OCR/text to Gemini for parsing, normalization, and patient-friendly summary.
    Explanations are short and calm for ALL tests, summary is shorter.
    """
    prompt = f"""
You are a medical text simplifier.
Input medical report text:
{extracted_text}

Steps:
1. Extract all tests, values, units, and status (low/normal/high).
2. If ranges are present, include them; otherwise leave null.
3. Generate explanations for ALL tests, each ≤18 words, calm and patient-friendly.
4. Generate a very short summary (≤20 words) combining key points.
5. Return a confidence score (0.0–1.0) for extraction and normalization.

Output ONLY valid JSON:
{{
  "tests_raw": ["list of raw extracted test lines"],
  "confidence": 0.82,
  "tests": [
    {{"name":"Hemoglobin","value":10.2,"unit":"g/dL","status":"low","ref_range":{{"low":12.0,"high":15.0}}}},
    {{"name":"WBC","value":11200,"unit":"/uL","status":"high","ref_range":{{"low":4000,"high":11000}}}}
  ],
  "normalization_confidence": 0.84,
  "explanations": [
    "Hemoglobin is slightly low, linked to low blood levels.",
    "White blood cells are high, suggesting body defense activity."
  ],
  "summary": "Low hemoglobin and high WBC detected.",
  "status": "ok"
}}
"""

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(
            temperature=temperature, top_k=1
        ))
        return safe_json_loads(response.text.strip())
    except Exception as e:
        print("Gemini error:", e)
        return {"status": "unprocessed", "reason": "Gemini processing error"}


# Routes

@app.route("/", methods=["GET", "POST"])
def index():
    final_output = {}
    uploaded_file_name = ""

    if request.method == "POST":
        file = request.files.get("report")
        if file and file.filename:
            uploaded_file_name = file.filename.lower()
            extracted_text = ""

            try:
                if uploaded_file_name.endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp")):
                    file.stream.seek(0)
                    extracted_text = ocr_from_image_file(file.stream)
                else:
                    file.stream.seek(0)
                    extracted_text = file.stream.read().decode("utf-8", errors="ignore")
            except Exception as e:
                print("OCR/Text read fallback:", e)
                extracted_text = ""

            if not extracted_text.strip():
                final_output = {"status": "unprocessed", "reason": "No valid text found"}
            else:
                gemini_output = generate_summary_gemini(extracted_text)
                final_output = gemini_output

    return render_template("index.html", final_output=final_output, filename=uploaded_file_name)

@app.route("/sample_reports/<filename>")
def download_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


# Run App

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
