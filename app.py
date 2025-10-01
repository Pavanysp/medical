from flask import Flask, request, render_template, jsonify, send_from_directory
import os
import re
import json
import tempfile
import google.generativeai as genai

try:
    from PIL import Image
    import pytesseract
    TESSERACT_AVAILABLE = True
except Exception:
    TESSERACT_AVAILABLE = False

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'sample_reports'

GEMINI_API_KEY = "AIzaSyBS0st-Ripk7WnpZNSAmQHvv5e5e2RDuYA"
genai.configure(api_key=GEMINI_API_KEY)

DEFAULT_REF_RANGES = {
    "Hemoglobin": {"unit": "g/dL", "low": 12.0, "high": 15.0},
    "WBC": {"unit": "/uL", "low": 4000, "high": 11000},
    "Platelets": {"unit": "/uL", "low": 150000, "high": 450000},
}

def ocr_from_image_file(file_stream):
    if not TESSERACT_AVAILABLE:
        raise RuntimeError("Install pillow + pytesseract + Tesseract engine for OCR")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(file_stream.read())
        tmp_path = tmp.name
    try:
        img = Image.open(tmp_path)
        img = img.convert("L")
        img = img.point(lambda x: 0 if x < 140 else 255)
        img = img.resize((img.width*2, img.height*2), Image.LANCZOS)
        text = pytesseract.image_to_string(img, config="--psm 6")
        return text
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

def extract_tests_local(text):
    text = text.replace("Hemglobin", "Hemoglobin").replace("Hbg", "Hemoglobin").replace("Hgh", "High")
    lines = []
    for part in re.split(r'[\n,;]+', text):
        part = part.strip()
        if part and re.search(r'\d', part):
            lines.append(part)
    confidence = min(0.95, 0.5 + len(lines)/10)
    return {"tests_raw": lines, "confidence": round(confidence,2)}

def normalize_tests(tests_raw):
    normalized = []
    for test in tests_raw:
        match = re.match(r"([A-Za-z ]+)\s*[:\-]?\s*([\d,\.]+)\s*([^\s\(,]*)\s*\(?([A-Za-z]*)\)?", test)
        if match:
            name = match.group(1).strip()
            val_str = match.group(2).replace(",", "")
            try:
                value = float(val_str)
            except:
                value = None
            unit = match.group(3) or ""
            status = (match.group(4).lower() if match.group(4) else "normal")
            ref = DEFAULT_REF_RANGES.get(name, {"low": None, "high": None, "unit": unit})
            if status == "normal" and value is not None and ref["low"] and ref["high"]:
                if value < ref["low"]:
                    status = "low"
                elif value > ref["high"]:
                    status = "high"
            normalized.append({
                "name": name,
                "value": value,
                "unit": ref["unit"],
                "status": status,
                "ref_range": {"low": ref["low"], "high": ref["high"]}
            })
    normalization_confidence = round(0.7 + len(normalized)/10, 2)
    return {"tests": normalized, "normalization_confidence": normalization_confidence}

def generate_summary_gemini(normalized_tests, temperature=0.7):
    tests_text = "\n".join([f"{t['name']} ({t['value']} {t['unit']}): {t['status']}" for t in normalized_tests])
    prompt = f"""
    You are a careful medical text processor.
    Input:
    {tests_text}

    Generate a JSON ONLY output with:
    {{
      "summary": "Patient-friendly summary (no diagnosis)",
      "explanations": ["Short explanation of each abnormal test"]
    }}
    """
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(
            [{"role": "user", "parts": [prompt]}],
            generation_config=genai.types.GenerationConfig(temperature=temperature, top_k=1)
        )
        text = response.text.strip()
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
        else:
            return {"summary": "Error generating summary", "explanations": []}
    except Exception as e:
        print("Gemini error:", e)
        return {"summary": "Error generating summary", "explanations": []}

def validate_no_hallucination(parsed_json, tests_raw):
    raw_join = " ".join(tests_raw).lower()
    normalized = parsed_json.get("tests", [])
    for t in normalized:
        name = t.get("name", "").lower()
        if name not in raw_join:
            return False, f"hallucinated test '{t.get('name')}' not present in input"
    return True, ""

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
                if uploaded_file_name.endswith((".png",".jpg",".jpeg",".tiff",".bmp")):
                    file.stream.seek(0)
                    extracted_text = ocr_from_image_file(file.stream)
                else:
                    file.stream.seek(0)
                    extracted_text = file.stream.read().decode("utf-8", errors="ignore")
            except Exception as e:
                extracted_text = file.stream.read().decode("utf-8", errors="ignore")
                print("OCR fallback:", e)

            if not extracted_text or not re.search(r'\d', extracted_text):
                final_output = {"status": "unprocessed", "reason": "No valid tests found"}
            else:
                extracted = extract_tests_local(extracted_text)
                normalized = normalize_tests(extracted["tests_raw"])
                summary = generate_summary_gemini(normalized["tests"])
                final_output = {
                    "tests_raw": extracted["tests_raw"],
                    "confidence": extracted["confidence"],
                    "tests": normalized["tests"],
                    "normalization_confidence": normalized["normalization_confidence"],
                    "summary": summary.get("summary", ""),
                    "explanations": summary.get("explanations", []),
                    "status": "ok"
                }
                ok, reason = validate_no_hallucination(final_output, extracted["tests_raw"])
                if not ok:
                    final_output = {"status": "unprocessed", "reason": reason}

    return render_template("index.html", final_output=final_output, filename=uploaded_file_name)

@app.route('/sample_reports/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
