import os
import tempfile

import requests
from flask import Flask, request, jsonify
import io
import json
from google.cloud import vision
from langchain_openai import ChatOpenAI
from google.oauth2 import service_account
from dotenv import load_dotenv

from concurrent.futures import ThreadPoolExecutor

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

def get_credentials_from_env_variables():
    creds_list = [
        "TYPE",
        "PROJECT_ID",
        "PRIVATE_KEY_ID",
        "PRIVATE_KEY",
        "CLIENT_EMAIL",
        "CLIENT_ID",
        "AUTH_URI",
        "TOKEN_URI",
        "AUTH_PROVIDER_X509_CERT_URL",
        "CLIENT_X509_CERT_URL",
        "UNIVERSE_DOMAIN",
    ]

    creds_dict = {}
    for cred in creds_list:
        loaded_cred = os.getenv(cred)
        creds_dict[cred.lower()] = loaded_cred.encode().decode("unicode_escape")

    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_file:
        temp_filename = temp_file.name
        json.dump(creds_dict, temp_file)

    # Load credentials from the temporary file
    credentials = service_account.Credentials.from_service_account_file(temp_filename)

    # Remove the temporary file
    os.remove(temp_filename)

    return credentials

# Initialize Vision API Client with credentials from environment variables
credentials = get_credentials_from_env_variables()
client = vision.ImageAnnotatorClient(credentials=credentials)

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

# Define Categories
categories = {
    1: {
        "name": "Cytotoxic and Cytostatic Drugs",
        "keywords": ["chemotherapy", "oncology"],
        "atc_prefixes": ["L"],
    },
    2: {
        "name": "Inhalers",
        "keywords": ["inhaler", "inhalation", "metered dose"],
        "atc_prefixes": ["R03"],
    },
    3: {
        "name": "Injectables or Syringes",
        "keywords": ["i.v.", "i.m.", "injectable", "injection", "ampoule", "vial", "solvent"],
        "priority": True,
    },
    4: {
        "name": "Insulin",
        "keywords": ["insulin", "diabetes"],
        "atc_prefixes": ["A10A"],
    },
    5: {
        "name": "Common Medicines",
        "keywords": [],
        "atc_prefixes": [],
    },
    6: {
        "name": "Supplements",
        "keywords": ["vitamin", "mineral", "supplement", "dietary aid"],
    },
    7: {
        "name": "Psycholeptics",
        "keywords": ["antipsychotic", "anxiolytic", "schizophrenia", "bipolar", "diazepam"],
        "atc_prefixes": ["N05"],
    },
}

category_descriptions = "\n".join(
    f"{i}. {cat['name']}: Keywords: {', '.join(cat.get('keywords', [])) or 'None'}; "
    f"ATC Prefixes: {', '.join(cat.get('atc_prefixes', [])) or 'None'}."
    for i, cat in categories.items()
)

def get_model_task():
    model_task_file = 'model-task.txt'

    if os.path.exists(model_task_file):
        with open(model_task_file, 'r') as file:
            model_task = file.read()
    else:
        model_task = os.getenv('MODEL_TASK')

    return model_task

# OCR Function
def extract_text_from_image(image_path):
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    print(texts)
    if response.error.message:
        raise Exception(f'Error with Vision API: {response.error.message}')
    if texts:
        return texts[0].description
    return "No text detected."

def process_query(ocr_text):
    if ocr_text == "No text detected.":
        return {"error": "No valid drug identified."}

    model_task = get_model_task()
    query = model_task.format(category_descriptions=category_descriptions, ocr_text=ocr_text)

    identified_drug = llm.invoke(query).content
    try:
        identified_drug = json.loads(identified_drug)
    except json.JSONDecodeError as e:
        print("Error parsing LLM response:", e)
        return {"error": "Failed to parse response from the LLM"}

    if not identified_drug.get("name", None):
        return {"error": "No valid drug identified."}
    print(identified_drug)

    result = {
        "name": identified_drug.get("name", None),
        "concentration": identified_drug.get("concentration", None),
        "category": identified_drug.get("category"),
        "ocr_result": ocr_text
    }
    return result

def send_recycle_request(recycle):
    response = requests.post('https://drugbin-backend-nestjs-production.up.railway.app/recycle', json=recycle)
    print(response.json())

@app.route('/process', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    image_file = request.files['image']
    image_path = f"/tmp/{image_file.filename}"
    image_file.save(image_path)
    try:
        ocr_text = extract_text_from_image(image_path)
        result = process_query(ocr_text)
        result["quantity"] = 1
        result["pack"] = "box"

        recycle = {
            "firstName": "Anonim",
            "hospitalId": 161,
            "lastName": "Anonim",
            "email": None,
            "drugList": [result],
            "address": None,
            "cnp": None
        }

        # Use ThreadPoolExecutor to send the request in parallel
        executor = ThreadPoolExecutor(max_workers=1)
        executor.submit(send_recycle_request, recycle)

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def index():
    return "Health Check OK",  200

# Run App
if __name__ == '__main__':
    app.run(debug=True)
