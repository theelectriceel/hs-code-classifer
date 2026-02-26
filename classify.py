# api/classify.py
# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import time
import logging
from openai import OpenAI
from pinecone import Pinecone

# -----------------------------------------------------
# Logging Setup
# -----------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# =====================================================
# 🔑 API KEYS (ROTATION SYSTEM)
# =====================================================
API_KEYS = [
    "5174fe1283b2cccd4060271c24e7f64285924ff7d873438f90b781a1c4fe6c3b",
]
current_key_index = 0

def create_client():
    logger.info(f"Creating OpenAI client with key index {current_key_index}")
    return OpenAI(
        api_key=API_KEYS[current_key_index],
        base_url="https://ml-openai.cloudcix.com"
    )

client = create_client()

# =====================================================
# 🌲 Pinecone Init
# =====================================================
logger.info("Initializing Pinecone...")
pc = Pinecone(api_key='pcsk_GYubY_MbqWRXd1hqFTyxKq6AtJd5KhjzMQ3bgRpSgTKjihZAuR4RcKCA1AtGTkdQg6yV1')
index = pc.Index('chapter30-codes')
logger.info("Pinecone index loaded: chapter30-codes")

# =====================================================
# 🔍 Main classify function
# =====================================================
def classify(description: str, model="UCCIX-Mistral-24B"):
    logger.info(f"Classifying description: {description}")

    global client, current_key_index

    # Step 1: Embedding
    logger.info("Generating embedding...")
    embedding = get_embedding(description)
    logger.info(f"Embedding length: {len(embedding)}")

    # Step 2: Pinecone query
    logger.info("Querying Pinecone...")
    results = index.query(
        vector=embedding,
        top_k=5,
        include_metadata=True
    )
    logger.info(f"Pinecone raw result: {results}")

    # Step 3: Build top-5 list
    top_5_candidates = []
    for i, match in enumerate(results['matches'], 1):
        code = match['id'].replace('.', '')[0:6]
        desc_text = match['metadata'].get('description', 'No description available')
        top_5_candidates.append(f"{i}. {code}: {desc_text}")

    top_5_text = "\n".join(top_5_candidates)
    logger.info(f"Top 5 candidates:\n{top_5_text}")

    # Step 4: LLM classification
    prompt = f"""
You are given the following product description:
\"\"\"{description}\"\"\"

Top 5 candidate NOMENCLATURE_CODEs:
{top_5_text}

Respond strictly in this format:
CODE: <chosen_code>
JUSTIFICATION: <short explanation>
"""

    while True:
        try:
            logger.info("Calling LLM...")
            chat_completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=False
            )

            llm_output = chat_completion.choices[0].message.content
            logger.info(f"LLM output: {llm_output}")

            match = re.search(r"CODE:\s*(\d+)", llm_output)
            predicted_code = match.group(1) if match else None

            return {"code": predicted_code, "justification": llm_output}

        except Exception as e:
            logger.error(f"Error with key {current_key_index+1}: {e}")
            current_key_index += 1
            if current_key_index >= len(API_KEYS):
                logger.critical("All API keys failed.")
                raise SystemExit("All API keys failed.")
            client = create_client()
            time.sleep(1)

# =====================================================
# 🔹 Embedding function
# =====================================================
def get_embedding(description: str):
    logger.info("Requesting embedding from OpenAI...")
    response = client.embeddings.create(
        model="cix_chunk_encoder",
        input=[description],
        encoding_format="float"
    )
    logger.info("Embedding received.")
    return response.data[0].embedding

# =====================================================
# CORS
# =====================================================
CORS(app,
     resources={r"/*": {"origins": "https://easyshipai.vercel.app"}},
     allow_headers=["Content-Type"],
     methods=["GET", "POST", "OPTIONS"])

@app.route("/classify", methods=["OPTIONS"])
def classify_options():
    logger.info("OPTIONS preflight received.")
    return jsonify({"status": "ok"}), 200

# =====================================================
# Health check
# =====================================================
@app.route("/", methods=["GET"])
def home():
    logger.info("Health check hit.")
    return jsonify({"message": "Flask serverless API is alive!"})

# =====================================================
# POST /classify
# =====================================================
@app.route("/classify", methods=["POST"])
def classify_endpoint():
    try:
        logger.info("POST /classify hit.")
        logger.info(f"Incoming JSON: {request.json}")

        data = request.json
        description = data.get("description", "").strip()

        if not description:
            logger.warning("Missing description in request.")
            return jsonify({"error": "Missing 'description' in request"}), 400

        result = classify(description)
        logger.info(f"Classification result: {result}")
        return jsonify(result)

    except Exception as e:
        logger.error(f"Server error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    logger.info("Starting Flask server...")
    app.run(debug=True, host="0.0.0.0", port=5000)