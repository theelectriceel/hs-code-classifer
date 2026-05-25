# api/classify.py
# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import time
from openai import OpenAI
from pinecone import Pinecone
import os
from dotenv import load_dotenv
app = Flask(__name__)

load_dotenv()

current_key_index = 0
API_KEYS = os.getenv('OPEN_AI_KEY')
def create_client():
    return OpenAI(
        api_key=API_KEYS[current_key_index],
        base_url="https://ml-openai.cloudcix.com"
    )

client = create_client()

# =====================================================
# 🌲 Pinecone Init
# =====================================================

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index('chapter30-codes')

# =====================================================
# 🔍 Main classify function
# =====================================================
def classify(description: str, model="UCCIX-Mistral-24B"):

    global client, current_key_index

    # Step 1: Embedding

    embedding = get_embedding(description)
   

    # Step 2: Pinecone query
    results = index.query(
        vector=embedding,
        top_k=5,
        include_metadata=True
    )
   

    # Step 3: Build top-5 list
    top_5_candidates = []
    for i, match in enumerate(results['matches'], 1):
        code = match['id'].replace('.', '')[0:6]
        desc_text = match['metadata'].get('description', 'No description available')
        top_5_candidates.append(f"{i}. {code}: {desc_text}")

    top_5_text = "\n".join(top_5_candidates)

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
    
            chat_completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=False
            )

            llm_output = chat_completion.choices[0].message.content
            

            match = re.search(r"CODE:\s*(\d+)", llm_output)
            predicted_code = match.group(1) if match else None

            return {"code": predicted_code, "justification": llm_output}

        except Exception as e:
            current_key_index += 1
            if current_key_index >= len(API_KEYS):
                raise SystemExit("All API keys failed.")
            client = create_client()
            time.sleep(1)

# =====================================================
# 🔹 Embedding function
# =====================================================
def get_embedding(description: str):
    response = client.embeddings.create(
        model="cix_chunk_encoder",
        input=[description],
        encoding_format="float"
    )
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
    return jsonify({"status": "ok"}), 200

# =====================================================
# Health check
# =====================================================
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Flask serverless API is alive!"})

# =====================================================
# POST /classify
# =====================================================
@app.route("/classify", methods=["POST"])
def classify_endpoint():
    try:
       
        data = request.json
        description = data.get("description", "").strip()

        if not description:
           
            return jsonify({"error": "Missing 'description' in request"}), 400

        result = classify(description)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)