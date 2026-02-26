# api/classify.py
    # app.py
from flask import Flask, request, jsonify
from flask_cors import CORS# 
import re
import time
from openai import OpenAI
from pinecone import Pinecone


app = Flask(__name__)

# =====================================================
# 🔑 API KEYS (ROTATION SYSTEM)
# =====================================================
API_KEYS = [
    "5174fe1283b2cccd4060271c24e7f64285924ff7d873438f90b781a1c4fe6c3b",
]
current_key_index = 0

def create_client():
    return OpenAI(
        api_key=API_KEYS[current_key_index],
        base_url="https://ml-openai.cloudcix.com"
    )

client = create_client()

# =====================================================
# 🌲 Pinecone Init
# =====================================================
pc = Pinecone(api_key='pcsk_GYubY_MbqWRXd1hqFTyxKq6AtJd5KhjzMQ3bgRpSgTKjihZAuR4RcKCA1AtGTkdQg6yV1')
index = pc.Index('chapter30-codes')

# =====================================================
# 🔍 Main classify function
# =====================================================
def classify(description: str, model="UCCIX-Mistral-24B"):
    """
    Input: description (string) of product
    Output: dict with 'code' and 'justification'
    """
    global client, current_key_index

    # For simplicity, we create a fake row dict with 'Embeddings' key
    # In production, you would convert description to embeddings first
    row = {
        "DESCRIPTION_OF_GOODS": description,
        "Embeddings": get_embedding(description)  # Implement this function
    }

    # Step 1: Retrieve Top-5 candidates from Pinecone
    results = index.query(
        vector=row['Embeddings'],
        top_k=5,
        include_metadata=True
    )

    top_5_candidates = []
    for i, match in enumerate(results['matches'], 1):
        code = match['id'].replace('.', '')[0:6]
        desc_text = match['metadata'].get('description', 'No description available')
        top_5_candidates.append(f"{i}. {code}: {desc_text}")

    top_5_text = "\n".join(top_5_candidates)

    prompt = f"""
You are given the following product description:
\"\"\"{description}\"\"\"

Top 5 candidate NOMENCLATURE_CODEs:
{top_5_text}

Respond strictly in this format:
CODE: <chosen_code>
JUSTIFICATION: <short explanation>
"""

    # Retry loop for key rotation
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
            print(f"⚠️ Error with key {current_key_index+1}: {e}")
            current_key_index += 1
            if current_key_index >= len(API_KEYS):
                raise SystemExit("All API keys failed.")
            client = create_client()
            time.sleep(1)

# =====================================================
# 🔹 Dummy embedding function (replace with your own)
# =====================================================

# -------------------------------
def get_embedding(description: str):
    """
    Input: product description (string)
    Output: embedding vector (list of floats)
    """
    response = client.embeddings.create(
        model="cix_chunk_encoder",  # same as your batch encoder
        input=[description],        # wrap in a list for API
        encoding_format="float"
    )
    
    # There is only one input, so take the first embedding
    embedding_vector = response.data[0].embedding
    return embedding_vector


# Allow calls from your frontend
CORS(app,
     resources={r"/*": {"origins": "https://easyshipai.vercel.app"}},
     supports_credentials=True,
     allow_headers=["Content-Type"],
     methods=["GET", "POST", "OPTIONS"])

# Health check endpoint
@app.route("/", methods=["GET"])
def home():
        return jsonify({"message": "Flask serverless API is alive!"})

# Classification endpoint
@app.route("/classify", methods=["POST"])
def classify_endpoint():
        try:
            data = request.json
            description = data.get("description", "").strip()

            if not description:
                return jsonify({"error": "Missing 'description' in request"}), 400

            result = classify(description)  # Call your function
            return jsonify(result)

        except Exception as e:
            return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

