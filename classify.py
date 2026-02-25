from flask import Flask, request, jsonify
from flask_cors import CORS
from pinecone import Pinecone
from openai import OpenAI
import re
import time

app = Flask(__name__)
CORS(app)

# API Keys
API_KEYS = ["2d2fcdd026ccdbd21b499af18d130c67ec61481ca6e3c96f08384096580cf3bc"]
current_key_index = 0

def create_client():
    return OpenAI(
        api_key=API_KEYS[current_key_index],
        base_url="https://ml-openai.cloudcix.com"
    )

client = create_client()

# Pinecone Init
pc = Pinecone(api_key="pcsk_GYubY_MbqWRXd1hqFTyxKq6AtJd5KhjzMQ3bgRpSgTKjihZAuR4RcKCA1AtGTkdQg6yV1")
index = pc.Index("chapter30-codes")

# Inference function
def infer_single_row(description, query_vector, model="UCCIX-Mistral-24B"):
    global client, current_key_index

    results = index.query(vector=query_vector, top_k=5, include_metadata=True)

    top_5_candidates = []
    for i, match in enumerate(results['matches'], 1):
        code = match['id'].replace('.', '')[0:6]
        desc = match['metadata'].get('description', 'No description available')
        top_5_candidates.append(f"{i}. {code}: {desc}")

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

    # Simple retry
    max_retries = len(API_KEYS)
    retries = 0
    while retries < max_retries:
        try:
            chat_completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=False
            )

            llm_output = chat_completion.choices[0].message.content
            match = re.search(r"CODE:\s*(\d+)", llm_output)
            predicted_code = match.group(1) if match else None

            return predicted_code, llm_output, top_5_text

        except Exception as e:
            print(f"⚠️ Error with key {current_key_index+1}: {e}")
            current_key_index += 1
            if current_key_index >= len(API_KEYS):
                return None, "All API keys failed.", top_5_text
            client = create_client()
            retries += 1
            time.sleep(1)

# Routes
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Flask serverless API is alive!"})

@app.route("/classify", methods=["POST"])
def classify():
    try:
        data = request.json
        description = data.get("description", "")
        embeddings = data.get("embeddings", None)

        if not description or embeddings is None:
            return jsonify({"error": "Missing 'description' or 'embeddings'"}), 400

        predicted_code, llm_output, top5 = infer_single_row(description, embeddings)

        return jsonify({"code": predicted_code, "justification": llm_output, "top5": top5})

    except Exception as e:
        return jsonify({"error": str(e)}), 500