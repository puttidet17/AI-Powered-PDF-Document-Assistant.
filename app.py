import os
import sqlite3
import json
import math
import time
from flask import Flask, request, jsonify, render_template
from pypdf import PdfReader
from google import genai
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize Gemini Client (Requires GEMINI_API_KEY in terminal environment)
client = client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# Database Connection
def get_db_connection():
    conn = sqlite3.connect('project.db')
    conn.row_factory = sqlite3.Row
    return conn

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text

# Split text into chunks with overlap
def chunk_text(text, chunk_size=500, overlap=50):
    chunks =[]
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# Generate Vector Embeddings using Gemini
def get_embedding(text):
    max_retries = 3
    retry_delay = 2  # wait 2 seconds before retrying

    for attempt in range(max_retries):
        try:
            response = client.models.embed_content(
                model='gemini-embedding-001',
                contents=text
            )
            return response.embeddings[0].values
        except Exception as e:
            # Check if the error is due to high demand (503 Service Unavailable)
            if "503" in str(e) and attempt < max_retries - 1:
                print(f"⚠️ Embedding API busy (503). Retrying attempt {attempt + 2}/{max_retries} in {retry_delay}s...")
                time.sleep(retry_delay)
                continue
            # If it's a different error or we ran out of retries, raise the exception
            raise e

# Calculate Cosine Similarity
def cosine_similarity(vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1,vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    return dot_product / (magnitude1 * magnitude2)

# Process and save to SQLite
def process_and_store_chunks(chunks):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM document_chunks")

    for chunk in chunks:
        vector = get_embedding(chunk)
        vector_json = json.dumps(vector)
        cursor.execute(
            "INSERT INTO document_chunks (text_content, embedding) VALUES (?, ?)",
            (chunk, vector_json)
        )

    conn.commit()
    conn.close()

# Retrieve most relevant chunks from SQLite
def retrieve_relevant_context(user_question, top_k=3):
    question_vector = get_embedding(user_question)
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT text_content, embedding FROM document_chunks")
    rows = cursor.fetchall()
    conn.close()

    scored_chunks = []
    for row in rows:
        chunk_text = row['text_content']
        chunk_vector = json.loads(row['embedding'])

        score = cosine_similarity(question_vector, chunk_vector)
        scored_chunks.append((chunk_text, score))

    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    top_chunks = [item[0] for item in scored_chunks[:top_k]]
    return "\n".join(top_chunks)

# ---------------------------- Web Routes -----------------------------------
# Render the frontend template instead of raw text
@app.route('/')
def home():
    return render_template('index.html')

# Handle PDF upload and text processing
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part provided."}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file."}), 400

    if file and file.filename.endswith('.pdf'):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        raw_text = extract_text_from_pdf(filepath)
        chunks = chunk_text(raw_text)
        process_and_store_chunks(chunks)

        return jsonify({
            "message": "PDF processed and chunked successfully!",
            "total_chunks": len(chunks)
        }), 200
    return jsonify({"error": "Only .pdf files are allowed."}), 400

# Handle Chat Requests and Call Gemini
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')

    if not user_message:
        return jsonify({"error": "Message cannot be empty."}), 400

    context = retrieve_relevant_context(user_message, top_k=3)

    prompt = f"""
    You are a professional AI Assistant. Your task is to answer the user's question based strictly on the provided PDF context below.

    Rules:
    1. If the context does not contain enough information to answer the question, politely reply with: "I'm sorry, but I cannot find the answer to this question in the provided document." Do not make up answers.
    2. Maintain an informative, clear, and polite tone.

    [PDF Context]
    {context}

    [User Question]
    {user_message}

    [Answer]
    """

    max_retries = 3
    retry_delay = 2

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            return jsonify({"response": response.text}), 200
        except Exception as e:
            # Handle temporary high demand errors by retrying
            if "503" in str(e) and attempt < max_retries - 1:
                print(f"⚠️ Chat API busy (503). Retrying attempt {attempt + 2}/{max_retries} in {retry_delay}s...")
                time.sleep(retry_delay)
                continue
            # Return the error response if max retries reached or it's a critical error
            return jsonify({"error": f"Gemini API Error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
