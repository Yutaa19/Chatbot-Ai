import os
from flask import Flask, render_template, request, jsonify, session
import redis
import json
import time
from dotenv import load_dotenv
import uuid  # Pastikan diimpor jika belum

# Load environment variables
load_dotenv()

# Import semua fungsi dari Main.py
from Main import (
    extract_text_from_pdf_llamaindex,
    extract_text_from_web_async,
    clean_text,
    chunk_text,
    get_embedder,
    store_to_qdrant,
    search_qdrant,
    construct_prompt,
    ask_gemini
)


app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY')

# Koneksi Redis
REDIS_URL = os.getenv('REDIS_URL')
redis_client = redis.from_url(REDIS_URL, decode_responses=True)
print(f"Connected to Redis at {REDIS_URL}")

# Initialize global variables
PDF_FILE = "sejarah_uin.pdf"
# Daftar URL (bisa juga dipindah ke .env atau config file jika perlu)
urls_list = [
    "https://www.uinsalatiga.ac.id/",
    "https://www.uinsalatiga.ac.id/tentang-uin-salatiga/",
    "https://www.uinsalatiga.ac.id/kehidupan-kampus/",
    "https://www.uinsalatiga.ac.id/visi-dan-misi/",
    "https://www.uinsalatiga.ac.id/logo/",
    "https://www.uinsalatiga.ac.id/bendera/",
    "https://www.uinsalatiga.ac.id/mars-dan-hymne/",
    "https://www.uinsalatiga.ac.id/akreditasi/",
    "https://www.uinsalatiga.ac.id/akreditasi-program-studi/",
    "https://www.uinsalatiga.ac.id/pimpinan/",
    "https://www.uinsalatiga.ac.id/struktur-organisasi/",
    "https://www.uinsalatiga.ac.id/tenaga-pendidik/",
    "https://www.uinsalatiga.ac.id/tenaga-kependidikan/",
]
QDRANT_URL = os.getenv('QDRANT_URL')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
<<<<<<< HEAD
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL")
COLLECTION_NAME = os.getenv('COLLECTION_NAME', "my_pdf_collection")
=======
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_MODEL = os.getenv('GEMINI_MODEL')
COLLECTION_NAME = os.getenv('COLLECTION_NAME')
>>>>>>> 8e932c5b4b3795d5b53d45a5c8ff763d3a6d239d

# --- Inisialisasi Global ---
client = None
embedder = None

# === FUNGSI REDIS ===
def get_history(user_id, limit=5):
    key = f"chat:{user_id}"
    raw = redis_client.lrange(key, 0, limit - 1)
    return [json.loads(item) for item in raw]

def save_history(user_id, user_msg, bot_msg):
    key = f"chat:{user_id}"
    item = json.dumps({'user': user_msg, 'ai': bot_msg, 'ts': time.time()})
    redis_client.lpush(key, item)
    redis_client.ltrim(key, 0, 9)  # Simpan max 10 percakapan
    redis_client.expire(key, 1800)  # Hapus setelah 30 menit

# Initialize RAG system components
print("Initializing RAG system...")
pdf_text = extract_text_from_pdf_llamaindex(PDF_FILE)
web_text = extract_text_from_web_async(urls_list)
# Logging debug
print(f"[DEBUG] Panjang teks PDF: {len(pdf_text)} karakter")
print(f"[DEBUG] Panjang teks Web: {len(web_text)} karakter")
    
    # Gabungkan jika web_text tidak kosong
if not web_text.strip():
        print("Peringatan: Hasil ekstraksi web kosong. Hanya menggunakan PDF.")
        combined_text = pdf_text
else:
        combined_text = pdf_text + "\n\n=== TEKS DARI WEB ===\n\n" + web_text

print(f"[DEBUG] Panjang teks Gabungan: {len(combined_text)} karakter")
cleaned_text = clean_text(combined_text)
chunks = chunk_text(cleaned_text, chunk_size=500, overlap=100)
embedder = get_embedder()
embeddings = embedder.encode(chunks, convert_to_tensor=False)
client = store_to_qdrant(chunks, embeddings, QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME)
print("RAG system initialized successfully!")

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/ask', methods=['POST'])
def ask():
    global client, embedder
    try:
        # Pastikan sistem RAG sudah diinisialisasi
        if client is None or embedder is None:
            return jsonify({
                'error': 'RAG system belum siap. Silakan coba lagi beberapa saat atau restart server.'
            }), 503

        # Ambil query dari request
        data = request.get_json()
        user_query = data.get('query', '').strip()

        if not user_query:
            return jsonify({'error': 'Pertanyaan tidak boleh kosong.'}), 400

        print(f"\n[USER QUERY]: {user_query}")

        # === 1. Dapatkan atau buat user_id ===
        if 'user_id' not in session:
            session['user_id'] = str(uuid.uuid4())
        user_id = session['user_id']

        # === 2. Ambil riwayat percakapan dari Redis ===
        conversation_history = get_history(user_id, limit=5)
        history_text = ""
        for turn in conversation_history:
            history_text += f"User: {turn['user']}\nAI: {turn['ai']}\n\n"

        # 6. Similarity Search — Cari chunk relevan dari Qdrant
        retrieved = search_qdrant(
            query=user_query,
            client=client,
            collection_name=COLLECTION_NAME,
            embedder=embedder,
            top_k=3
        )

        # Logging untuk debugging (opsional di production)
        print("\n=== RETRIEVED CHUNKS ===")
        for i, chunk in enumerate(retrieved, 1):
            preview = chunk[:150].replace('\n', ' ') + "..." if len(chunk) > 150 else chunk
            print(f"[{i}] {preview}")

        # 7. Construct Prompt — Siapkan prompt untuk LLM
       # === Construct Prompt — SERTAKAN RIWAYAT! ===
        system_prompt, user_prompt = construct_prompt(
        query=user_query,
        retrieved_chunks=retrieved,
        conversation_history=history_text.strip()  # Kirim riwayat ke Main.py
)

        # 8. Tanya LLM via OpenRouter
        answer = ask_gemini(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            api_key=GEMINI_API_KEY,
            model_name=GEMINI_MODEL
        )

        # === 6. Simpan ke Redis ===
        save_history(user_id, user_query, answer)

        # Kembalikan jawaban + konteks (untuk debugging/frontend)
        return jsonify({
            'answer': answer,
            'context_used': retrieved  # Bisa dihilangkan jika tidak diperlukan di frontend
        })

    except Exception as e:
        # Log error untuk debugging
        print(f"Error saat memproses permintaan: {str(e)}")
        import traceback
        traceback.print_exc()

        # Beri respons user-friendly
        return jsonify({
            'error': 'Maaf, terjadi pemeliharaan chatbot. Tim kami sedang memperbaikinya.'
        }), 500


if __name__ == '__main__':
    # Jalankan Flask app
    # Untuk production: ganti debug=False dan gunakan WSGI server (Gunicorn/Waitress)
    app.run(debug=True, host='0.0.0.0', port=5000)

