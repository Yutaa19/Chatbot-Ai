import os
from flask import Flask, render_template, request, jsonify
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
    ask_openrouter
)


app = Flask(__name__)

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
QDRANT_URL = os.getenv('QDRANT_URL', "https://fef232a5-ad33-47e2-b0ce-a374a9a13f15.europe-west3-0.gcp.cloud.qdrant.io:6333")
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY', "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.SMKtT2_FbPb5DE1GlDP7Q9xC6MA0w77rWZrZHbxEK4s")
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY', "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")
OPENROUTER_MODEL = os.getenv('OPENROUTER_MODEL', "x-ai/grok-4-fast:free")
COLLECTION_NAME = os.getenv('COLLECTION_NAME', "my_pdf_collection")

# --- Inisialisasi Global ---
client = None
embedder = None

# Initialize RAG system components
print("Initializing RAG system...")
pdf_text = extract_text_from_pdf_llamaindex(PDF_FILE)
web_text = extract_text_from_web_async(urls_list)
# Logging debug
print(f"[DEBUG] Panjang teks PDF: {len(pdf_text)} karakter")
print(f"[DEBUG] Panjang teks Web: {len(web_text)} karakter")
    
    # Gabungkan jika web_text tidak kosong
if not web_text.strip():
        print("⚠️  Peringatan: Hasil ekstraksi web kosong. Hanya menggunakan PDF.")
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
        system_prompt, user_prompt = construct_prompt(user_query, retrieved)

        # 8. Tanya LLM via OpenRouter
        answer = ask_openrouter(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            api_key=OPENROUTER_API_KEY,
            model_name=OPENROUTER_MODEL
        )

        # Kembalikan jawaban + konteks (untuk debugging/frontend)
        return jsonify({
            'answer': answer,
            'context_used': retrieved  # Bisa dihilangkan jika tidak diperlukan di frontend
        })

    except Exception as e:
        # Log error untuk debugging
        print(f"❌ Error saat memproses permintaan: {str(e)}")
        import traceback
        traceback.print_exc()

        # Beri respons user-friendly
        return jsonify({
            'error': 'Maaf, terjadi kesalahan internal. Tim kami sedang memperbaikinya.'
        }), 500


if __name__ == '__main__':
    # Jalankan Flask app
    # Untuk production: ganti debug=False dan gunakan WSGI server (Gunicorn/Waitress)
    app.run(debug=True, host='0.0.0.0', port=5000)

