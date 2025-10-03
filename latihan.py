import uuid
import re
from pathlib import Path
from llama_index.readers.file import PDFReader
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import google.generativeai as genai
from dotenv import load_dotenv
import os

#load semua yang ada pad .env
load_dotenv()

#membuat function untuk membaca file pdf

def baca_pdf(file_path):
    print("Sedalang dalam proses extract file pdf...")
    reader = PDFReader()
    documents = reader.load_data(file=Path(file_path))

    full_text = "\n".join([doc.text for doc in documents])
    panjang_halaman = len(documents)
    print(f"File pdf berhasil di extract dengan jumlah halaman: {panjang_halaman}")
    return full_text

def extract_web(urls):
   from llama_index.readers.web import TrafilaturaWebReader
   print("Sedang dalam proses extract data dari web...")
   reader = TrafilaturaWebReader()
   documents = reader.load_data(urls=urls)

   full_text = "\n".join([doc.text for doc in documents])
   panjang_halaman = len(documents)
   print(f"Data web berhasil di extract dengan jumlah halaman: {panjang_halaman}")
   return full_text

def clean_text(text):
    # Menghapus karakter non-ASCII
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    # Menghapus karakter khusus
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    # Menghapus spasi berlebih
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def chunking(text, chunk_size=500, overlap=100):
    print("sedang proses chunking...")
    chunk,start,text_length = [],0,len(text)
    while start < text_length:
        end = start + chunk_size
        chunk.append(text[start:end].strip())
        start += chunk_size - overlap
    print(f"Proses chunking selesai dengan jumlah chunk: {len(chunk)}")
    return chunk

def embeding(model_name="firqaaa/indo-sentence-bert-base"):
    print("sedang proses embeding...")
    model = SentenceTransformer(model_name)
    return model

def simpan_vektor(chunk, embedings,qdrant_url,qdrant_api_key,nama_collection,batch_size=50):
    print("sedang proses simpan vektor ke qrant...")
    client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
        timeout=30
    )
    client.recreate_collection(
        collection_name=nama_collection,
        vectors_config=VectorParams(size=len(embedings[0]), distance=Distance.COSINE)
    )
    print(f"Collection {nama_collection} berhasil dibuat di Qdrant")
    
    #Menyimpan vektor sesuai bacth size
    total=len(chunk)
    for i in range(0,total,batch_size):
        batch_chunks = chunk[i:i+batch_size]
        batch_embeddings = embedings[i:i+batch_size]
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding.tolist(),
                payload={"text": chunk}
            )
            for chunk, embedding in zip(batch_chunks, batch_embeddings)
        ]
        client.upsert(
            collection_name=nama_collection,
            points=points
        )
        print(f"Batch {i//batch_size + 1} dengan jumlah {len(points)} vektor berhasil disimpan.")

    print(f"Proses simpan vektor ke Qdrant selesai dengan total vektor: {total} kepada collection {nama_collection}")
    return client

def prprosesing_query(query):
    print("sedang proses prprosesing query...")
    query = clean_text(query)
    print("Proses prprosesing query selesai")
    return query

def search_vektor(query, client, nama_collection, embedings, top_k=3):
    print("sedang proses pencarian vektor di qrant...")
    from sklearn.metrics.pairwise import cosine_similarity

    #step 1 preprocessing query
    query = prprosesing_query(query)

    #step 2 embeding query
    query_embedding = embedings.encode([query])[0]

    #step 3 mencari vektor di qdrant
    results = client.search(
        nama_collection=nama_collection,
        query_vector=query_embedding.tolist(),
        limit=top_k,
        whith_vektors=True
    )

    #step 4 beranking sesuai cosine similarity
    if results:
        # Ambil vektor dari hasil pencarian
        vektor_qdrant = [hit.vector for hit in results]
        #hitung cosine similarity
        similarities = cosine_similarity([query_embedding], vektor_qdrant)[0]

        #merangking hasil pencarian berdasarkan similarity
        reranked_results = []
        for i, hit in enumerate(results):
                reranked_results.append({
                'text': hit.payload['text'],
                'score': similarities[i],
                'original_score': hit.score
            })
                
        #mengurutkan berdasarkan similarity
        reranked_results.sort(key=lambda x: x['similarity'], reverse=True)

        #ambil top k
        top_results = [item['text'] for item in reranked_results[:top_k]]
        print(f"Proses pencarian vektor di qdrant selesai dengan jumlah top k: {len(top_results)}")
        print(f"Similarity scores: {[round(item['score'], 3) for item in reranked_results[:top_k]]}")
        return top_results
    else:
        print("Tidak ada hasil yang ditemukan.")
        return []
    
    # 7. Prompt Construction
def construct_prompt(query, retrieved_chunks):
    print("\n[7] Membangun prompt...")
    context = "\n\n".join(retrieved_chunks)
    system_prompt = (
        "Anda adalah UINSAGA-AI, sebuah asisten virtual dan customer service yang ramah, sopan, dan beretika tinggi, khusus untuk melayani seluruh pertanyaan seputar UIN Salatiga.. "
        "Fokus utama Anda adalah membantu berbagai pihak, mulai dari calon mahasiswa (siswa SMA/luar negeri), mahasiswa aktif, orang tua/wali murid, dosen, Karyawan dan Staf Administrasi, Rektor dan Jajaran Manajemen dengan menjawab pertanyaan mereka secara akurat dan profesional. "
        "Gunakan Bahasa yang Relevan: Sesuaikan gaya bahasa Anda dengan audiens.. "
        "Untuk siswa SMA/calon mahasiswa Gunakan bahasa yang jelas, lugas, dan menarik. Berikan informasi yang memotivasi mereka untuk bergabung dengan UIN Salatiga dan jangan terlalu panjang teks nya simple saja."
        "Untuk mahasiswa aktif, dosen, dan karyawan atau staff Gunakan bahasa yang formal dan informatif, fokus pada detail teknis terkait prosedur atau data akademik,kemahasiswaan, dan keuangan."
        "Untuk orang tua atau wali Gunakan bahasa yang meyakinkan, sopan, dan mudah dipahami, memberikan rasa aman dan kepercayaan"
        "Sumber Informasi Seluruh jawaban Anda harus didasarkan pada konteks yang diberikan di bawah ini. Jangan pernah menggunakan pengetahuan umum atau informasi lain di luar konteks yang tersedia."
        "Jika jawaban ditemukan, berikan jawaban yang singkat, padat, dan langsung ke inti permasalahan."
        "Jika jawaban tidak ditemukan di dalam konteks, jawab dengan sopan, dan berikan permohonan maaf untuk informasi lebih lanjut ada di portal uin salatiga untuk informasi lebih lengkap."
        "Tindakan Tambahan: Jika pertanyaan mengarah pada data spesifik (misalnya, jadwal mata kuliah atau nama dosen) jika jawaban di temukan jawab dengan singkat padat jelas sesuai data yang ada jika tidak ada jawaban sarankan pengguna untuk mengecek aplikasi smart mhs atau SIAKAD UIN SALATIGA."
        "Anda diharapkan memiliki pemahaman mendalam tentang terminologi akademis, keuangan, dan kemahasiswaan yang berlaku di UIN Salatiga."
        "setiap anda menjawab pertanyaan jawablah dengan simple dan singkat jangan terlalu panjang. teks nya langsung kepada inti nya dan ini berlaku pada setiap pertanyaan user yang ada "
    )
    user_prompt = f"""
    Pertanyaan user:
    {query}

    Konteks dokumen:
    {context}

    Jawablah dengan gaya seorang CS dan gunakan pengetahuan tambahan dari internet jika perlu dan jadilah customer service international yang di gunakan perusahaan seperti apple nvidia google microsoft dan openAi jika ingin memberikan salam hangat seperti"Selamat malam /siang/pagi gunakan salam yang sesuai dengan waktu user bertanya"
    """
    print("Prompt berhasil dibuat.")
    return system_prompt, user_prompt

