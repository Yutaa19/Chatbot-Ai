from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from llama_index.readers.web import TrafilaturaWebReader

# 1.1 Ekstraksi dari Web
def extract_text_from_web_async(urls):
    """
    Mengekstrak teks dari daftar URL secara async menggunakan TrafilaturaWebReader.
    """
    print(f"\n[1.1] Ekstraksi dari {len(urls)} URL (async)...")
    
    reader = TrafilaturaWebReader()
    documents = reader.load_data(urls=urls)  # Mendukung async secara internal
    
    full_text = "\n".join([doc.text for doc in documents])
    
    print("Ekstraksi web selesai.")
    return full_text

# Fungsi untuk membuat PDF dari teks yang diberikan
def create_pdf_from_text(text, filename="sejarah_uin.pdf"):
    """
    Membuat file PDF dari string teks yang diberikan.
    Setiap paragraf dipisahkan oleh baris kosong.
    """
    doc = SimpleDocTemplate(filename, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    # Pecah teks menjadi paragraf berdasarkan baris kosong
    paragraphs = text.split("\n\n")
    
    for para in paragraphs:
        if para.strip():  # Hanya tambahkan jika tidak kosong
            story.append(Paragraph(para.strip(), styles["Normal"]))
            story.append(Spacer(1, 12))  # Jarak antar paragraf

    doc.build(story)
    print(f"✅ PDF '{filename}' berhasil dibuat dengan konten dari web.")

# MAIN EXECUTION
if __name__ == "__main__":
    # Daftar URL yang ingin di-scrape
    urls_list = [
        "https://www.uinsalatiga.ac.id/tentang-uin-salatiga/",
        "https://www.uinsalatiga.ac.id/visi-dan-misi/",
        "https://www.uinsalatiga.ac.id/logo/",
        "https://www.uinsalatiga.ac.id/bendera/",
        "https://www.uinsalatiga.ac.id/mars-dan-hymne/",
        "https://www.uinsalatiga.ac.id/pimpinan/",
        "https://www.uinsalatiga.ac.id/struktur-organisasi/",
        "https://www.uinsalatiga.ac.id/tenaga-pendidik/",
        "https://www.uinsalatiga.ac.id/tenaga-kependidikan/",# ← Penting! Tapi mungkin gagal di Trafilatura
    ]

    # 1. Ekstrak teks dari web
    web_text = extract_text_from_web_async(urls_list)

    # 2. Simpan teks hasil ekstraksi ke PDF
    create_pdf_from_text(web_text, "sejarah_uin.pdf")

    print("\n📄 Proses selesai. File PDF siap digunakan.")
