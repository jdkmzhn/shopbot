import os
import datetime
import threading
import openai
import numpy as np
import faiss
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_admin import Admin, AdminIndexView, expose
from crawl import crawl_website

# Definiere den absoluten Pfad für die knowledge.txt
KNOWLEDGE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge.txt")

# Flask-App und Konfiguration
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///shopbot.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SECRET_KEY"] = "mein-geheimer-schluessel"
app.config["UPLOAD_FOLDER"] = "uploads"
db = SQLAlchemy(app)

# ✅ NEUE ROUTE HINZUFÜGEN
@app.route("/")
def home():
    return "Hello, Flask is running!"

# Falls du weitere Routen hast, sollten die hier sein

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6000, debug=True)
# Sicherstellen, dass der Upload-Ordner existiert
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ... Rest deines Codes ...


# --------------------------
# Datenbankmodelle
# --------------------------
class CrawledPage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    url = db.Column(db.String(500), nullable=False)
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)

class CrawledWebsite(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    url = db.Column(db.String(500), nullable=False, unique=True)

class UploadedFile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)

class APIKey(db.Model):
    __tablename__ = "api_key"
    __table_args__ = {"extend_existing": True}
    id = db.Column(db.Integer, primary_key=True)
    key = db.Column(db.String(100), nullable=False)

class BotPrompt(db.Model):
    __tablename__ = "bot_prompt"
    __table_args__ = {"extend_existing": True}
    id = db.Column(db.Integer, primary_key=True)
    prompt = db.Column(db.Text, nullable=False)

class GreetingMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.String(500), nullable=False)

with app.app_context():
    db.create_all()

# --------------------------
# Globale Variablen für FAISS-Index-Caching
# --------------------------
faiss_index = None
faiss_chunk_list = None

# --------------------------
# Hilfsfunktionen für Retrieval und Index-Aufbau
# --------------------------
def get_embedding(text):
    """
    Ruft mithilfe des text-embedding-ada-002 Modells (openai==0.28.0) das Embedding eines Textes ab.
    """
    try:
        response = openai.Embedding.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response["data"][0]["embedding"]
    except Exception as e:
        print("Fehler beim Abrufen des Embeddings:", e)
        return None

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def chunk_text(text, max_length=500):
    """
    Teilt einen langen Text in Chunks (ca. max_length Zeichen). Diese Implementierung teilt zeilenweise.
    """
    chunks = []
    current_chunk = ""
    for line in text.splitlines():
        if len(current_chunk) + len(line) + 1 > max_length:
            chunks.append(current_chunk.strip())
            current_chunk = line
        else:
            current_chunk += " " + line
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def build_faiss_index_from_knowledge(knowledge_text):
    """
    Zerlegt den kombinierten Wissens-Text in Chunks, berechnet deren Embeddings und erstellt einen FAISS-Index.
    Das Mapping der Chunks wird in der globalen Variable faiss_chunk_list gespeichert.
    """
    global faiss_index, faiss_chunk_list
    chunks = chunk_text(knowledge_text, max_length=500)
    embeddings = []
    for chunk in chunks:
        emb = get_embedding(chunk)
        if emb is not None:
            embeddings.append(emb)
        else:
            print("Kein Embedding für Chunk:", chunk[:30])
    if len(embeddings) == 0:
        print("Keine gültigen Embeddings gefunden!")
        return
    embeddings = np.array(embeddings).astype("float32")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss_index = index
    faiss_chunk_list = chunks
    print(f"FAISS-Index aufgebaut mit {len(chunks)} Chunks.")

def load_current_knowledge():
    """
    Liest den aktuellen Wissensbestand aus der knowledge.txt und aus allen hochgeladenen .txt-Dateien.
    """
    knowledge = ""
    try:
        with open("knowledge.txt", "r", encoding="utf-8") as f:
            file_content = f.read()
            print("Inhalt von knowledge.txt (erste 100 Zeichen):", file_content[:100])
            knowledge += file_content + "\n"
    except FileNotFoundError:
        print("knowledge.txt nicht gefunden.")
        knowledge = ""
    uploaded_knowledge = ""
    uploaded_files = UploadedFile.query.all()
    for file_record in uploaded_files:
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file_record.filename)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                print(f"Inhalt von {file_record.filename} (erste 100 Zeichen):", content[:100])
                uploaded_knowledge += content + "\n"
        except Exception as e:
            print(f"Fehler beim Lesen der Datei {file_record.filename}:", e)
            continue
    full_knowledge = knowledge + "\n" + uploaded_knowledge
    return full_knowledge

def refresh_knowledge_index():
    """
    Aktualisiert den FAISS-Index mit den aktuellen Daten aus der Wissensbasis.
    Setzt den API-Key aus der DB und baut den Index neu auf.
    """
    global faiss_index, faiss_chunk_list
    with app.app_context():
        stored_key = APIKey.query.first()
        if stored_key:
            openai.api_key = stored_key.key
        new_knowledge = load_current_knowledge()
        if new_knowledge.strip():
            build_faiss_index_from_knowledge(new_knowledge)
            print("Wissensindex wurde aktualisiert.")
        else:
            print("Kein Wissen gefunden. Index nicht aktualisiert.")

# --------------------------
# Admin-Oberfläche
# --------------------------
class MyAdminIndexView(AdminIndexView):
    @expose("/")
    def index(self):
        return self.render("admin_index.html")

admin = Admin(app, name="ShopBot Admin", template_mode="bootstrap3", index_view=MyAdminIndexView())

# --------------------------
# Routen für Webseiten-Verwaltung, Datei-Upload, Löschen usw.
# --------------------------
@app.route("/admin/websites", methods=["GET", "POST"])
def manage_websites():
    if request.method == "POST":
        new_url = request.form.get("url", "").strip()
        if new_url:
            # Falls kein Protokoll eingegeben wurde, füge automatisch https:// hinzu.
            if not (new_url.startswith("http://") or new_url.startswith("https://")):
                new_url = "https://" + new_url

            if not CrawledWebsite.query.filter_by(url=new_url).first():
                db.session.add(CrawledWebsite(url=new_url))
                db.session.commit()
                flash("Webseite erfolgreich hinzugefügt!", "success")
            else:
                flash("Diese Webseite ist bereits in der Liste.", "warning")
    websites = CrawledWebsite.query.all()
    return render_template("manage_websites.html", websites=websites)

@app.route("/admin/upload", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename.endswith(".txt"):
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)
            new_file = UploadedFile(filename=file.filename)
            db.session.add(new_file)
            db.session.commit()
            flash("Datei erfolgreich hochgeladen!", "success")
        else:
            flash("Nur .txt-Dateien erlaubt!", "error")
    files = UploadedFile.query.all()
    return render_template("upload.html", files=files)

@app.route("/admin/delete_website/<int:website_id>", methods=["POST"])
def delete_website(website_id):
    website = CrawledWebsite.query.get(website_id)
    if website:
        db.session.delete(website)
        db.session.commit()
        flash("Webseite gelöscht!", "success")
    else:
        flash("Webseite nicht gefunden!", "error")
    return redirect(url_for("manage_websites"))

@app.route("/admin/delete-all", methods=["POST"])
def delete_all_crawled_data():
    try:
        num_deleted = db.session.query(CrawledPage).delete()
        db.session.commit()
        flash(f"✅ {num_deleted} gecrawlte Einträge wurden gelöscht!", "success")
    except Exception as e:
        db.session.rollback()
        flash(f"❌ Fehler beim Löschen: {str(e)}", "error")
    return redirect(url_for("crawled_data"))

@app.route("/admin/crawled-data")
def crawled_data():
    pages = CrawledPage.query.order_by(CrawledPage.id.desc()).all()
    return render_template("crawled_data.html", pages=pages)

@app.route("/admin/delete_crawled/<int:page_id>", methods=["POST"])
def delete_crawled(page_id):
    page = CrawledPage.query.get(page_id)
    if page:
        db.session.delete(page)
        db.session.commit()
        flash("✅ Gecrawlte Seite wurde gelöscht!", "success")
    else:
        flash("❌ Seite nicht gefunden!", "error")
    return redirect(url_for("crawled_data"))

@app.route("/admin/start-crawling", methods=["POST"])
def start_crawling():
    with app.app_context():
        websites = CrawledWebsite.query.all()
        if not websites:
            flash("❌ Keine Webseiten zum Crawlen eingetragen!", "error")
        else:
            def crawl():
                crawled_text = []
                for site in websites:
                    print(f"🔍 Starte Crawling für: {site.url}")
                    text = crawl_website(site.url)
                    if text:
                        crawled_text.append(text)
                all_text = "\n".join(crawled_text)
                try:
                    with open(KNOWLEDGE_FILE, "w", encoding="utf-8") as f:
                        f.write(all_text)
                    print("knowledge.txt wurde geschrieben.")
                except Exception as e:
                    print("Fehler beim Schreiben von knowledge.txt:", e)
                flash("✅ Crawling abgeschlossen und Wissen aktualisiert!", "success")
                # Setze den FAISS-Index zurück, damit er beim nächsten Chat neu aufgebaut wird.
                global faiss_index, faiss_chunk_list
                faiss_index, faiss_chunk_list = None, None
            thread = threading.Thread(target=crawl)
            thread.start()
            flash("🚀 Crawling gestartet! Dies kann einige Minuten dauern.", "info")
    return redirect(url_for("manage_websites"))


@app.route("/admin/basisprompt", methods=["GET", "POST"])
def manage_basisprompt():
    if request.method == "POST":
        new_prompt = request.form.get("prompt")
        if new_prompt:
            prompt_entry = BotPrompt.query.first()
            if prompt_entry:
                prompt_entry.prompt = new_prompt
            else:
                prompt_entry = BotPrompt(prompt=new_prompt)
                db.session.add(prompt_entry)
            db.session.commit()
            flash("Basisprompt gespeichert!", "success")
    stored_prompt = BotPrompt.query.first()
    current_prompt = stored_prompt.prompt if stored_prompt else "Ich bin ein hilfreicher Assistent."
    return render_template("basisprompt.html", basisprompt=current_prompt)

@app.route("/admin/delete_file/<int:file_id>", methods=["POST"])
def delete_file(file_id):
    file_record = UploadedFile.query.get(file_id)
    if file_record:
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file_record.filename)
        if os.path.exists(file_path):
            os.remove(file_path)
        db.session.delete(file_record)
        db.session.commit()
        flash("Datei erfolgreich gelöscht!", "success")
    else:
        flash("Datei nicht gefunden!", "error")
    return redirect(url_for("upload_file"))

@app.route("/admin/settings", methods=["GET", "POST"])
def manage_settings():
    if request.method == "POST":
        new_key = request.form.get("api_key")
        new_text = request.form.get("greeting_text")
        new_prompt = request.form.get("bot_prompt")
        if new_key:
            existing_key = APIKey.query.first()
            if existing_key:
                existing_key.key = new_key
            else:
                db.session.add(APIKey(key=new_key))
        if new_text:
            greeting = GreetingMessage.query.first()
            if greeting:
                greeting.text = new_text
            else:
                db.session.add(GreetingMessage(text=new_text))
        if new_prompt:
            prompt_entry = BotPrompt.query.first()
            if prompt_entry:
                prompt_entry.prompt = new_prompt
            else:
                db.session.add(BotPrompt(prompt=new_prompt))
        db.session.commit()
        flash("Einstellungen gespeichert!", "success")
    stored_key = APIKey.query.first()
    current_key = stored_key.key if stored_key else ""
    stored_greeting = GreetingMessage.query.first()
    current_greeting = stored_greeting.text if stored_greeting else "Hallo! Wie kann ich helfen?"
    stored_prompt = BotPrompt.query.first()
    current_prompt = stored_prompt.prompt if stored_prompt else "Ich bin ein hilfreicher Assistent."
    return render_template("settings.html", api_key=current_key, greeting_text=current_greeting, bot_prompt=current_prompt)

@app.route("/api/greeting", methods=["GET"])
def get_greeting():
    stored_greeting = GreetingMessage.query.first()
    greeting_text = stored_greeting.text if stored_greeting else "Hallo! Wie kann ich helfen?"
    return jsonify({"greeting_text": greeting_text})

# Neuer Endpunkt: Wissensindex aktualisieren per Knopfdruck
@app.route("/admin/update_index", methods=["POST"])
def update_index():
    thread = threading.Thread(target=refresh_knowledge_index)
    thread.start()
    flash("Der Wissensindex wird aktualisiert. Bitte etwas Geduld!", "info")
    return redirect(url_for("admin.index"))

# --------------------------
# Chatbot-Endpoint mit Retrieval-Augmented Generation (FAISS)
# --------------------------
@app.route("/chat", methods=["POST"])
def chatbot():
    user_input = request.json.get("message")
    if not user_input:
        return jsonify({"error": "Keine Eingabe erhalten"}), 400

    # API-Schlüssel abrufen
    def get_openai_api_key():
        env_key = os.getenv("OPENAI_API_KEY")
        if env_key:
            return env_key

        stored_key = APIKey.query.first()
        if stored_key:
            return stored_key.key

        return None

    openai.api_key = get_openai_api_key()
    if not openai.api_key:
        return jsonify({"error": "Kein API-Key gespeichert"}), 500

    # Basis-Prompt abrufen
    stored_prompt = BotPrompt.query.first()
    base_prompt = stored_prompt.prompt if stored_prompt else "Ich bin ein hilfreicher Assistent."

    # Begrüßungstext abrufen
    stored_greeting = GreetingMessage.query.first()
    greeting_message = stored_greeting.text if stored_greeting else "Hallo! Wie kann ich helfen?"

    # FAISS-Index laden, falls noch nicht geschehen
    global faiss_index, faiss_chunk_list
    if faiss_index is None:
        knowledge_text = load_current_knowledge()
        if knowledge_text.strip():
            build_faiss_index_from_knowledge(knowledge_text)

    # Falls kein FAISS-Index vorhanden ist
    relevant_context = ""
    if faiss_index is not None:
        query_embedding = get_embedding(user_input)
        if query_embedding is None:
            return jsonify({"error": "Fehler beim Abrufen des Embeddings."}), 500
        
        query_embedding = np.array(query_embedding).astype("float32").reshape(1, -1)
        k = 3
        distances, indices = faiss_index.search(query_embedding, k)
        relevant_chunks = [faiss_chunk_list[idx] for idx in indices[0] if idx < len(faiss_chunk_list)]
        relevant_context = "\n\n".join(relevant_chunks)

    # OpenAI API aufrufen
    messages = [
        {"role": "system", "content": base_prompt},
        {"role": "system", "content": f"Relevante Informationen:\n{relevant_context}"},
        {"role": "assistant", "content": greeting_message},
        {"role": "user", "content": user_input}
    ]

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages
        )
        return jsonify({"response": response.choices[0].message.content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500