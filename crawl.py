import os
import time
import datetime
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

# Flask-App und Datenbank-Konfiguration
app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///shopbot.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# Models für gecrawlte Seiten und die zu crawlenden Webseiten
class CrawledPage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    url = db.Column(db.String(500), nullable=False, unique=True)
    content = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)

class CrawledWebsite(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    url = db.Column(db.String(500), nullable=False, unique=True)

# Erstelle die Tabellen, falls sie noch nicht existieren
with app.app_context():
    db.create_all()

# Verzeichnis, in dem hochgeladene TXT-Dateien liegen (anpassen, falls nötig)
UPLOADS_DIR = os.path.abspath("uploads")

def get_text_from_page(url):
    """
    Lädt eine Webseite und extrahiert den sichtbaren Text (falls HTML).
    Bei anderen Inhalten (z. B. PDFs, Bildern) wird None zurückgegeben.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/110.0.0.0 Safari/537.36"
        )
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        # Prüfen, ob es sich um HTML handelt
        content_type = response.headers.get("Content-Type", "").lower()
        if "text/html" not in content_type:
            print(f"❌ Kein HTML-Inhalt für {url} ({content_type})")
            return None

        # HTML parsen – hier verwenden wir den lxml-Parser
        soup = BeautifulSoup(response.text, "lxml")

        # Entferne unerwünschte Tags (Script, Style, Header, Footer, Navigation)
        for tag in soup(["script", "style", "header", "footer", "nav"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)
        return text

    except requests.RequestException as e:
        print(f"❌ Fehler beim Laden von {url}: {e}")
        return None
    except Exception as e:
        print(f"❌ Parser-Fehler bei {url}: {e}")
        return None

def find_all_links(url, base_domain):
    """
    Sucht alle internen Links auf einer Seite, die zur Domain base_domain gehören.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/110.0.0.0 Safari/537.36"
        )
    }
    found_links = set()
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "").lower()
        if "text/html" not in content_type:
            return found_links

        soup = BeautifulSoup(response.text, "lxml")
        for link in soup.find_all("a", href=True):
            absolute_link = urljoin(url, link["href"])
            parsed_link = urlparse(absolute_link)
            if parsed_link.netloc == base_domain:
                found_links.add(absolute_link)
    except requests.RequestException as e:
        print(f"❌ Fehler beim Suchen von Links auf {url}: {e}")

    return found_links

def load_uploaded_txts():
    """
    Liest alle .txt-Dateien im UPLOADS_DIR und gibt ihren Inhalt als String zurück.
    """
    texts = []
    if not os.path.isdir(UPLOADS_DIR):
        print(f"Upload-Verzeichnis nicht gefunden: {UPLOADS_DIR}")
        return ""
    for filename in os.listdir(UPLOADS_DIR):
        if filename.lower().endswith(".txt"):
            file_path = os.path.join(UPLOADS_DIR, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    file_content = f.read()
                    print(f"Inhalt von {filename} (erste 100 Zeichen):", file_content[:100])
                    texts.append(f"Datei: {filename}\n{file_content}\n")
            except Exception as e:
                print(f"❌ Fehler beim Lesen von {filename}: {e}")
    return "\n".join(texts)

def crawl_website(start_url, visited=None):
    """
    Crawlt eine Webseite und gibt den gesamten gefundenen Text (aller Unterseiten) als String zurück.
    Gleichzeitig speichert es den Text in der Datenbank und sammelt den Text, um ihn später in die
    knowledge.txt zu schreiben.
    """
    if visited is None:
        visited = set()

    parsed_url = urlparse(start_url)
    if not parsed_url.scheme:
        start_url = "https://" + start_url

    base_domain = urlparse(start_url).netloc
    to_visit = {start_url}
    collected_texts = []

    while to_visit:
        url = to_visit.pop()
        if url in visited:
            continue

        print(f"🔍 Crawling: {url}")
        text = get_text_from_page(url)
        if text:
            with app.app_context():
                existing_page = CrawledPage.query.filter_by(url=url).first()
                if not existing_page:
                    new_page = CrawledPage(url=url, content=text)
                    db.session.add(new_page)
                    db.session.commit()
                    print(f"✅ Gespeichert: {url}")
            collected_texts.append(f"URL: {url}\n{text}\n")
        visited.add(url)
        new_links = find_all_links(url, base_domain)
        to_visit.update(new_links - visited)
        time.sleep(1)

    return "\n".join(collected_texts)

def update_knowledge():
    """
    Führt das Crawling durch und liest zusätzlich alle hochgeladenen TXT-Dateien ein.
    Schreibt dann den gesamten gesammelten Text in die Datei knowledge.txt.
    """
    # Hier kannst du festlegen, welche Webseiten gecrawlt werden sollen.
    # Beispielsweise: alle in der Datenbank gespeicherten Webseiten oder eine feste URL.
    # Für dieses Beispiel nutzen wir eine feste URL:
    url_to_crawl = "https://example.com"  # <-- hier die gewünschte Start-URL eintragen

    crawled_text = crawl_website(url_to_crawl)
    uploaded_text = load_uploaded_txts()

    combined_text = crawled_text + "\n" + uploaded_text

    # Schreibe in die knowledge.txt
    knowledge_file = os.path.abspath("knowledge.txt")
    print("Schreibe knowledge.txt in:", knowledge_file)
    try:
        with open(knowledge_file, "w", encoding="utf-8") as f:
            f.write(combined_text)
        print("knowledge.txt wurde erfolgreich geschrieben.")
    except Exception as e:
        print("Fehler beim Schreiben von knowledge.txt:", e)

    return combined_text

if __name__ == "__main__":
    # Beispiel: Update der Wissensdatenbank (Crawling + TXT-Dateien)
    update_knowledge()
