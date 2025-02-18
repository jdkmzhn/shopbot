import os
OPENAI_API_KEY = "DEIN-API-KEY"
class Config:
    SECRET_KEY = "supersecretkey"  # Später sicherer machen!
    UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")  # Speicherort für Dateien
    MAX_CONTENT_LENGTH = 2 * 1024 * 1024  # Max. 2MB pro Datei
    SQLALCHEMY_DATABASE_URI = "sqlite:///files.db"  # SQLite-Datenbank
    SQLALCHEMY_TRACK_MODIFICATIONS = False