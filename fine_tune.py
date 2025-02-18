import openai
from app import app, db, APIKey  # Importiere deine Flask-App und das APIKey-Modell

# Setze den API-Key aus der Datenbank innerhalb des Anwendungskontexts
with app.app_context():
    stored_key = APIKey.query.first()
    if stored_key:
        openai.api_key = stored_key.key
    else:
        print("Kein API-Key gefunden. Bitte gib einen API-Key in der Admin-Oberfläche ein.")
        exit(1)

# Trainingsdaten-Datei hochladen
file_response = openai.File.create(
    file=open("training_data.jsonl", "rb"),
    purpose="fine-tune"
)
print("Datei hochgeladen:", file_response["id"])

# Fine-Tuning starten
fine_tune_response = openai.FineTune.create(
    training_file=file_response["id"],
    model="davinci"  # Passe hier ggf. das Modell an, das Fine-Tuning unterstützt
)
print("Fine-Tuning gestartet, ID:", fine_tune_response["id"])
