FROM python:3.11
WORKDIR /app
COPY requirements.txt ./
COPY app.py ./
COPY docker-entrypoint.sh ./
COPY model_50_sens.keras ./
COPY lstm.tflite ./

# Installa le dipendenze
RUN pip install --no-cache-dir -r requirements.txt

# Rendi lo script di entrypoint eseguibile
RUN chmod +x docker-entrypoint.sh

# Dichiarazione del volume (opzionale)
VOLUME ["/usr/src/app/dist/dbFiles"]

# Imposta il comando di avvio
ENTRYPOINT ["./docker-entrypoint.sh"]
