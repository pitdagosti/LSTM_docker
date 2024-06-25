import socket
import numpy as np
import tensorflow as tf
import re
import time
import sqlite3
from sklearn.preprocessing import MinMaxScaler
import logging
import os

# Configurazione del logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s')

def predict_future(X_test, scaler, model):
    try:
        X_test_scaled = scaler.transform(X_test)
        future = model.predict(X_test_scaled.reshape(1, -1, 1))
        future_value = scaler.inverse_transform(future)[0][0]
        return future_value
    except Exception as e:
        logging.error(f"Errore durante la predizione: {e}")
        return None

def clean_value(value):
    cleaned_value = re.sub(r"[\[\]']", "", str(value))
    return cleaned_value

# Funzione per inizializzare il database
def init_db(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS temperatures
                 (node_name TEXT, id_attuale INTEGER, future_value REAL, temperature_label TEXT, temperatura_float REAL, timestamp INTEGER)''')
    conn.commit()
    conn.close()

# Funzione per salvare i dati nel database
def save_to_db(db_path, data):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("INSERT INTO temperatures (node_name, id_attuale, future_value, temperature_label, temperatura_float, timestamp) VALUES (?, ?, ?, ?, ?, ?)", data)
    conn.commit()
    conn.close()

# Path del database
db_path = '/experiment/temperature_data.db'

# Inizializza il database
init_db(db_path)

# Carica il modello una volta
loaded_model = tf.keras.models.load_model("model_J0700_VELOCE.keras")
logging.info("Modello caricato con successo.")

# Configurazione di TensorFlow
tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()

# Inizializza il MinMaxScaler per normalizzare i dati
scaler = MinMaxScaler(feature_range=(0, 1))

# Definizione della finestra di dati scorrevole
window_size = 20
current_window = []

# Connetti al server TCP
HOST = '127.0.0.1'  # Indirizzo IP del server
PORT = 7070         # Porta su cui il server è in ascolto

# Variabile per tenere traccia se il MinMaxScaler è stato addestrato
scaler_fitted = False

# Variabile per tracciare l'ID attuale delle temperature
id_attuale = 0

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    logging.info('Connesso al server TCP')

    while True:
        data = s.recv(1024)  # Ricevi dati dal server

        if not data:
            break

        decoded_data = data.decode('utf-8').strip()  # Decodifica e rimuovi spazi bianchi
        #logging.debug(f"Dati ricevuti: {decoded_data}")

        parts = decoded_data.split(';')
        if len(parts) == 3:
            try:
                node_name = parts[0]
                temperature_label = parts[1]  # Etichetta della temperatura
                temperatura_float = float(parts[2])
                
                 if temperatura_float == 0.0 or temperatura_float is None:
                continue
                
            except ValueError as e:
                logging.error(f"Errore nella conversione dei dati: {e}")
                continue
        else:
            logging.error("Formato dei dati ricevuti non valido.")
            continue

        # Aggiorna la finestra di dati scorrevole
        current_window.append(temperatura_float)
        if len(current_window) > window_size:
            current_window.pop(0)  # Rimuove il valore più vecchio se la finestra supera window_size

        #logging.debug(f"Finestra corrente: {current_window}")

        # Addestra il MinMaxScaler sui primi dati ricevuti, se non è ancora stato addestrato
        if not scaler_fitted and len(current_window) >= window_size:
            data_array = np.array(current_window[-window_size:]).reshape(-1, 1)
            scaler.fit(data_array)
            scaler_fitted = True
            logging.info("MinMaxScaler addestrato con successo.")

        # Effettua la predizione basata sulla finestra di dati scorrevole attuale
        if scaler_fitted and len(current_window) >= window_size:
            X_test = np.array(current_window[-window_size:]).reshape(-1, 1)
            #logging.debug(f"Dati per la predizione: {X_test}")
            
            future_value = predict_future(X_test, scaler, loaded_model)
            if future_value is not None:
                future_value_clean = clean_value(future_value)
                current_time_epoch = int(time.time())
                
                # Incrementa l'ID attuale
                id_attuale += 1
                
                # Crea i dati da salvare nel database
                data_to_save = (node_name, id_attuale, future_value_clean, temperature_label, temperatura_float, current_time_epoch)
                
                # Salva i dati nel database
                save_to_db(db_path, data_to_save)
                
                # Logga l'output
                logging.info(f"Output: {data_to_save}")

        time.sleep(1)  # Aggiungi un ritardo per evitare di ricevere troppi dati rapidamente
