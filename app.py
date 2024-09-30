import numpy as np
import tensorflow as tf
import re
import time
import os
import csv
import socket
from sklearn.preprocessing import MinMaxScaler

def load_tflite_model(model_path):
    """Carica il modello TFLite e alloca i tensori."""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def predict_with_tflite(interpreter, X_test):
    """Esegue la predizione utilizzando un modello TFLite."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Prepara i dati per l'input
    interpreter.set_tensor(input_details[0]['index'], X_test.astype(np.float32))

    # Esegui l'inferenza
    interpreter.invoke()

    # Recupera l'output
    output = interpreter.get_tensor(output_details[0]['index'])
    return output

def predict_future(X_test, scaler, interpreter):
    try:
        # Scala i dati di input
        X_test_scaled = scaler.transform(X_test)
        
        # Assicurati che X_test_scaled abbia la forma corretta per il modello
        X_test_scaled = X_test_scaled.reshape(1, -1, 1).astype(np.float32)  # Dimensione fissa (1, window_size, 1)

        # Effettua la predizione con il modello TFLite
        future = predict_with_tflite(interpreter, X_test_scaled)
        future_value = scaler.inverse_transform(future)[0][0]  # Inverso della scalatura
        return future_value
    except Exception as e:
        print(f"Errore durante la predizione: {e}")
        return None

def save_to_csv(file_path, data):
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['ID', 'Node_Name', 'Temperature_Actual', 'Temperature_Future', 'Label', 'Timestamp'])
        writer.writerow(data)

# Carica il modello TFLite
try:
    interpreter = load_tflite_model("lstm.tflite")
    print("Modello TFLite caricato con successo.")
except Exception as e:
    print(f"Errore durante il caricamento del modello TFLite: {e}")
    raise

tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()

window_size = 20  # Cambia a 168 se il tuo modello si aspetta 168

# Dizionari per tracciare le finestre di dati e gli scaler per ogni nodo
node_windows = {}
node_scalers = {}
node_scaler_fitted = {}

# Dizionario per tracciare l'ID corrente per ogni nodo
node_ids = {}

# Indirizzo e porta del server TCP
TCP_IP = '100.109.221.5'  # Cambia con l'indirizzo del server
TCP_PORT = 7070       # Cambia con la porta corretta
BUFFER_SIZE = 1024

# Connessione al server TCP
try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((TCP_IP, TCP_PORT))
    print(f"Connesso al server TCP {TCP_IP}:{TCP_PORT}.")
except Exception as e:
    print(f"Errore durante la connessione al server: {e}")
    raise

csv_file_path = '/usr/src/app/dist/dbFiles/temperature_predictions.csv'

try:
    while True:
        data = sock.recv(BUFFER_SIZE).decode('utf-8')
        if not data:
            print("Nessun dato ricevuto. Chiusura connessione.")
            break

        decoded_data = data.strip()

        if not decoded_data:
            print("Dati vuoti ricevuti, saltati.")
            continue

        # Dividi i dati in righe
        lines = decoded_data.split('\n')

        for line in lines:
            parts = line.split(';')

            if len(parts) == 3:
                try:
                    node_name = parts[0]
                    temperature_label = parts[1]
                    temperatura_float = float(parts[2])

                    # Controllo per escludere esplicitamente lo zero e valori anomali
                    if temperatura_float == 0.0 or temperatura_float is None or abs(temperatura_float) < 0.1 or temperatura_float < -50 or temperatura_float > 100:
                        print(f"Temperatura non valida (0 o fuori range): {temperatura_float} per il nodo {node_name}")
                        continue

                except ValueError as e:
                    print(f"Errore nella conversione dei dati: {e}")
                    continue
            else:
                print(f"Formato dei dati ricevuti non valido per la riga: {line}")
                continue

        # Inizializza la finestra di dati e lo scaler per il nodo se non esistono
        if node_name not in node_windows:
            node_windows[node_name] = []
            node_scalers[node_name] = MinMaxScaler(feature_range=(0, 1))
            node_scaler_fitted[node_name] = False

        # Aggiorna la finestra di dati scorrevole per il nodo corrente
        node_windows[node_name].append(temperatura_float)
        if len(node_windows[node_name]) > window_size:
            node_windows[node_name].pop(0)  # Rimuove il valore più vecchio se la finestra supera window_size

        # Addestra il MinMaxScaler per il nodo corrente
        if not node_scaler_fitted[node_name] and len(node_windows[node_name]) >= window_size:
            data_array = np.array(node_windows[node_name][-window_size:]).reshape(-1, 1)
            node_scalers[node_name].fit(data_array)
            node_scaler_fitted[node_name] = True
            print(f"MinMaxScaler addestrato con successo per il nodo {node_name}.")

        # Effettua la predizione per il nodo corrente
        if node_scaler_fitted[node_name] and len(node_windows[node_name]) >= window_size:
            # Assicurati che X_test sia sempre di dimensione (1, window_size, 1)
            X_test = np.array(node_windows[node_name][-window_size:]).reshape(-1, 1)

            # Effettua la predizione utilizzando il modello TFLite
            future_value = predict_future(X_test, node_scalers[node_name], interpreter)
            if future_value is not None and abs(future_value) >= 0.1 and future_value != 0:
                current_time_epoch = int(time.time())

                if node_name not in node_ids:
                    node_ids[node_name] = 0
                else:
                    node_ids[node_name] += 1

                data_to_save = [node_ids[node_name], node_name, temperatura_float, future_value, temperature_label, current_time_epoch]
                save_to_csv(csv_file_path, data_to_save)
                print(f"Output: {data_to_save}")

        # Ritardo per evitare di ricevere troppi dati rapidamente
        time.sleep(1)

except KeyboardInterrupt:
    print("Interruzione manuale ricevuta.")
except Exception as e:
    print(f"Errore inaspettato: {e}")
finally:
    sock.close()
    print("Connessione al server TCP chiusa.")
