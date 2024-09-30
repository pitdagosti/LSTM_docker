import csv
import os
import socket
import time
from typing import Dict, List

import numpy as np
import tensorflow as tf
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
    interpreter.set_tensor(input_details[0]["index"], X_test.astype(np.float32))

    # Esegui l'inferenza
    interpreter.invoke()

    # Recupera l'output
    output = interpreter.get_tensor(output_details[0]["index"])
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
    with open(file_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(
                [
                    "ID",
                    "Node_Name",
                    "Temperature_Actual",
                    "Temperature_Future",
                    "Label",
                    "Timestamp",
                ]
            )
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
node_windows: Dict[str, List[float]] = {}
node_scalers: Dict[str, MinMaxScaler] = {}
node_scaler_fitted: Dict[str, bool] = {}

# Dizionario per tracciare l'ID corrente per ogni nodo
node_ids: Dict[str, int] = {}

# Indirizzo e porta del server TCP
TCP_HOSTNAME = "100.109.221.5"  # Nome del dominio mDNS della Raspberry Pi
TCP_PORT = 7070  # Cambia con la porta corretta
BUFFER_SIZE = 1024


# Funzione per tentare la connessione al server
def connect_to_server():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        # Risolvi il nome mDNS (es. raspberrypi.local)
        sock.connect((TCP_HOSTNAME, TCP_PORT))
        print(f"Connesso al server TCP {TCP_HOSTNAME}:{TCP_PORT}.")
        return sock
    except Exception as e:
        print(f"Errore durante la connessione al server {TCP_HOSTNAME}:{TCP_PORT}: {e}")
        return None


# Connessione al server TCP
sock = None
while sock is None:
    print("Tentativo di connessione al server...")
    sock = connect_to_server()
    time.sleep(5)  # Attende 5 secondi prima di un nuovo tentativo

csv_file_path = "/usr/src/app/dist/dbFiles/temperature_predictions.csv"

try:
    # Ciclo infinito che mantiene attiva la connessione TCP e gestisce i dati ricevuti
    while True:
        try:
            # Riceve i dati dal server TCP, con una dimensione massima definita da BUFFER_SIZE
            data = sock.recv(BUFFER_SIZE).decode("utf-8")  # I dati ricevuti vengono decodificati in UTF-8

            # Se non riceve alcun dato (es. il server chiude la connessione), tenta di riconnettersi
            if not data:
                print("Nessun dato ricevuto. Tentativo di riconnessione...")
                sock.close()  # Chiude il socket
                sock = None  # Imposta il socket a None per indicare che la connessione è persa

                # Ciclo di riconnessione: tenta di ristabilire la connessione finché non riesce
                while sock is None:
                    print("Tentativo di riconnessione al server...")
                    sock = connect_to_server()  # Chiama la funzione per riconnettersi al server
                    time.sleep(5)  # Attende 5 secondi prima di un nuovo tentativo
                continue  # Ritorna all'inizio del ciclo principale una volta che la connessione è ristabilita

            # Rimuove eventuali spazi bianchi o caratteri di nuova linea dai dati ricevuti
            decoded_data = data.strip()

            # Se i dati sono vuoti (anche dopo aver rimosso spazi), non fa nulla e continua
            if not decoded_data:
                print("Dati vuoti ricevuti, saltati.")
                continue  # Salta al prossimo ciclo del loop

            # Divide i dati in righe (nel caso il server invii più righe di dati in un singolo pacchetto)
            lines = decoded_data.split("\n")

            # Itera su ogni riga separatamente per processarla
            for line in lines:
                # Divide ogni riga in tre parti separate dal delimitatore ';'
                parts = line.split(";")

                # Controlla che ci siano esattamente 3 parti (nome del nodo, etichetta della temperatura, valore della temperatura)
                if len(parts) == 3:
                    try:
                        # Estrae le tre parti: nome del nodo, etichetta della temperatura, e la temperatura stessa
                        node_name = parts[0]
                        temperature_label = parts[1]
                        temperatura_float = float(parts[2])  # Converte il valore della temperatura in un float

                        # Filtra i dati con controlli di validità per evitare valori anomali
                        # Ignora le temperature che sono esattamente 0.0, fuori dal range logico (-50°C a 100°C), o valori molto piccoli
                        if (
                            temperatura_float == 0.0
                            or temperatura_float is None
                            or abs(temperatura_float) < 0.1
                        ):
                            print(f"Temperatura non valida (0 o fuori range): {temperatura_float} per il nodo {node_name}")
                            continue  # Salta al prossimo ciclo del loop se i dati non sono validi

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

                                # Salva i risultati nel file CSV
                                data_to_save = [
                                    node_ids[node_name],
                                    node_name,
                                    temperatura_float,
                                    future_value,
                                    temperature_label,
                                    current_time_epoch,
                                ]
                                save_to_csv(csv_file_path, data_to_save)
                                print(f"Output: {data_to_save}")

                    except ValueError as e:
                        print(f"Errore nella conversione dei dati: {e}")
                        continue  # Salta al prossimo ciclo del loop in caso di errore
                else:
                    # Se la riga non è nel formato previsto (con 3 parti separate da ';'), viene ignorata
                    print(f"Formato dei dati ricevuti non valido per la riga: {line}")
                    continue  # Salta al prossimo ciclo del loop

        except (socket.error, KeyboardInterrupt):
            print("Errore o interruzione manuale. Tentativo di riconnessione...")
            sock.close()
            sock = None
            while sock is None:
                print("Tentativo di riconnessione al server...")
                sock = connect_to_server()
                time.sleep(5)  # Attende 5 secondi prima di un nuovo tentativo

except KeyboardInterrupt:
    print("Interruzione manuale ricevuta.")
except Exception as e:
    print(f"Errore inaspettato: {e}")
finally:
    if sock is not None:
        sock.close()
        print("Connessione al server TCP chiusa.")
