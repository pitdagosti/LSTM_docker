import csv
import os
import socket
import time
from typing import Dict, List

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def predict_with_tflite(interpreter, X_test):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]["index"], X_test.astype(np.float32))
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])
    return output


def predict_future(X_test, scaler, interpreter):
    try:
        X_test_scaled = scaler.transform(X_test)
        X_test_scaled = X_test_scaled.reshape(1, -1, 1).astype(np.float32)
        future = predict_with_tflite(interpreter, X_test_scaled)
        future_value = scaler.inverse_transform(future)[0][0]
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


try:
    interpreter = load_tflite_model("lstm.tflite")
    print("Modello TFLite caricato con successo.")
except Exception as e:
    print(f"Errore durante il caricamento del modello TFLite: {e}")
    raise

tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()

window_size = 20

node_windows: Dict[str, List[float]] = {}
node_scalers: Dict[str, MinMaxScaler] = {}
node_scaler_fitted: Dict[str, bool] = {}
node_ids: Dict[str, int] = {}

TCP_HOSTNAME = "100.109.221.5"
TCP_PORT = 7070
BUFFER_SIZE = 1024


def connect_to_server():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((TCP_HOSTNAME, TCP_PORT))
        print(f"Connesso al server TCP {TCP_HOSTNAME}:{TCP_PORT}.")
        return sock
    except Exception as e:
        print(f"Errore durante la connessione al server {TCP_HOSTNAME}:{TCP_PORT}: {e}")
        return None


sock = None
while sock is None:
    print("Tentativo di connessione al server...")
    sock = connect_to_server()
    time.sleep(5)

csv_file_path = "/usr/src/app/dist/dbFiles/temperature_predictions.csv"

try:
    while True:
        try:
            data = sock.recv(BUFFER_SIZE).decode("utf-8")
            if not data:
                print("Nessun dato ricevuto. Tentativo di riconnessione...")
                sock.close()
                sock = None
                while sock is None:
                    print("Tentativo di riconnessione al server...")
                    sock = connect_to_server()
                    time.sleep(5)
                continue

            decoded_data = data.strip()
            if not decoded_data:
                print("Dati vuoti ricevuti, saltati.")
                continue

            lines = decoded_data.split("\n")
            for line in lines:
                parts = line.split(";")
                if len(parts) == 3:
                    try:
                        node_name = parts[0]
                        temperature_label = parts[1]
                        temperatura_float = float(parts[2])

                        if (
                            temperatura_float == 0.0
                            or temperatura_float is None
                            or abs(temperatura_float) < 0.1
                        ):
                            print(
                                f"Temperatura non valida (0 o fuori range): {temperatura_float} per il nodo {node_name}"
                            )
                            continue

                        if node_name not in node_windows:
                            node_windows[node_name] = []
                            node_scalers[node_name] = MinMaxScaler(feature_range=(0, 1))
                            node_scaler_fitted[node_name] = False

                        node_windows[node_name].append(temperatura_float)
                        if len(node_windows[node_name]) > window_size:
                            node_windows[node_name].pop(0)

                        if (
                            not node_scaler_fitted[node_name]
                            and len(node_windows[node_name]) >= window_size
                        ):
                            data_array = np.array(node_windows[node_name][-window_size:]).reshape(-1, 1)
                            node_scalers[node_name].fit(data_array)
                            node_scaler_fitted[node_name] = True
                            print(
                                f"MinMaxScaler addestrato con successo per il nodo {node_name}."
                            )

                        if (
                            node_scaler_fitted[node_name]
                            and len(node_windows[node_name]) >= window_size
                        ):
                            X_test = np.array(node_windows[node_name][-window_size:]).reshape(-1, 1)
                            future_value = predict_future(
                                X_test, node_scalers[node_name], interpreter
                            )
                            if (
                                future_value is not None
                                and abs(future_value) >= 0.1
                                and future_value != 0
                            ):
                                current_time_epoch = int(time.time())
                                if node_name not in node_ids:
                                    node_ids[node_name] = 0
                                else:
                                    node_ids[node_name] += 1
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
                        continue
                else:
                    print(f"Formato dei dati ricevuti non valido per la riga: {line}")
                    continue

        except (socket.error, KeyboardInterrupt):
            print("Errore o interruzione manuale. Tentativo di riconnessione...")
            sock.close()
            sock = None
            while sock is None:
                print("Tentativo di riconnessione al server...")
                sock = connect_to_server()
                time.sleep(5)

except KeyboardInterrupt:
    print("Interruzione manuale ricevuta.")
except Exception as e:
    print(f"Errore inaspettato: {e}")
finally:
    if sock is not None:
        sock.close()
        print("Connessione al server TCP chiusa.")
