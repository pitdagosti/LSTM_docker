# LSTM Temperature Prediction with TFLite

## Overview

This project implements a **Long Short-Term Memory (LSTM)** model, converted to TensorFlow Lite (TFLite) format, to predict future temperature values based on real-time data received over a TCP connection. Using a sliding window approach, the model uses past temperature data to generate future predictions. The results are then saved in a CSV file.

## How It Works

1. **Data Collection**: The script listens for incoming temperature data via a TCP socket connection. Each node sends temperature readings and labels in the format `node_name;temperature_label;temperature_value`.
   
2. **Preprocessing**:
   - Each node's temperature data is normalized using a **MinMaxScaler**.
   - Data is stored in sliding windows containing a fixed number of past temperature readings (20 time steps).

3. **LSTM Model**:
   - A pre-trained LSTM model is loaded to perform predictions in **TFLite** format.
   - For each node, once enough data is collected, the model predicts the next temperature value based on the sliding window of past temperatures.

4. **Prediction Output**:
   - The predicted temperature is unscaled and saved to a CSV file along with the actual temperature, label, node ID, and timestamp.

## Model Architecture

- The model used is an LSTM network trained on historical temperature data, which is then converted to TFLite format for optimized inference on edge devices or limited-resource environments.
- 
<img width="662" alt="Screenshot 2024-09-26 alle 16 51 20" src="https://github.com/user-attachments/assets/bdb6aa85-8d7f-4c33-8d39-641e9f294242">

```python
# Example of how the LSTM model might be structured
model = Sequential()
model.add(LSTM(128, input_shape=(20,1)))
model.add(Dense(1))  # Output layer for predicting the temperature
```

## Requirements

To run the script, you will need the following dependencies installed:

- **Python 3.x**
- **NumPy** for numerical operations (`numpy==1.26.3`)
- **Pandas** for data manipulation (`pandas==2.1.4`)
- **h5py** for handling HDF5 files (`h5py==3.10.0`)
- **ujson** for fast JSON processing (`ujson==5.4.0`)
- **TensorFlow** for TFLite model inference (`tensorflow==2.16.2`)
- **Scikit-learn** for data preprocessing (`scikit-learn==1.2.2`)

You can install all dependencies with:

```bash
pip install numpy==1.26.3 pandas==2.1.4 h5py==3.10.0 ujson==5.4.0 tensorflow==2.16.2 scikit-learn==1.2.2
```

## Usage

This script is designed to run in a containerized environment alongside another container responsible for reading, sending data, and creating the TCP server. The pre-built Docker image is available at `pit836/lstm:latest`, and the docker-compose.yml file can spin up the container that performs these operations. All prediction results are saved in a CSV file (`temperature_predictions.csv`), located in the shared volume `dbFiles`, mapped to the directory: `/dbFiles:/usr/src/app/dist/dbFiles`.

### Step 1: Load and Prepare the Model
Ensure the LSTM model in TFLite format (`lstm.tflite`) is in the working directory. The script will load this model using the TensorFlow Lite interpreter.

### Step 2: Connect to TCP Server
The script establishes a TCP connection to receive real-time data from nodes.

1. Set the correct `TCP_IP` and `TCP_PORT` in the script.
2. The incoming data should be in the format:
   ```
   node_name;temperature_label;temperature_value
   ```

### Step 3: Run the Script
To start the script, run:

```bash
python temperature_prediction.py
```

This will:
- Continuously listen for data from the TCP server.
- Perform predictions using the LSTM model whenever enough data is collected for a node.
- Save the predictions and other relevant data into a CSV file.

### Input Data Example

The incoming data is expected to follow this format:

```
NODE_01;T1;22.5
NODE_02;T0;19.8
...
```

### Output Format

The predicted temperatures are saved in a CSV file (`temperature_predictions.csv`) with the following columns:

- **ID**: Unique identifier for each node's prediction.
- **Node_Name**: Name of the node that sent the data.
- **Temperature_Actual**: Actual temperature value received.
- **Temperature_Future**: Predicted future temperature.
- **Label**: Temperature label (e.g., "T_actual").
- **Timestamp**: The timestamp of the prediction.

Example of output in CSV format:

```
ID, Node_Name, Temperature_Actual, Temperature_Future, Label, Timestamp
1, NODE_01, 22.5, 23.1, T1, 1701234567
2, NODE_02, 19.8, 20.3, T0, 1701234570
```

### CSV Output File Location

By default, the predictions are saved in the following path:
```
/usr/src/app/dist/dbFiles/temperature_predictions.csv
```

## Limitations

- **Model Training**: The accuracy of the predictions depends on the quality of the pre-trained LSTM model. The current script only performs inference with a TFLite model and does not retrain it.
- **Real-Time Data**: The script assumes continuous real-time data input from multiple nodes. If data transmission stops, the script will wait for new data.
- **TCP Connection**: The TCP server and client setup must be configured to ensure data is sent and received correctly.

## Results

The container works with a usage of a maximum of 20% of CPU and 260 MB of memory.

<img width="1390" alt="Screenshot 2024-09-30 alle 16 08 26" src="https://github.com/user-attachments/assets/751b4e13-b03e-44cc-94ac-e92cc890ac19">

**The predictions are quite precise, with a RMSE of ????**


## License

This project is licensed under the MIT License.

## Contributing

Feel free to open an issue or submit a pull request if you'd like to contribute to the project.
