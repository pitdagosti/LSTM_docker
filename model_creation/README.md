# Temperature Prediction using LSTM

This repository contains the code to build and train an LSTM (Long Short-Term Memory) neural network to predict future temperatures based on past sensor data from multiple nodes.

## Table of Contents

- [Overview](#overview)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Training and Evaluation](#training-and-evaluation)
- [Model Conversion to TensorFlow Lite](#model-conversion-to-tensorflow-lite)
- [How to Run](#how-to-run)
- [Dependencies](#dependencies)

## Overview

This project focuses on predicting temperature values from sensor data collected across multiple nodes using an LSTM neural network. The dataset contains time series data from various sensor nodes, and the LSTM model is trained to predict future temperatures based on historical data.

## Data Preprocessing

1. **Reading Data**: The data is loaded from a text file (`0_report_total.txt`), which contains temperature measurements from 50 different sensor nodes. The data is processed and stored in a Pandas DataFrame for further analysis.

2. **Handling Missing Data**: 
    - Temperatures recorded as `0.0` are considered invalid and are removed.
    - Missing values and zeros are replaced with the mean value of the respective columns to ensure continuous data.

3. **Data Splitting**:
    - The data is split into training, validation, and test sets using an 80-10-10 ratio. 
    - The train, validation, and test sets are normalized using `MinMaxScaler` to scale values between 0 and 1.

4. **Sequence Generation**: 
    - Sequences of 20 time steps (sliding window) are generated for each node. The LSTM is trained on these sequences to predict the temperature for the next hour.

## Model Architecture

The LSTM model is built using the following layers:
- An `LSTM` layer with 128 units, which processes the time series input.
- A `Dense` layer with a single output, which predicts the temperature for the next hour.

The input shape is `(n_steps, n_features)`, where:
- `n_steps` is the number of previous time steps used (20 in this case).
- `n_features` is the number of nodes (sensor readings) included in the input data.

The model uses the **mean squared error (MSE)** loss function and the **Adam** optimizer.

## Training and Evaluation

1. **Training**: 
    - The model is trained for 20 epochs with a batch size of 30.
    - Early stopping is used to prevent overfitting, with a patience of 5 epochs.

2. **Evaluation**: 
    - The test set is used to evaluate the performance of the model, and the root mean squared error (RMSE) is calculated to assess accuracy.
    
## Model Conversion to TensorFlow Lite

Once the LSTM model is trained, it is converted into a TensorFlow Lite model for deployment on edge devices:

1. The model is saved using the TensorFlow SavedModel format.
2. The SavedModel is converted to TensorFlow Lite format using `TFLiteConverter`.
3. A test inference is run using the TensorFlow Lite interpreter to verify the model conversion and check its predictions.

## How to Run

1. **Install the dependencies**: Make sure you have all the required libraries installed. See the [Dependencies](#dependencies) section for details.

2. **Preprocess the data**:
    - Download the sensor data and place it in the `../dati/50_sensors/` directory.
    - Run the data preprocessing and visualization steps to ensure the data is ready for training.

3. **Train the model**:
    - Run the LSTM model training using the provided code.
    - You can modify the hyperparameters (batch size, number of epochs, LSTM units) as needed.

4. **Evaluate the model**:
    - After training, the model is evaluated on the test set to compute the RMSE.

5. **Convert the model to TensorFlow Lite**:
    - After training, the model can be converted to TensorFlow Lite format using the provided code.

## Dependencies

- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- TensorFlow

You can install all dependencies using:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
```
