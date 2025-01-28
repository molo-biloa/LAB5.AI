# Install required libraries
!pip install pandas matplotlib scikit-learn tensorflow gradio openpyxl
# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import gradio as gr

# Step 1: Data Preprocessing
# Function to load and preprocess the dataset
def preprocess_data(filepath):
    data = pd.read_excel(filepath)
    
    # Ensure correct data structure
    if 'Traffic Flow' not in data.columns or 'Timestamp' not in data.columns:
        return "The dataset must include 'Traffic Flow' and 'Timestamp' columns.", None
    
    # Fill missing values
    data.fillna(method='ffill', inplace=True)

    # Create time-based features
    data['Hour'] = pd.to_datetime(data['Timestamp']).dt.hour
    data['Day'] = pd.to_datetime(data['Timestamp']).dt.day
    data['Month'] = pd.to_datetime(data['Timestamp']).dt.month
    data['Weekday'] = pd.to_datetime(data['Timestamp']).dt.weekday

    # Select relevant features
    features = ['Hour', 'Day', 'Month', 'Weekday']
    target = 'Traffic Flow'

    # Normalize features
    scaler = MinMaxScaler()
    data[features] = scaler.fit_transform(data[features])
    
    return data[features], data[target]

# Function to build and train the model
def build_and_train_model(features, target):
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32, verbose=0)
    
    # Plot training history
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Model Training Performance')
    plt.savefig('training_plot.png')  # Save plot to file
    
    return model, 'training_plot.png'

# Gradio interface function
def train_model(filepath):
    try:
        features, target = preprocess_data(filepath)
        if isinstance(features, str):  # Error message returned
            return features, None
        
        model, plot_path = build_and_train_model(features, target)
        model.save('traffic_model.h5')  # Save the trained model
        return "Model trained and saved successfully!", plot_path
    except Exception as e:
        return str(e), None

# Set up Gradio interface
gr_interface = gr.Interface(
    fn=train_model,
    inputs=gr.File(label="Upload Excel Dataset"),
    outputs=[
        gr.Textbox(label="Status"),
        gr.Image(label="Training Plot")
    ],
    title="Traffic Flow Prediction Tool",
    description="Upload a traffic dataset to train a neural network for traffic flow prediction."
)

# Launch the Gradio app
gr_interface.launch()
