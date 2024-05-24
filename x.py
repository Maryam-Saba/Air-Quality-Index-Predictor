import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier  # Import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
import nltk

# Sample dataset (you can replace this with your own dataset)
aqi_data = pd.DataFrame({
    'Temperature': [20, 25, 30, 35, 40],
    'Humidity': [50, 55, 60, 65, 70],
    'AQI': [50, 45, 40, 35, 30]
})

# Function to predict AQI based on user input and display the graph
def predict_aqi(model_type):
    try:
        # Read user inputs for temperature and humidity
        temperature = float(temperature_entry.get())
        humidity = float(humidity_entry.get())
        
        # Define features (X) and target (y)
        X = aqi_data[['Temperature', 'Humidity']]
        y = aqi_data['AQI']
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Initialize and train the selected model
        if model_type == 'Linear Regression':
            model = LinearRegression()
            model.fit(X_train, y_train)
        elif model_type == 'KNN':
            model = KNeighborsClassifier(n_neighbors=3)
            model.fit(X_train, y_train)
        
        # Predict AQI for the user input
        aqi = model.predict([[temperature, humidity]])
        
        result_label.config(text="Predicted AQI: {:.2f}".format(aqi[0]))

        # Calculate accuracy (mean squared error for Linear Regression)
        y_pred = model.predict(X_test)
        if model_type == 'Linear Regression':
            mse = mean_squared_error(y_test, y_pred)
            accuracy_label.config(text="Model Mean Squared Error: {:.2f}".format(mse))
        elif model_type == 'KNN':
            acc = accuracy_score(y_test, y_pred)
            accuracy_label.config(text="Model Accuracy: {:.2f}".format(acc))

        # Plot the data and the prediction
        plt.figure(figsize=(10, 5))
        plt.scatter(aqi_data['Temperature'], aqi_data['AQI'], color='blue', label='Historical Data')
        plt.scatter([temperature], [aqi], color='red', label='Predicted AQI')
        plt.title('AQI Prediction')
        plt.xlabel('Temperature (°C)')
        plt.ylabel('AQI')
        plt.legend()
        plt.grid(True)
        plt.show()

    except ValueError:
        # Handle case where inputs are not valid numbers
        messagebox.showerror("Input Error", "Please enter valid numeric values for Temperature and Humidity.")
    except Exception as e:
        # Handle any other unexpected errors
        messagebox.showerror("Error", f"An unexpected error occurred: {e}")

# GUI setup
root = tk.Tk()
root.title("Air Quality Index Predictor")
root.geometry("400x350")  # Increased height to accommodate more controls
root.configure(bg="#f0f0f0")

# Styling
style = ttk.Style()
style.configure("TLabel", font=("Arial", 12), background="#f0f0f0")
style.configure("TButton", font=("Arial", 12), padding=10)
style.configure("TEntry", font=("Arial", 12))

# Frame to contain the input fields and buttons
frame = ttk.Frame(root, padding="20 20 20 20")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
frame.columnconfigure(0, weight=1)
frame.rowconfigure(0, weight=1)

# Temperature input field
temperature_label = ttk.Label(frame, text="Enter Temperature (in °C):")
temperature_label.grid(row=0, column=0, padx=10, pady=10, sticky="e")

temperature_entry = ttk.Entry(frame)
temperature_entry.grid(row=0, column=1, padx=10, pady=10)

# Humidity input field
humidity_label = ttk.Label(frame, text="Enter Humidity (in %):")
humidity_label.grid(row=1, column=0, padx=10, pady=10, sticky="e")

humidity_entry = ttk.Entry(frame)
humidity_entry.grid(row=1, column=1, padx=10, pady=10)

# Option to choose the prediction model
model_label = ttk.Label(frame, text="Choose Model:")
model_label.grid(row=2, column=0, padx=10, pady=10, sticky="e")

model_combobox = ttk.Combobox(frame, values=["Linear Regression", "KNN"])
model_combobox.grid(row=2, column=1, padx=10, pady=10)
model_combobox.current(0)  # Set default value to "Linear Regression"

# Button to trigger AQI prediction
predict_button = ttk.Button(frame, text="Predict AQI", command=lambda: predict_aqi(model_combobox.get()))
predict_button.grid(row=3, column=0, columnspan=2, padx=10, pady=10)

# Label to display the predicted AQI
result_label = ttk.Label(frame, text="", font=("Arial", 14, "bold"), foreground="#007BFF")
result_label.grid(row=4, column=0, columnspan=2, padx=10, pady=10)

# Label to display the model accuracy (mean squared error or accuracy)
accuracy_label = ttk.Label(frame, text="", font=("Arial", 12), foreground="#FF5733")
accuracy_label.grid(row=5, column=0, columnspan=2, padx=10, pady=10)

# Start the GUI event loop
root.mainloop()
