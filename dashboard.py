import streamlit as st
import pandas as pd
import json
import paho.mqtt.client as mqtt
import time
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import io  # Para manejar el archivo en memoria

# MQTT Configuration
MQTT_BROKER = "broker.emqx.io"
MQTT_PORT = 1883
MQTT_TOPIC = "mineria/impacto_ambiental"
real_time_data = []

# Callback function for MQTT messages
def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        real_time_data.append(payload)
        st.write(f"Datos recibidos: {payload}")
    except Exception as e:
        st.error(f"Error al procesar el mensaje MQTT: {e}")

# Setup MQTT client
client = mqtt.Client()
client.on_connect = lambda client, userdata, flags, rc: print("Conectado!")
client.on_message = on_message

# Connect to broker
client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.subscribe(MQTT_TOPIC)
client.loop_start()

# Streamlit Dashboard
st.title("Dashboard de Impacto Ambiental")
st.sidebar.header("Opciones")

if real_time_data:
    st.write("Últimos datos recibidos:", real_time_data[-1])
else:
    st.write("No se han recibido datos aún. Realiza una simulación para ver los resultados.")

# Simulación de datos
def simulate_environmental_data(rows=100):
    np.random.seed(42)
    data = {
        "timestamp": pd.date_range(start="2024-01-01", periods=rows, freq="h"),
        "area_id": np.random.randint(1, 5, size=rows),
        "co2_emissions": np.random.normal(500, 50, size=rows),
        "water_usage": np.random.normal(200, 20, size=rows),
        "waste_generated": np.random.normal(30, 5, size=rows),
    }
    return pd.DataFrame(data)

data = simulate_environmental_data()

# Entrenamiento de modelo
X = data[["area_id"]]
y = data[["co2_emissions", "water_usage", "waste_generated"]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {}
for column in y.columns:
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train[column])
    models[column] = model

# Simular predicciones
if st.button("Predecir Métricas Futuras"):
    future_data = pd.DataFrame({"area_id": [1, 2, 3, 4]})
    predictions = {column: models[column].predict(future_data) for column in y.columns}
    st.write("Predicciones de métricas futuras:")
    st.write(pd.DataFrame(predictions))

# Mostrar datos simulados y gráficos
st.subheader("Resultados de la Simulación")
st.write(data)

if st.button("Generar gráfico"):
    st.subheader("Gráficos de las métricas")
    st.line_chart(data[["co2_emissions", "water_usage", "waste_generated"]])

# Generar reporte de análisis
def generate_report(data, predictions):
    report = ""
    report += "Análisis del Impacto Ambiental\n"
    report += "=============================\n\n"
    report += "Resumen de los datos simulados:\n"
    report += f"Promedio de Emisiones de CO2: {data['co2_emissions'].mean():.2f} kg\n"
    report += f"Promedio de Uso de Agua: {data['water_usage'].mean():.2f} litros\n"
    report += f"Promedio de Residuos Generados: {data['waste_generated'].mean():.2f} kg\n\n"
    
    report += "Predicciones Futuras:\n"
    for column in predictions:
        report += f"Predicción para {column}:\n"
        for i, prediction in enumerate(predictions[column]):
            report += f"Area {i + 1}: {prediction:.2f}\n"
        report += "\n"
    
    return report

# Botón de descarga de reporte
if st.button("Generar y Descargar Reporte"):
    predictions = {column: models[column].predict(pd.DataFrame({"area_id": [1, 2, 3, 4]})) for column in y.columns}
    report = generate_report(data, predictions)
    
    # Convertir el reporte en texto a un formato binario
    report_bytes = report.encode("utf-8")
    
    # Botón para descargar el archivo
    st.download_button(
        label="Descargar Reporte",
        data=report_bytes,
        file_name="reporte_impacto_ambiental.txt",
        mime="text/plain"
    )
