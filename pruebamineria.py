import pandas as pd
import numpy as np
import time
import random

# Simulación de datos ambientales
def simulate_environmental_data(rows=1000):
    np.random.seed(42)
    data = {
        "timestamp": pd.date_range(start="2024-01-01", periods=rows, freq="h"),
        "area_id": np.random.randint(1, 5, size=rows),
        "co2_emissions": np.random.normal(500, 50, size=rows),
        "water_usage": np.random.normal(200, 20, size=rows),
        "waste_generated": np.random.normal(30, 5, size=rows),
    }
    return pd.DataFrame(data)

# Crear el DataFrame simulado
environmental_data = simulate_environmental_data()

# Guardar los datos en un archivo CSV
environmental_data.to_csv("environmental_data.csv", index=False)
print(environmental_data.head())

# Visualización de distribuciones de las métricas
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar datos
data = pd.read_csv("environmental_data.csv")

# Visualización de distribuciones
plt.figure(figsize=(15, 5))
for i, column in enumerate(["co2_emissions", "water_usage", "waste_generated"], 1):
    plt.subplot(1, 3, i)
    sns.histplot(data[column], kde=True, bins=30)
    plt.title(f"Distribución de {column.replace('_', ' ').capitalize()}")
plt.tight_layout()
plt.show()

# Preparar datos para el modelo
from sklearn.model_selection import train_test_split
X = data[["area_id"]]
X["hour"] = pd.to_datetime(data["timestamp"]).dt.hour  # Hora del día como característica
y = data[["co2_emissions", "water_usage", "waste_generated"]]

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Entrenar modelos de Random Forest para cada métrica
models = {}
predictions = {}
for column in y.columns:
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train[column])
    predictions[column] = model.predict(X_test)
    models[column] = model
    print(f"{column} - Error cuadrático medio:", mean_squared_error(y_test[column], predictions[column]))

# Comparar valores reales vs predichos
for column in y.columns:
    plt.figure(figsize=(10, 5))
    plt.plot(y_test[column].values[:50], label="Real")
    plt.plot(predictions[column][:50], label="Predicho")
    plt.title(f"Comparación Real vs Predicho - {column.replace('_', ' ').capitalize()}")
    plt.legend()
    plt.show()

# Definir umbrales
THRESHOLDS = {
    "co2_emissions": 550,  # kg
    "water_usage": 250,    # litros
    "waste_generated": 40  # kg
}

# Función para generar alertas
def check_alerts(data_row):
    alerts = []
    for metric, threshold in THRESHOLDS.items():
        if data_row[metric] > threshold:
            alerts.append(f"⚠️ ALERTA: {metric.replace('_', ' ').capitalize()} supera el umbral ({data_row[metric]:.2f} > {threshold})")
    return alerts

# Monitoreo en tiempo real
def real_time_monitoring(data):
    print("Iniciando monitoreo en tiempo real...\n")
    for _, row in data.iterrows():
        alerts = check_alerts(row)
        if alerts:
            print(f"[{row['timestamp']}] ALERTAS DETECTADAS:")
            for alert in alerts:
                print(alert)
        time.sleep(0.5)  # Simula un intervalo de tiempo
    print("\nMonitoreo finalizado.")

# Simular datos en tiempo real
subset = data.head(20)  # Simularemos los primeros 20 registros
real_time_monitoring(subset)

# Estrategias de reducción
SCENARIOS = {
    "Eficiencia energética": {"co2_emissions": -10},
    "Reciclaje de agua": {"water_usage": -20},
    "Gestión de residuos": {"waste_generated": -15}
}

# Función para simular escenarios
def simulate_scenarios(data, scenarios):
    results = {}
    for scenario, changes in scenarios.items():
        simulated_data = data.copy()
        for metric, change in changes.items():
            simulated_data[metric] *= (1 + change / 100)
        results[scenario] = simulated_data
    return results

# Ejecutar simulaciones
simulated_results = simulate_scenarios(data, SCENARIOS)

# Visualización de resultados
for scenario, simulated_data in simulated_results.items():
    print(f"\nEscenario: {scenario}")
    print(simulated_data[["co2_emissions", "water_usage", "waste_generated"]].mean())

# Comparar antes y después de los cambios
for scenario, simulated_data in simulated_results.items():
    plt.figure(figsize=(10, 5))
    for metric in THRESHOLDS.keys():
        plt.plot(data[metric].head(50), label=f"Original - {metric}")
        plt.plot(simulated_data[metric].head(50), label=f"{scenario} - {metric}")
    plt.title(f"Impacto de {scenario}")
    plt.legend()
    plt.show()
