
# Simulación y Monitoreo de Datos Ambientales

Este proyecto implementa un sistema para simular, visualizar, monitorear y reducir el impacto de los datos ambientales. Utiliza un enfoque basado en modelos de machine learning para predecir métricas clave y generar alertas en tiempo real.

## Tabla de Contenidos

- [Descripción](#descripción)
- [Instalación](#instalación)
- [Uso](#uso)
- [Simulación de Datos](#simulación-de-datos)
- [Visualización](#visualización)
- [Modelos de Predicción](#modelos-de-predicción)
- [Monitoreo en Tiempo Real](#monitoreo-en-tiempo-real)
- [Simulación de Estrategias de Reducción](#simulación-de-estrategias-de-reducción)
- [Contribuciones](#contribuciones)
- [Licencia](#licencia)

## Descripción

Este proyecto genera datos ambientales simulados y utiliza modelos de **Random Forest** para predecir emisiones de CO2, uso de agua y generación de residuos. También incluye funcionalidad para generar alertas en tiempo real y simular estrategias para reducir el impacto ambiental.

## Instalación

Para instalar las dependencias necesarias, ejecuta:

```sh
pip install pandas numpy scikit-learn paho-mqtt matplotlib seaborn
#Uso
#Puedes ejecutar el script principal para simular datos, entrenar modelos, monitorear en tiempo real y visualizar los resultados.

sh
python main.py


#Simulación de Datos
#Genera un DataFrame con datos simulados de emisiones de CO2, uso de agua y generación de residuos:

#python
environmental_data = simulate_environmental_data()
environmental_data.to_csv("environmental_data.csv", index=False)

#Visualización
#Visualiza las distribuciones de las métricas ambientales utilizando matplotlib y seaborn:
#python
plt.figure(figsize=(15, 5))
for i, column in enumerate(["co2_emissions", "water_usage", "waste_generated"], 1):
    plt.subplot(1, 3, i)
    sns.histplot(data[column], kde=True, bins=30)
    plt.title(f"Distribución de {column.replace('_', ' ').capitalize()}")
plt.tight_layout()
plt.show()


#Modelos de Predicción
#Entrena modelos de Random Forest para predecir cada una de las métricas:
#python

for column in y.columns:
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train[column])
    predictions[column] = model.predict(X_test)
    print(f"{column} - Error cuadrático medio:", mean_squared_error(y_test[column], predictions[column]))

#Monitoreo en Tiempo Real
#Genera alertas en tiempo real basadas en umbrales predeterminados:

#python
def real_time_monitoring(data):
    for _, row in data.iterrows():
        alerts = check_alerts(row)
        if alerts:
            for alert in alerts:
                print(alert)
        time.sleep(0.5)  # Simula un intervalo de tiempo

#Simulación de Estrategias de Reducción
#Simula escenarios con estrategias para reducir el impacto ambiental:
#python
def simulate_scenarios(data, scenarios):
    results = {}
    for scenario, changes in scenarios.items():
        simulated_data = data.copy()
        for metric, change in changes.items():
            simulated_data[metric] *= (1 + change / 100)
        results[scenario] = simulated_data
    return results
#Contribuciones
#Las contribuciones son bienvenidas. Siéntete libre de abrir un pull request o reportar problemas.

Licencia
Este proyecto está licenciado bajo los términos de la MIT License.


Puedes adaptar este README según tus necesidades y añadir cualquier información adicional que consideres relevante. ¡Espero que te sea útil! 😊
