import paho.mqtt.client as mqtt
import json
import time
import random

# Configuración de MQTT
MQTT_BROKER = "broker.emqx.io"
MQTT_PORT = 1883
MQTT_TOPIC = "mineria/impacto_ambiental"

# Crear cliente MQTT
client = mqtt.Client()

# Conectar al broker
client.connect(MQTT_BROKER, MQTT_PORT, 60)

# Función para generar datos simulados
def generate_data():
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "co2_emissions": round(random.uniform(450, 600), 2),
        "water_usage": round(random.uniform(150, 250), 2),
        "waste_generated": round(random.uniform(25, 50), 2),
    }

# Publicar datos en un bucle
try:
    while True:
        data = generate_data()
        client.publish(MQTT_TOPIC, json.dumps(data))
        print(f"Publicado: {data}")
        time.sleep(2)  # Publicar cada 2 segundos
except KeyboardInterrupt:
    print("Simulación detenida.")
    client.disconnect()
