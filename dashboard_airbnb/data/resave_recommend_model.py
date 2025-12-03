from tensorflow import keras

# Cargar el modelo original
model = keras.models.load_model("recommend_model.keras")

# Guardarlo de nuevo (sobrescribiendo) con el formato actual
model.save("recommend_model.keras")
print("Modelo de recomendación re-guardado correctamente")
