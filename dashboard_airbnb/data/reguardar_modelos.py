from tensorflow import keras

# Re-guardar modelo de recomendación
try:
    model_rec = keras.models.load_model("recommend_model.keras")
    model_rec.save("recommend_model.keras")
    print("Modelo de recomendación re-guardado correctamente.")
except Exception as e:
    print("Error re-guardando modelo de recomendación:", e)

# Re-guardar modelo de precio
try:
    model_price = keras.models.load_model("price_model.keras")
    model_price.save("price_model.keras")
    print("Modelo de precio re-guardado correctamente.")
except Exception as e:
    print("Error re-guardando modelo de precio:", e)
