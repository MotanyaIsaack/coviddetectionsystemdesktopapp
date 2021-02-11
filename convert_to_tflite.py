import tensorflow as tf


model=tf.keras.models.load_model('C:/xampp/htdocs/School/year-four/final-year-project/SMART-CT-SCAN_BASED-COVID19_VIRUS_DETECTOR/covid_final_model.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.experimental_new_converter = True
tflite_model = converter.convert()
open("Main_model.tflite", "wb").write(tflite_model)