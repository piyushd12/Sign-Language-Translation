import tensorflow as tf

def convert_to_tflite(model_path, tflite_path):
    """
    Converts a TensorFlow/Keras model to TensorFlow Lite format.

    Args:
        model_path (str): Path to the saved TensorFlow/Keras model.
        tflite_path (str): Path to save the converted TensorFlow Lite model.
    """
    model = tf.keras.models.load_model(model_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS, 
        tf.lite.OpsSet.SELECT_TF_OPS    
    ]

    converter._experimental_lower_tensor_list_ops = False

    tflite_model = converter.convert()

    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

    print(f"Model successfully converted to TensorFlow Lite and saved at {tflite_path}")


if __name__ == "__main__":
    keras_model_path = "/models/SLR.keras" 
    tflite_model_path = "/models/SLR.tflite"  

    convert_to_tflite(keras_model_path, tflite_model_path)