# converter.py
import tensorflow as tf
from tensorflow.keras.models import load_model
from ultralytics import YOLO
import tf2onnx
import torch
import os


output_dir = "Models"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Diretório '{output_dir}' criado.")


print("\n--- Convertendo o modelo de Classificação de Topografia (MobileNetV2) ---")
try:
   
    keras_topo_model_path = "topografia_classifier_mobilenetv2.h5"
    onnx_topo_output_path = os.path.join(output_dir, "Modelo_Topografia_Classifier.onnx")

    print(f"Carregando modelo de: {keras_topo_model_path}")
    keras_topo_model = load_model(keras_topo_model_path)

  
    onnx_model_topo, _ = tf2onnx.convert.from_keras(keras_topo_model, opset=15)

    with open(onnx_topo_output_path, "wb") as f:
        f.write(onnx_model_topo.SerializeToString())
    print(f"Modelo MobileNetV2 convertido com sucesso para: {onnx_topo_output_path}")

except FileNotFoundError:
    print(f"ERRO: O arquivo '{keras_topo_model_path}' não foi encontrado. Pule esta conversão.")
except Exception as e:
    print(f"Ocorreu um erro inesperado durante a conversão do MobileNetV2: {e}")



print("\n--- Convertendo o modelo Keras original (Ceratocone) ---")
try:

    keras_original_model_path = "modeloFinalNaoAguentoMaisKeras.h5"
    onnx_original_output_path = os.path.join(output_dir, "Modelo_Keras_Ceratocone.onnx")

    print(f"Carregando modelo de: {keras_original_model_path}")
    keras_original_model = load_model(keras_original_model_path)

    onnx_model_original, _ = tf2onnx.convert.from_keras(keras_original_model, opset=15)

    with open(onnx_original_output_path, "wb") as f:
        f.write(onnx_model_original.SerializeToString())
    print(f"Modelo Keras original convertido com sucesso para: {onnx_original_output_path}")

except FileNotFoundError:
    print(f"ERRO: O arquivo '{keras_original_model_path}' não foi encontrado. Pule esta conversão.")
except Exception as e:
    print(f"Ocorreu um erro inesperado durante a conversão do Keras original: {e}")

print("\n--- Convertendo o modelo YOLO ---")
try:
    yolo_model_path = "Modelo_Yolov11_Improve_Final.pt"
    
    print(f"Carregando modelo de: {yolo_model_path}")
    yolo_model = YOLO(yolo_model_path)

    yolo_model.export(format="onnx", imgsz=224, opset=15)
    print("Modelo YOLO convertido com sucesso! Verifique a pasta para o arquivo .onnx")

except FileNotFoundError:
     print(f"ERRO: O arquivo '{yolo_model_path}' não foi encontrado. Pule esta conversão.")
except Exception as e:
    print(f"Ocorreu um erro inesperado durante a conversão do YOLO: {e}")