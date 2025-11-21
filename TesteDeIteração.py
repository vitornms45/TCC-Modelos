import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from ultralytics import YOLO
import torch
from PIL import Image

keras_model_path = 'modeloFinalNaoAguentoMaisKeras.h5'
yolo_model_path = 'Modelo_Yolov11_Improve_Final.pt'
test_dir = 'test'
base_dir = r"C:\Users\vitor\Documents\TCC\KerasVSYolo11\runs"

IMG_SIZE = (224, 224)
BATCH_SIZE = 1
N_EXECUCOES = 100

datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

class_names = sorted(os.listdir(test_dir))
label_map = {name: idx for idx, name in enumerate(class_names)}
reverse_label_map = {v: k for k, v in label_map.items()}

print("Label map:", label_map)
print("Reverse label map:", reverse_label_map)

img_paths, labels = [], []
for cls_name in class_names:
    cls_dir = os.path.join(test_dir, cls_name)
    img_names = os.listdir(cls_dir)
    img_paths.extend([os.path.join(cls_dir, name) for name in img_names])
    labels.extend([label_map[cls_name]] * len(img_names))

img_paths, labels = np.array(img_paths), np.array(labels)

keras_model = load_model(keras_model_path)
yolo_model = YOLO(yolo_model_path)

def init_results_dict():
    return {'Acurácia': [], 'Precisão': [], 'Recall': [], 'F1-score': [], 'AUC': [], 'Tempo Inferência (s)': []}

keras_results = init_results_dict()
yolo_results = init_results_dict()

def calcular_metricas(model_preds, true_labels, model_name):
    accuracy = accuracy_score(true_labels, model_preds)
    precision = precision_score(true_labels, model_preds, zero_division=0)
    recall = recall_score(true_labels, model_preds, zero_division=0)
    f1 = f1_score(true_labels, model_preds, zero_division=0)
    auc = roc_auc_score(true_labels, model_preds)

    print(f"{model_name} - Acurácia: {accuracy:.4f}, Precisão: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}, AUC: {auc:.4f}")

    return accuracy, precision, recall, f1, auc

def aplicar_transformacoes(imagem, datagen):
    img_array = np.array(imagem)
    img_array = img_array.reshape((1, *img_array.shape))
    img_aumentada = next(datagen.flow(img_array, batch_size=1))[0]
    img_aumentada = (img_aumentada * 255).astype(np.uint8)
    return img_aumentada

for rodada in range(N_EXECUCOES):
    print(f"\n🔁 Rodada {rodada + 1}/{N_EXECUCOES}")

    indices = np.random.permutation(len(img_paths))
    shuffled_img_paths = img_paths[indices]
    shuffled_labels = labels[indices]

    print(f"\n🔀 Primeiras 5 imagens após shuffle (Rodada {rodada + 1}):")
    for i in range(5):
        print(f"{shuffled_img_paths[i]} -> {shuffled_labels[i]}")

    temp_df = pd.DataFrame({
        'filename': shuffled_img_paths,
        'class': shuffled_labels
    })

    keras_generator = datagen.flow_from_dataframe(
        dataframe=temp_df,
        x_col='filename',
        y_col='class',
        target_size=IMG_SIZE,
        class_mode='raw',
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    start = time.time()
    keras_preds = keras_model.predict(keras_generator, verbose=0)
    elapsed = time.time() - start

    keras_pred_classes = (keras_preds > 0.5).astype(int).flatten()

    keras_accuracy, keras_precision, keras_recall, keras_f1, keras_auc = calcular_metricas(keras_pred_classes, shuffled_labels, "Keras")

    keras_results['Acurácia'].append(keras_accuracy)
    keras_results['Precisão'].append(keras_precision)
    keras_results['Recall'].append(keras_recall)
    keras_results['F1-score'].append(keras_f1)
    keras_results['AUC'].append(keras_auc)
    keras_results['Tempo Inferência (s)'].append(elapsed)

    yolo_preds, yolo_probs = [], []
    start = time.time()

    for path in shuffled_img_paths:
        img = Image.open(path)
        img = img.resize(IMG_SIZE) 

        augmented_img = aplicar_transformacoes(img, datagen)

        result = yolo_model(augmented_img, imgsz=224, verbose=False)[0]
        cls = int(torch.argmax(result.probs.data).item())
        prob = float(result.probs.data[1])
        yolo_preds.append(cls)
        yolo_probs.append(prob)

    elapsed = time.time() - start

    yolo_accuracy, yolo_precision, yolo_recall, yolo_f1, yolo_auc = calcular_metricas(yolo_preds, shuffled_labels, "YOLOv11")

    yolo_results['Acurácia'].append(yolo_accuracy)
    yolo_results['Precisão'].append(yolo_precision)
    yolo_results['Recall'].append(yolo_recall)
    yolo_results['F1-score'].append(yolo_f1)
    yolo_results['AUC'].append(yolo_auc)
    yolo_results['Tempo Inferência (s)'].append(elapsed)

df_keras = pd.DataFrame(keras_results)
df_yolo = pd.DataFrame(yolo_results)

col_order = ['Acurácia', 'Precisão', 'Recall', 'F1-score', 'AUC', 'Tempo Inferência (s)']
df_keras = df_keras[col_order]
df_yolo = df_yolo[col_order]

output_path = os.path.join(base_dir, 'Resultados_100_Rodadas_Keras_YOLO_com_Recall_Precisao.xlsx')
with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
    df_keras.to_excel(writer, sheet_name='Keras', index=False)
    df_yolo.to_excel(writer, sheet_name='YOLOv11', index=False)

print(f"\n✅ Arquivo Excel gerado com sucesso: {output_path}")