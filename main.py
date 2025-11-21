import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import f1_score, roc_auc_score
from ultralytics import YOLO


def plot_confusion(cm, class_names, title='Matriz de Confusão'):
    """Plota uma matriz de confusão de forma legível."""
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='.0f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.show()


BASE_DIR = Path(r'C:\Users\vitor\Documents\TCC\KerasVSYolo11')
CLASSES = ['keratoconus', 'normal']
MODEL_VERSION = "yolo11n-cls.pt" 


print("Analisando a distribuição do dataset...")
counts = {}
for subset in ['train', 'val', 'test']:
    counts[subset] = {}
    for cls in CLASSES:
        class_path = BASE_DIR / subset / cls
        if class_path.exists():
            counts[subset][cls] = len(list(class_path.glob('*.*')))
        else:
            counts[subset][cls] = 0

print("Distribuição de imagens por classe:", counts)

plt.bar(CLASSES, [counts['train'][c] for c in CLASSES], color=['C0', 'C1'])
plt.title("Distribuição de classes (Treino)")
plt.ylabel("Número de imagens")
plt.show()


print(f"\nCarregando modelo pré-treinado: {MODEL_VERSION}")
model_yolo = YOLO(MODEL_VERSION)

print("Iniciando o treinamento do modelo YOLO...")
start_time = time.time()
model_yolo.train(
    data=str(BASE_DIR),
    epochs=3,
    imgsz=224,
    batch=32,
    seed=42,
    flipud=0.5,
    fliplr=0.5,
    degrees=15,
    scale=0.1,
    shear=10,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4
)
yolo_training_time = time.time() - start_time
print(f"Treinamento concluído em {yolo_training_time:.1f} segundos.")

print("\nIniciando avaliação no conjunto de teste...")
val_metrics = model_yolo.val(split='test', batch=16)
acc_yolo = val_metrics.top1
cm_yolo = val_metrics.confusion_matrix.matrix
print(f"Acurácia geral (Top-1) do .val(): {acc_yolo:.4f}")

start_time = time.time()

results = model_yolo.predict(source=str(BASE_DIR / 'test' / '*' / '*.jpg'), stream=False)
yolo_inference_time = time.time() - start_time

y_true_yolo = []
y_pred_yolo = []
y_prob_yolo = []

for r in results:
    true_label_name = Path(r.path).parent.name
    true_label_index = CLASSES.index(true_label_name)
    y_true_yolo.append(true_label_index)

    pred_label_index = r.probs.top1
    y_pred_yolo.append(pred_label_index)
    
    prob_normal = r.probs.data[1].item() 
    y_prob_yolo.append(prob_normal)

f1_yolo = f1_score(y_true_yolo, y_pred_yolo)
auc_yolo = roc_auc_score(y_true_yolo, y_prob_yolo)

print("\n--- Resultados Detalhados da Avaliação ---")
print("Matriz de Confusão:\n", cm_yolo)
print(f"Acurácia (calculada via .val): {acc_yolo:.4f}")
print(f"F1-score (calculado via .predict): {f1_yolo:.4f}")
print(f"AUC (calculada via .predict): {auc_yolo:.4f}")
print(f"Tempo de treino: {yolo_training_time:.1f}s")
print(f"Tempo de Inferência no Teste (.predict): {yolo_inference_time:.1f}s")


plot_confusion(cm_yolo, class_names=CLASSES, title='Matriz de Confusão - YOLO')

results_df = pd.DataFrame({
    'Modelo': [MODEL_VERSION],
    'Acurácia': [acc_yolo],
    'F1-score': [f1_yolo],
    'AUC': [auc_yolo],
    'Tempo Treino (s)': [yolo_training_time],
    'Tempo Inferência (s)': [yolo_inference_time]
})

print("\n--- Resumo do Desempenho ---")
print(results_df)

results_df.plot(x='Modelo', y=['Acurácia', 'F1-score', 'AUC'], kind='bar', figsize=(10, 6), legend=True)
plt.title(f'Desempenho do Modelo {MODEL_VERSION}')
plt.ylabel('Métricas')
plt.xticks(rotation=0)
plt.show()


model_save_path = f"Modelo_{MODEL_VERSION.replace('.pt', '')}_Final.pt"
model_yolo.export(format='torchscript') 
print(f"\nModelo final salvo em: runs/classify/train/weights/best.pt (e exportado como .torchscript)")