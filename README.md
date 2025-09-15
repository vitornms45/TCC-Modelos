Beleza, Victor 👌
Com base no **documento que você me enviou** (Entrega 1) e no que já conversamos sobre o repositório, montei um **README.md acadêmico**, que reflete tanto a parte técnica (Keras vs YOLO) quanto o contexto do seu TCC (plataforma, métricas, resultados).

---

# TCC-Modelos 🧠👁️

Este repositório reúne os experimentos, scripts e modelos desenvolvidos para o Trabalho de Conclusão de Curso:

**“Desenvolvimento de uma Plataforma Digital para Diagnóstico de Ceratocone com Visão Computacional e Análise Comparativa de Modelos”**

📍 Faculdade Impacta – São Paulo, 2025
👨‍🏫 Orientador: Prof. Me. Gilberto Alves Pereira

---

## 🎯 Objetivo

* Desenvolver uma **plataforma computacional** para auxiliar no diagnóstico automatizado de **ceratocone**, doença ocular que afeta a córnea.
* Avaliar comparativamente diferentes arquiteturas de **deep learning** para classificação de imagens oftalmológicas.
* Implementar **CNNs customizadas em Keras** e **YOLOv8/YOLOv11-cls (Ultralytics)**, usando técnicas de **transfer learning**.
* Validar os modelos com métricas padrão da área médica: **Acurácia, Precisão, Recall, F1-Score e AUC-ROC**.

---

## 📂 Estrutura do Repositório

```text
TCC-Modelos/
│
├── Models/                     # Modelos finais e intermediários (.h5, .pt, .onnx)
├── dataset_éumPredioOuUmOlho/  # Dataset auxiliar para testes
├── train/                      # Dados de treino
├── val/                        # Dados de validação
├── test/                       # Dados de teste
├── runs/                       # Logs e métricas do YOLO
├── configs/                    
│   ├── data.yaml               # Configuração do dataset YOLO
│   └── yolo_config.yaml        # Configuração de hiperparâmetros
├── scripts/
│   ├── main.py                 # Treino YOLO
│   ├── main-keras.py           # Treino CNN (Keras/EfficientNetB0)
│   ├── SizeDown.py             # Pré-processamento (redimensionamento)
│   ├── TesteDeIteração.py      # Testes comparativos
│   └── ModeloParaReconhecimentoTopografico.py
└── README.md
```

---

## ⚙️ Tecnologias e Dependências

* **Python 3.10+**
* **TensorFlow / Keras 2.12.2**
* **PyTorch 2.7.2**
* **Ultralytics 8.3.01 (YOLOv8/YOLOv11)**
* **scikit-learn 1.7.2**
* **pandas 2.3.2**
* **numpy 1.23.5**
* **matplotlib 3.10.2 / seaborn 0.13.2**
* **Pillow 11.3.2**
* **openpyxl 3.1.2 / xlsxwriter 3.2.2**
* **Flask 3.1.2** (para integração da plataforma OFTSYS)
* **python-dotenv 1.1.2**

---

## 🚀 Como Executar

### 1. Clone o repositório

```bash
git clone https://github.com/vitornms45/TCC-Modelos.git
cd TCC-Modelos
```

### 2. Crie o ambiente virtual

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4. Treinamento

#### CNN (Keras / EfficientNetB0)

```bash
python scripts/main-keras.py
```

#### YOLOv8 / YOLOv11 (Ultralytics)

```bash
python scripts/main.py --config configs/data.yaml
```

---

## 📊 Resultados Obtidos

Comparação entre CNN customizada em Keras e YOLOv11-cls:

| Modelo        | Acurácia  | Precisão | Recall (Sens.) | F1-Score   | AUC      | Tempo de Inferência |
| ------------- | --------- | -------- | -------------- | ---------- | -------- | ------------------- |
| **Keras CNN** | **90.0%** | \~88%    | **94.9%**      | **90.47%** | **0.90** | **4.38s/lote**      |
| YOLOv11-cls   | 85.56%    | \~87%    | 85.48%         | 86%        | 0.85     | 6.21s/lote          |

📌 **Destaques:**

* O **modelo Keras CNN** teve desempenho superior em quase todas as métricas.
* O **Recall mais alto** (94.9%) é essencial para reduzir falsos negativos no diagnóstico.
* O Keras foi também **29.6% mais rápido** na inferência em relação ao YOLOv11.

---

## 🖥️ Plataforma OFTSYS

Além dos modelos, foi desenvolvida a plataforma **OFTSYS**, que integra:

* oftsys.onrender.com
* Dashboard interativo para visualização de métricas.
* Chatbot especializado em ceratocone.
* Módulo de análise de exames oftalmológicos.
* Landing page institucional.

---

## 📌 Conclusões

* A CNN customizada em **Keras** se mostrou mais eficaz para **detecção de ceratocone**, principalmente pela sensibilidade elevada e pela eficiência computacional.
* O **YOLOv11** trouxe robustez e facilidade de uso (transfer learning), mas com desempenho inferior.
* Ambos os modelos se mostraram estáveis (baixo desvio padrão nas execuções).

🔮 **Trabalhos futuros:**

* Expandir a base de dados (diferentes dispositivos e populações).
* Avaliar arquiteturas mais avançadas (EfficientNetV2, Vision Transformers).
* Validar os modelos em ambiente clínico real em parceria com instituições médicas.
* Ampliar a plataforma para diagnóstico de **glaucoma, degeneração macular** e outras doenças oculares.

---

## 📜 Licença

Este projeto é de uso acadêmico, licenciado sob a **MIT License**.

---
