import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


# Defina o diretório do seu dataset
train_dir = r'C:\Users\vitor\Documents\TCC\KerasVSYolo11\train'
val_dir = r'C:\Users\vitor\Documents\TCC\KerasVSYolo11\val'

# Usando ImageDataGenerator para carregar as imagens e realizar pré-processamento
train_datagen = ImageDataGenerator(
    rescale=1./255,          # Normaliza os valores dos pixels entre 0 e 1
    rotation_range=20,       # Rotaciona as imagens aleatoriamente
    width_shift_range=0.2,   # Translação horizontal aleatória
    height_shift_range=0.2,  # Translação vertical aleatória
    shear_range=0.2,         # Shear transformations
    zoom_range=0.2,          # Zoom aleatório
    horizontal_flip=True,    # Flip horizontal aleatório
    fill_mode='nearest'      # Preenchimento de pixels durante transformação
)

val_datagen = ImageDataGenerator(
    rescale=1./255          # Apenas normalização para o conjunto de validação
)

# Carregamento das imagens e definição das configurações de treino e validação
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224), 
    batch_size=32,
    class_mode='binary',      # Como é um problema de 2 classes (keratoconus e normal)
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
)

# Criando o modelo sequencial
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # Sigmoid para classificação binária
])

# Compilando o modelo
model.compile(optimizer='adam',
              loss='binary_crossentropy',  # Usando a perda binária para classificação de 2 classes
              metrics=['accuracy'])

# Treinando o modelo
model_checkpoint_callback = ModelCheckpoint(
    filepath='best_model_keras.h5',  # Nome do arquivo para salvar o melhor modelo
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True) # Essencial: salva apenas se for o melhor até agora

# Callback para parar o treinamento se não houver melhora
early_stopping_callback = EarlyStopping(
    monitor='val_loss', # Monitora a perda na validação
    patience=5,         # Número de épocas sem melhora antes de parar
    restore_best_weights=True) # Restaura os pesos do melhor modelo ao final

# Treinando o modelo COM os callbacks
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=50,  # Pode aumentar o número de épocas, o EarlyStopping vai parar se necessário
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    callbacks=[model_checkpoint_callback, early_stopping_callback] # Adiciona os callbacks aqui
)

# Exibindo as métricas de validação e acurácia
print("Acurácia de treino: ", history.history['accuracy'][-1])
print("Acurácia de validação: ", history.history['val_accuracy'][-1])
print("Perda de treino: ", history.history['loss'][-1])
print("Perda de validação: ", history.history['val_loss'][-1])

# Plotando a acurácia e a perda durante o treinamento
plt.figure(figsize=(12, 6))

# Plotando a acurácia
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Acurácia de Treinamento')
plt.plot(history.history['val_accuracy'], label='Acurácia de Validação')
plt.title('Acurácia durante o Treinamento')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()

# Plotando a perda
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Perda de Treinamento')
plt.plot(history.history['val_loss'], label='Perda de Validação')
plt.title('Perda durante o Treinamento')
plt.xlabel('Épocas')
plt.ylabel('Perda')
plt.legend()

plt.tight_layout()
plt.show()

# Salvando o modelo treinado
model.save('modeloFinalNaoAguentoMaisKeras.h5')
