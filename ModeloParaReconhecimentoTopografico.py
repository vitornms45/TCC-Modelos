import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

print("TensorFlow Version:", tf.__version__)


IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 20

DATA_DIR = 'dataset_éumPredioOuUmOlho'
train_dir = os.path.join(DATA_DIR, 'train')
validation_dir = os.path.join(DATA_DIR, 'validation')

train_datagen = ImageDataGenerator(
    rescale=1./255.,         # Normaliza os pixels para o intervalo [0, 1]
    rotation_range=30,       # Rotaciona as imagens aleatoriamente
    width_shift_range=0.2,   # Desloca a largura
    height_shift_range=0.2,  # Desloca a altura
    shear_range=0.2,         # Aplica cisalhamento
    zoom_range=0.2,          # Aplica zoom
    horizontal_flip=True,    # Inverte horizontalmente
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255.)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',  # Essencial para classificação de 2 classes
    shuffle=True
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)


print("Classes encontradas:", train_generator.class_indices)



base_model = MobileNetV2(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
                         include_top=False,  
                         weights='imagenet')

base_model.trainable = False


model = Sequential([
    base_model,
    GlobalAveragePooling2D(), # Converte os mapas de características em um único vetor
    Dropout(0.5),             # Adiciona dropout para regularização e evitar overfitting
    Dense(1, activation='sigmoid') # A nossa camada de saída: 1 neurônio com sigmoid para binário
])

model.summary()

model.compile(optimizer=Adam(learning_rate=0.0001), 
              loss='binary_crossentropy',
              metrics=['accuracy'])


history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE
)


def plot_history(history_obj):
    """Plota as curvas de acurácia e perda do treinamento."""
    acc = history_obj.history['accuracy']
    val_acc = history_obj.history['val_accuracy']
    loss = history_obj.history['loss']
    val_loss = history_obj.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Acurácia de Treino')
    plt.plot(epochs_range, val_acc, label='Acurácia de Validação')
    plt.legend(loc='lower right')
    plt.title('Acurácia de Treino e Validação')
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Perda de Treino')
    plt.plot(epochs_range, val_loss, label='Perda de Validação')
    plt.legend(loc='upper right')
    plt.title('Perda de Treino e Validação')
    plt.xlabel('Épocas')
    plt.ylabel('Perda')
    
    plt.show()

# Plota os gráficos do histórico de treinamento
plot_history(history)

model.save('topografia_classifier_mobilenetv2.h5')
print("\nModelo 'topografia_classifier_mobilenetv2.h5' salvo com sucesso!")