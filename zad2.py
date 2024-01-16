import os
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import pandas as pd
import matplotlib.pyplot  as plt

PATH = 'D:\\Studia\\aip\\data\\'
train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')
test_dir = os.path.join(PATH, 'newset')

BATCH_SIZE = 32
IMG_SIZE = (224, 224)

# Tworzenie zbiorów danych
train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir,
                                                            shuffle=True,
                                                            batch_size=BATCH_SIZE,
                                                            image_size=IMG_SIZE,
                                                            label_mode='categorical')

validation_dataset = tf.keras.utils.image_dataset_from_directory(validation_dir,
                                                                 shuffle=True,
                                                                 batch_size=BATCH_SIZE,
                                                                 image_size=IMG_SIZE,
                                                                 label_mode='categorical')

# Przygotowanie modelu ResNet50
base_model = ResNet50(weights=None, include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(len(train_dataset.class_names), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Zamrażanie wag warstw bazowego modelu
# for layer in base_model.layers:
#     layer.trainable = False

# Kompilacja modelu
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Trenowanie modelu
history = model.fit(train_dataset,
          epochs=30,  # Możesz dostosować liczbę epok
          validation_data=validation_dataset)

model.save('Resnet 30')
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()

# Ewaluacja modelu na danych walidacyjnych
validation_loss, validation_accuracy = model.evaluate(validation_dataset)
print(f'Validation Accuracy: {validation_accuracy * 100:.2f}%')