import os
import tensorflow as tf
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

PATH = 'D:\\Studia\\aip\\data\\'
train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')
test_dir=os.path.join(PATH, 'test')

BATCH_SIZE = 32
IMG_SIZE = (224, 224)

# Tworzenie zbioru danych walidacyjnych
validation_dataset = tf.keras.utils.image_dataset_from_directory(validation_dir,
                                                                 shuffle=False,
                                                                 batch_size=BATCH_SIZE,
                                                                 image_size=IMG_SIZE,
                                                                 label_mode='categorical')
train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir,
                                                            shuffle=True,
                                                            batch_size=BATCH_SIZE,
                                                            image_size=IMG_SIZE,
                                                            label_mode='categorical')
test_dataset = tf.keras.utils.image_dataset_from_directory(test_dir,
                                                            shuffle=True,
                                                            batch_size=BATCH_SIZE,
                                                            image_size=IMG_SIZE,
                                                            label_mode='categorical')

# Przygotowanie modelu ResNet50
base_model = ResNet50(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(len(validation_dataset.class_names), activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

y_pred_probs = model.predict(validation_dataset)
y_true = np.concatenate([labels.numpy() for _, labels in validation_dataset])

# Convert probabilities to class indices
y_pred_classes = y_pred_probs.argmax(axis=1)
y_true_classes = y_true.argmax(axis=1)

precision = precision_score(y_true_classes, y_pred_classes, average='weighted')
recall = recall_score(y_true_classes, y_pred_classes, average='weighted')
f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')

# Ewaluacja modelu na danych walidacyjnych
validation_loss, validation_accuracy = model.evaluate(test_dataset)
print(f'Validation Accuracy: {validation_accuracy * 100:.2f}%')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')
