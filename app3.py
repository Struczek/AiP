import os
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot  as plt
import numpy as np
import cv2
#from tensorflow.keras.utils import to_categorical
from keras import layers , models
from keras.utils.vis_utils import plot_model
#from tf.keras.utils import plot_model
#from skimage import io

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Variables
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
PATH = 'D:\\Studia\\aip\\data\\'
train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')
test_dir = os.path.join(PATH, 'newset')

BATCH_SIZE = 32                                              # 32
IMG_SIZE = (224, 224)

print(train_dir)

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
                                                            
class_names = train_dataset.class_names
print(class_names)
#print(train_dataset.shape)
#print(train_dataset.dtype)


model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=( 224, 224, 3 )))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dropout(0.3))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(6,activation='softmax'))

model.summary()
plot_model(model,to_file='model1.png',show_shapes=True)

'''optimizer = tf.keras.optimizers.Nadam(
    learning_rate=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-07,
    name='Nadam'
) # 0.00001'''
optimizer='rmsprop'


'''lossfn = tf.keras.losses.SparseCategoricalCrossentropy( from_logits=False, reduction=tf.keras.losses.Reduction.AUTO, name='sparse_categorical_crossentropy')
'''
lossfn='categorical_crossentropy'
'''model.compile(loss="sparse_categorical_crossentropy",
    optimizer="sgd",
    metrics=["accuracy"])'''
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])#, tf.keras.metrics.CategoricalAccuracy()])

#img = io.imread(file_path)
img = cv2.imread(".\\baweln1.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.title("Sheep Image")
plt.xlabel("X pixel scaling")
plt.ylabel("Y pixels scaling")
 
plt.imshow(img)
plt.show()

#train_dataset=to_categorical(train_dataset)
history= model.fit(train_dataset,epochs=100,validation_data=(validation_dataset))

model.save('Wlasny 100')

pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()




validation_loss, validation_accuracy = model.evaluate(validation_dataset)
print(f'Validation Accuracy: {validation_accuracy * 100:.2f}%')

test_dataset=tf.keras.utils.image_dataset_from_directory(test_dir,
                                                            shuffle=True,
                                                            batch_size=BATCH_SIZE,
                                                            image_size=IMG_SIZE)

prediction = model.predict(np.array([img]))

while(1):
    path= input()
    try:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        prediction = model.predict(np.array([img]))
        index = np.argmax(prediction)
        print(prediction)
        print('Prediction is ', class_names[index])
        title='Prediction is '+ class_names[index]
        plt.title(title)
        plt.imshow(img)
        plt.show() 
    except:
        print('Błąd')
