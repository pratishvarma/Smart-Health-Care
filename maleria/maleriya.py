# Importing the Keras libraries and packages

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D , MaxPool2D , Flatten,Dense,Dropout
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# re-size all the images to this
IMAGE_SIZE = [224,224]

train_path = 'cell_images/train'
valid_path = 'cell_images/test'





train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('cell_images/Train',
                                                 target_size = (224,224),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('cell_images/Test',
                                            target_size = (224,224),
                                            batch_size = 32,
                                            class_mode = 'binary')

model  = Sequential()
model.add(Conv2D(16,(3,3),input_shape = (224,224,3) , activation = 'relu'))
model.add(MaxPool2D(2,2))
model.add(Dropout(0.2))
model.add(Conv2D(32,(3,3),activation = 'relu'))
model.add(MaxPool2D(2,2))
model.add(Dropout(0.3))

model.add(Flatten())

model.add(Dense(64,activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation = 'sigmoid'))


# view the structure of the model
model.summary()

model.compile(optimizer = 'adam', loss = 'binary_crossentropy' , metrics = ['accuracy'])



# fit the model
r = model.fit(
  training_set,
  validation_data=test_set,
  epochs=5,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)

# loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()

# accuracies
plt.plot(r.history['accuracy'], label='train accuracy')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()



model.save('model_maleria.h5')






