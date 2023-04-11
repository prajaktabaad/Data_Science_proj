import model
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import matplotlib.pyplot as plt

#BUild CNN - Initializing
from networkx.generators.tests.test_community import test_generator

classifier = Sequential()

# First convolution layer and pooling
classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
# Second convolution layer and pooling
classifier.add(Conv2D(32, (3, 3), activation='relu'))
# input_shape is going to be the pooled feature maps from the previous convolution layer
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Flattening the layers
classifier.add(Flatten())

# Adding a fully connected layer
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=10, activation='softmax')) # softmax for more than 2

# Compiling the CNN
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # categorical_crossentropy for more than 2



#Preparing data and training the model

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('data/train',
                                                 target_size=(64, 64),
                                                 batch_size=5,
                                                 color_mode='grayscale',
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory('data/test',
                                            target_size=(64, 64),
                                            batch_size=5,
                                            color_mode='grayscale',
                                            class_mode='categorical')
classifier.fit_generator(
        training_set,
        steps_per_epoch=600,
        epochs=10,
        validation_data=test_set,
        validation_steps=30) # No of images in test set

# score = classifier.evaluate_generator(test_set)
# print("Test accuracy:",score*100)
# Saving the model and weights
model_json = classifier.to_json()
with open("model-bw.json", "w") as json_file:
    json_file.write(model_json)
classifier.save_weights('model-bw.h5')



# # Evaluate the performance of the CNN on the testing set
# loss, accuracy = model.evaluate_generator(test_generator, test_generator.samples/test_generator.batch_size)
# print('Test accuracy:',accuracy)

# Plot training & validation accuracy values
# plt.plot(classifier.history.history['acc'])
# plt.plot(classifier.history.history['loss'])
# plt.title('Model Accuracy and Loss')
# plt.ylabel('Accuracy/Loss')
# plt.xlabel('Epoch')
# plt.legend(['Accuracy', 'Loss'], loc='upper left')
# plt.show()
# # Plot training & validation loss values
# plt.plot(classifier.history.history['loss'])
# plt.plot(classifier.history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()