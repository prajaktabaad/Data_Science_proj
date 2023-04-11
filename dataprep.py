import os
import cv2
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
# Define the path to the sign language dataset
data_path = 'D:/Books_6thsem/Data science/Project/Numberdetection2/data/train'

# Define a list of classes
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# Initialize empty arrays to store the data and labels
data = []
labels = []

# Loop over the classes and load the images
for i, c in enumerate(classes):
    class_path = os.path.join(data_path, c)
    for file_name in os.listdir(class_path):
        if file_name.endswith('.jpg'):
            # Load the image and preprocess it
            image_path = os.path.join(class_path, file_name)
            image = Image.open(image_path)
            image = image.convert('L')
            image = image.resize((64, 64))
            image = np.array(image).flatten()

            # Append the data and label to the arrays
            data.append(image)
            labels.append(i)

# Convert the data and labels to NumPy arrays and save them
data = np.array(data)
labels = np.array(labels)
np.save('signs_data.npy', data)
np.save('signs_labels.npy',labels)


# Load the dataset
data = np.load('signs_data.npy')
labels = np.load('signs_labels.npy')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Train an SVM model on the training set
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Evaluate the model on the testing set
accuracy = clf.score(X_test, y_test)
print('Accuracy:', accuracy)

# Make predictions on new data
# You can replace this with your own image data

image2 = Image.open('D:/Books_6thsem/Data science/Project/Numberdetection2/data/test/9/1.jpg')
image2 = image2.convert('L')
image2 = image2.resize((64, 64))
flattened_image2 = np.array(image2).flatten().reshape(1,-1)
# Make predictions on the image
prediction = clf.predict(flattened_image2)
print('Prediction:',prediction)