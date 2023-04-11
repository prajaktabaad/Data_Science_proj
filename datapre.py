import numpy as np
import os
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
data = []
labels = []
images = []
datadir = 'D:/Books_6thsem/Data science/Project/Numberdetection2/data/train'
for sign in os.listdir(datadir):
    for filename in os.listdir(datadir + '/' + sign):
        # Load the image
        image = Image.open(datadir + '/' + sign + '/' + filename)

        # Convert the image to grayscale and resize it
        image = image.convert('L')
        image = image.resize((64, 64))

        # Flatten the pixel values into a 1D array
        flattened_image = np.array(image).flatten()

        # Add the flattened image to the data list
        data.append(flattened_image)

        # Add the label to the labels list
        labels.append(sign)

data = np.array(data)
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train,y_train)

accuracy = clf.score(X_test, y_test)
print(f'Accuracy:{accuracy}')


# new_data = np.random.rand(10, 4096)  # 10 new data samples, each with 4096 features
# print(new_data)
# predictions = clf.predict(new_data)
# print('Predictions:',predictions)

image2 = Image.open('D:/Books_6thsem/Data science/Project/Numberdetection2/data/test/4/1.jpg')
image2 = image2.convert('L')
image2 = image2.resize((64, 64))
flattened_image2 = np.array(image2).flatten().reshape(1,-1)
# Make predictions on the image
prediction = clf.predict(flattened_image2)
print('Prediction:',prediction)