#Importing  Tools
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
import requests
import torch

#Initializing Test and Train Variables
(training_images, training_labels) , (testing_images, testing_labels) = datasets.cifar10.load_data()
training_images, testing_images = training_images/255, testing_images/255

#Creating Class Variables
class_names = ['Plane', 'Car','Bird', 'Cat','Deer','Dog','Frog','Horse','Ship','Truck']

#Setting the training limit to dataset
training_images = training_images[:100000]
training_labels = training_labels[:100000]
testing_images = testing_images[:20000]
testing_labels = testing_labels[:20000]

#Viewing the training images
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[training_labels[i][0]])
plt.show()

#Now the model is saved onto a file 'image_classifier.keras'

'''#Creating the Model
model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation="relu", input_shape = (32,32,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation="relu"))
model.add(layers.Flatten())
model.add(layers.Dense(64,activation="relu"))
model.add(layers.Dense(10,activation="softmax"))
model.compile(optimizer="adam", loss='sparse_categorical_crossentropy',metrics=["accuracy"])
model.fit(training_images, training_labels, epochs=11, validation_data=(testing_images,testing_labels))

model.save('image_classifier.keras')
'''

#Loading the model from the file 'image_classifier.keras'
model = models.load_model('image_classifier.keras')

#Evaluating the Model
loss, accuracy = model.evaluate(testing_images, testing_labels)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}/n")

#Getting the test image
img = cv.imread('car.jpg')

#Image Preprocessing
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
plt.imshow(img, cmap=plt.cm.binary)
plt.show()

#Model Deployment
prediction = model.predict(np.array([img])/255)
index = np.argmax((prediction))
print(f"The image is a {class_names[index]}")

#Deploying Google Search API
print(f"Additional links for {class_names[index]}:\n")
API_KEY = open('GOOGLE_SEARCH_API_KEY').read()
Search_Engine_ID = open('Search_Engine_ID').read()
search_query = class_names[index]
url = 'https://www.googleapis.com/customsearch/v1'
params = {
    'q': search_query,
    'key': API_KEY,
    'cx': Search_Engine_ID,
    'searchType': 'image'
}
response = requests.get(url, params=params)
results = response.json()['items']

#Links for Predicted Output
for item in results:
    print(item['link'])












