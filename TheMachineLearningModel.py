import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
#I'm not sure that we need the above line of code because we already did the reformatting in the ConvertIMagetoCifar class, but we'll let it slide 

imagesize = (224,224)
#this can also be changed to image.resizing(). I am also now realizing that (32,32) may be a tad too small, so I changed it according to an online guide
batchsize = 25
# I dont thinlk this number matters too much, but I'll keep it low for the sake of the computer.

pathtodata = "images" 
#I dont know if the above name also needs to be "imagesdirectory" or not, I am also unsure wehether they are two different variables.

imagedata = Imageregenerator(rescale = 1./255, split 20%, rotationrange = 50, widthshift = 0.2, heightshift = 0.2, shearrange = 0.2, zoomrange = 0.2, horizontalflip = True)

#These are all just peramters for how much the image can be altered. 

trainingdata = datagen.flow_from_directory(
  pathtodata,
  targetsize = imagesize,
  batchsize = batchsize,
  classtype = 'categorical',
  subset = 'training'

)

validationdata = datagen.flow_from_directory(
  pathtodata,
  targetsize = imagesize,
  batchsize = batchsize
  classtype = 'categorical'
  subset = 'validation'
)

#I had different data for validation, but I guess we'll just thug it out, adding the different files is like highkey-lowkey a pain. 

numberofclasses = 4

#Whatevers below is specific to the CNN model. Convolution Nueral Networks are used for images because of their ability to break down and recognize patterns within the image itself in 2D. So that's why i'm using it. 
# Also btw RELU means non-linear activation function. Whereas some regressions and progressions are linear. It's used for DEEP nueral networking.
model = Sequential([
  Conv2D(32, (3, 3), activation = 'relu', inputshape = (224, 244,   3)),
  MaxPooling2D(2,2),

  Conv2D(64, (3, 3), activation = 'relu'),
  MaxPooling2D(2,2),

  Conv2D(128, (3,3), activation = 'relu'),
  MaxPooling2D(2,2),

  Flatten(),
  Dense(256, activation = 'relu')
  Dropout(0.5) 
  Dense(numberofclasses, activation = 'softmax')
])


#Here we are gettting to Specific aim 2.3, were are using cross entropy, and perfecting our logits with softmax
model.compile(optimizer='adam', loss = 'categorial_crossentropy', metrics = ['accuracy'])


#Apparently dropout is to reduce overfitting, but we'll cross that boat when we get to it
#ADAM is short for Adaptive Moment Estimation, basically it works to redduce the amount of "loss"

history = model.fit(trainingdata, validationdata, epochs = 10)

#Epochs means one pass of training data thgrough the algorithm, we are going to fit the model each time between 10 passes.

model.save("imagerecognitionAI")

plt.plot(history.history['accuracy'], label = 'training accuracy')
plt.plot(history.history['validation_accuracy'], label = 'validation accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
