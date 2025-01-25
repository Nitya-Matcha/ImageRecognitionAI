import os
from PIL import Image
import pickle
import numpy as np

def resizing(imagepath, size = (32,32)):
  image = Image.open(imagepath).convert("RGB").resize(size)
  return np.array(image)

"""
Setting up the argument for what quantifies as "Resizing" an image. This argument will be called 
when it's time to actually change the images, I could combine this with calling for the image, but thats a lot of work so 
I'm going to isolate the two
"""

def convertimagetocifar(imagedirectory, savefile, namestonumbers):
  data = []
  labels = []

for labelnames, labelindex in namestonumbers.items():
  classdirectory = os.path.join(imagedirectory, labelnames)
  if not os.path.isdir(classdirectory):
    print("Not in this folder of images")
  else:
    continue
  for imagename in os.listdir(classdirectory):
    imagepath = os.path.join(classdirectory, imagename)
    else:
      print("Error! Can't find image, (or cant join image and class directory paths)")

data = np.array(data).transpose((0,3,1,2)).reshape(len(data), -1)

"""
this one arranges the data into the CIfar-10 format, it's deconstructing the array of the images and re-doing them
to be fully honest I don't know the logisitics, someone online found this
"""
labels = np.array(labels)

#this one is just setting the labels as an array called labels, it allows them to be easily manipulated using numpy

"""
This code is for training, in a testing code the machine would not react to 
an os folder or anything else. It would simply try to identify the image regardless of wehterh or not it's in the database
"""
batch = { b"data": data, b"labels": label.tolist() }

with open(savefile, "nochange2file") as i:
  pickle.dump(batch, i)
print("successfully saved!")

if __name__ == "__main__":
  imagedirectory = "images"
  savefile = "savefiles"
#giving names for the directories that hold the groups of images. I couldn't think of a creative name for savefile

namestonumbers = {
  "Myocardial" = 0
  "Abnormal" = 1
  "History" = 2
  "Normal" = 3
}

#The above is assigning a class of images to a specific label. It's way easier to do it in terms of a number than setting up a string, mainly because I can't spell.

createnewcifar10group(imagedirectory, savefile, namestonumbers)

#It's dumping whatever output the computer got into an object called "i" si ut can eb traced later

