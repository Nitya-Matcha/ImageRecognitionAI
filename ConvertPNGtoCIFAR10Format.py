import os
from PIL import Image
import pickle
import numpy as np

def resizing(imagepath, size = (32,32)):
  image = Image.open(imagepath).convert("RGB").resize(size)
  return np.array(image)

def convertimagetocifar(imagedirectory, savefile, namestonumbers):
  data = []
  labels = []

for labelnames, labelindex in label.items():
  classdirectory = os.path.join(imagedirectory, labelnames)
  if not os.path.isdir(classdirectory):
    print("Not in this folder of images")
  else:
    continue


