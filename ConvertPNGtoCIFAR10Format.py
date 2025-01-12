import os
import numpy as np
from PIL import Image
import pickle

def resizing(image_path, size = (32,32)):
  image = Image.open(image_path).convert("RGB").resize(size)
  return np.array(image)
