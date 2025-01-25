import os
import shutil

os.makedirs(images, exist = True) 
os.cddir(images) 
'''
I'm pretty sure the above code that says os.cddir images is highkey wrong, 
because it's supposed to contain brackets but I'm unsure whether I'm supposed to be calling the main directory or the subdirectories
'''
#I'm making this a comment, I dont think this is how you make subdirectories os.makedirs(Myocardial, Abnormal, History, Normal)

#Tbh the code below I got from an online source because like creating subdirectories for everything and their mom was kind of a pain

for name in names:
  classdirectory = os.path.join(images, name)
  os.makedirs(classdirectory, exist = True)
  for imagepaths in imagepathsforallclasses(name, []):
    shutil.copy(imagepaths, classdirectory)
else:
  print("Sorry, couldn't copy image path :(")
  
