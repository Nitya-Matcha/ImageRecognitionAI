import os
from PIL import Image
import pickle
import numpy as np  # Fix the import statement
import shutil
import random

import matplotlib.pyplot as plt
import PIL  # This is supposed to be PIL, but it keeps changing to plistlib for some reason(?)
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout   
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K


def organize(images, names, imagepathsforallclasses, test_split=0.2):
    os.makedirs("cifar-10-batches-py", exist_ok=True)
    os.makedirs("cifar-10-batches-py/test", exist_ok=True)
    os.chdir("cifar-10-batches-py")
    
    for name in names:
        classdirectory = os.path.join("cifar-10-batches-py", name)
        os.makedirs(classdirectory, exist_ok=True)
        test_classdirectory = os.path.join("cifar-10-batches-py/test", name)
        os.makedirs(test_classdirectory, exist_ok=True)
        
        imagepaths = imagepathsforallclasses.get(name, [])
        random.shuffle(imagepaths)
        split_index = int(len(imagepaths) * (1 - test_split))
        
        train_imagepaths = imagepaths[:split_index]
        test_imagepaths = imagepaths[split_index:]
        
        for imagepath in train_imagepaths:
            if os.path.isfile(imagepath):
                shutil.copy(imagepath, classdirectory)
            else:
                print(f"Sorry, couldn't find image/image path: {imagepath}")
        
        for imagepath in test_imagepaths:
            if os.path.isfile(imagepath):
                shutil.copy(imagepath, test_classdirectory)
            else:
                print(f"Sorry, couldn't find image/image path: {imagepath}")

def resizing(imagepath, size=(32, 32)):
    try:
        image = Image.open(imagepath).convert('RGB').resize(size)
        return np.array(image)
    except (PIL.UnidentifiedImageError, IOError) as e:
        print(f"Error opening image {imagepath}: {e}")
        return None

def convertimagetocifar(imagedirectory, savefile, namestonumbers):
    data = []
    labels = []

    for labelnames, labelindex in namestonumbers.items():
        classdirectory = os.path.join(imagedirectory, labelnames)
        if not os.path.isdir(classdirectory):
            print(f"Directory not found: {classdirectory}")
            continue

        for imagename in os.listdir(classdirectory):
            imagepath = os.path.join(classdirectory, imagename)
            if os.path.isfile(imagepath):
                image = resizing(imagepath)
                if image is not None:
                    data.append(image)
                    labels.append(labelindex)
                else:
                    print(f"Skipping image: {imagepath}")
            else:
                print(f"Error cannot find the image: {imagepath}")
        
    if data:
        data = np.array(data).transpose(0, 3, 1, 2).reshape(len(data), -1)
    else:
        print("No data to process")
        return

    labels = np.array(labels)

    batch = {b"data": data, b"labels": labels.tolist()}

    with open(savefile, "wb") as i:
        pickle.dump(batch, i)

    print("Successfully saved!")


def organize_test_images(test_imagepathsforallclasses):
    os.makedirs("cifar-10-batches-py/test", exist_ok=True)
    for name, paths in test_imagepathsforallclasses.items():
        for imagepath in paths:
            if os.path.isfile(imagepath):
                shutil.copy(imagepath, "cifar-10-batches-py/test")
            else:
                print(f"Sorry, couldn't find image/image path: {imagepath}")

def check_images_in_directory(directory):
    if not os.path.exists(directory):
        print(f"Directory does not exist: {directory}")
        return False
    if not os.listdir(directory):
        print(f"No images found in directory: {directory}")
        return False
    return True

def display_class_activation_map(model, img_path, class_index):
    
    if not model.built:
        model.build(input_shape=(None, 32, 32, 3))

   
    img2 = image.load_img(img_path, target_size=(32, 32))
    x = image.img_to_array(img2)
    x = np.expand_dims(x, axis=0)
    x /= 255.0

    preds = model.predict(x)
    predicted_class = np.argmax(preds[0])
    print(f"Predicted class: {predicted_class}")

   
    conv_layer = next(layer for layer in reversed(model.layers) if isinstance(layer, Conv2D))
    if conv_layer is None:
        raise ValueError("No convolutional layer found in the model.")

    
    model2 = Model(inputs=model.input, outputs=[conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = model2(x)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)


    image2 = image.load_img(img_path)
    image2 = image.img_to_array(image2)
    import cv2

    heatmap = cv2.resize(heatmap, (image2.shape[1], image2.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    ontop_image = cv2.addWeighted(image2.astype(np.uint8), 0.6, heatmap, 0.4, 0)
    ontop_image = cv2.cvtColor(ontop_image, cv2.COLOR_BGR2RGB)

    plt.imshow(ontop_image)
    plt.title("CAM")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    images = "cifar10.pkl"
    names = ["Myocardial", "Abnormal", "History", "Normal"]
    num_classes = 5  # Determine the number of classes

    imagepathsforallclasses = {
        "Myocardial": ["c:/Users/Nitya/Desktop/cifar-10-batches-py/Myocardial/MI(1).png","c:/Users/Nitya/Desktop/cifar-10-batches-py/Myocardial/MI(2).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/Myocardial/MI(3).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/Myocardial/MI(4).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/Myocardial/MI(5).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/Myocardial/MI(6).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/Myocardial/MI(7).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/Myocardial/MI(8).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/Myocardial/MI(9).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/Myocardial/MI(10).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/Myocardial/MI(12).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/Myocardial/MI(13).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/Myocardial/MI(14).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/Myocardial/MI(15).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/Myocardial/MI(16).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/Myocardial/MI(17).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/Myocardial/MI(18).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/Myocardial/MI(19).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/Myocardial/MI(20).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/Myocardial/MI(21).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/Myocardial/MI(22).png",
        "c:/Users/Nitya/Desktop/cifar-10-batches-py/Myocardial/MI(23).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/Myocardial/MI(24).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/Myocardial/MI(25).png","c:/Users/Nitya/Desktop/cifar-10-batches-py/Myocardial/MI(26).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/Myocardial/MI(27).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/Myocardial/MI(28).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/Myocardial/MI(29).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/Myocardial/MI(30).png"],

        "Abnormal": ["c:/Users/Nitya/Desktop/cifar-10-batches-py/Abnormal/HB(1).png","c:/Users/Nitya/Desktop/cifar-10-batches-py/Abnormal/HB(2).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/Abnormal/HB(3).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/Abnormal/HB(4).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/Abnormal/HB(5).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/Abnormal/HB(6).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/Abnormal/HB(7).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/Abnormal/HB(8).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/Abnormal/HB(9).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/Abnormal/HB(10).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/Abnormal/HB(11).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/Abnormal/HB(12).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/Abnormal/HB(13).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/Abnormal/HB(14).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/Abnormal/HB(15).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/Abnormal/HB(16).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/Abnormal/HB(17).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/Abnormal/HB(18).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/Abnormal/HB(19).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/Abnormal/HB(20).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/Abnormal/HB(21).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/Abnormal/HB(22).png",
        "c:/Users/Nitya/Desktop/cifar-10-batches-py/Abnormal/HB(23).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/Abnormal/HB(24).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/Abnormal/HB(25).png","c:/Users/Nitya/Desktop/cifar-10-batches-py/Abnormal/HB(26).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/Abnormal/HB(27).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/Abnormal/HB(28).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/Abnormal/HB(29).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/Abnormal/HB(30).png"],

        "History": ["c:/Users/Nitya/Desktop/cifar-10-batches-py/History/PMI(1).png","c:/Users/Nitya/Desktop/cifar-10-batches-py/History/PMI(2).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/History/PMI(3).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/History/PMI(4).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/History/PMI(5).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/History/PMI(6).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/History/PMI(7).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/History/PMI(8).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/History/PMI(9).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/History/PMI(10).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/History/PMI(11).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/History/PMI(12).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/History/PMI(13).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/History/PMI(14).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/History/PMI(15).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/History/PMI(16).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/History/PMI(17).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/History/PMI(18).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/History/PMI(19).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/History/PMI(20).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/History/PMI(21).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/History/PMI(22).png",
        "c:/Users/Nitya/Desktop/cifar-10-batches-py/History/PMI(23).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/History/PMI(24).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/History/PMI(25).png","c:/Users/Nitya/Desktop/cifar-10-batches-py/History/PMI(26).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/History/PMI(27).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/History/PMI(28).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/History/PMI(29).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/History/PMI(30).png"],

        "Normal": ["c:/Users/Nitya/Desktop/cifar-10-batches-py/Normal/Normal(1).png","c:/Users/Nitya/Desktop/cifar-10-batches-py/Normal/Normal(2).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/Normal/Normal(3).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/Normal/Normal(4).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/Normal/Normal(5).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/Normal/Normal(6).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/Normal/Normal(7).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/Normal/Normal(8).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/Normal/Normal(9).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/Normal/Normal(10).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/Normal/Normal(11).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/Normal/Normal(12).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/Normal/Normal(13).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/Normal/Normal(14).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/Normal/Normal(15).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/Normal/Normal(16).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/Normal/Normal(17).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/Normal/Normal(18).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/Normal/Normal(19).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/Normal/Normal(20).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/Normal/Normal(21).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/Normal/Normal(22).png",
        "c:/Users/Nitya/Desktop/cifar-10-batches-py/Normal/Normal(23).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/Normal/Normal(24).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/Normal/Normal(25).png","c:/Users/Nitya/Desktop/cifar-10-batches-py/Normal/Normal(26).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/Normal/Normal(27).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/Normal/Normal(28).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/Normal/Normal(29).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/Normal/Normal(30).png"],
    }

    test_imagepathsforallclasses = {

        "Myocardial":["c:/Users/Nitya/Desktop/cifar-10-batches-py/test/MI(120).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/test/MI(119).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/test/MI(118).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/test/MI(117).png"],
        "Abnormal":["c:/Users/Nitya/Desktop/cifar-10-batches-py/test/HB(120).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/test/HB(119).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/test/HB(118).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/test/HB(117).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/test/HB(116).png"],
        "History":["c:/Users/Nitya/Desktop/cifar-10-batches-py/test/PMI(120).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/test/PMI(119).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/test/PMI(118).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/test/PMI(117).png"],
        "Normal":["c:/Users/Nitya/Desktop/cifar-10-batches-py/test/Normal(120).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/test/Normal(119).png", "c:/Users/Nitya/Desktop/cifar-10-batches-py/test/Normal(118).png",]
    }

    organize(images, names, imagepathsforallclasses)

    # Check if directories contain images
    if not check_images_in_directory("cifar-10-batches-py") or not check_images_in_directory("cifar-10-batches-py/test"):
        print("Exiting due to missing images.")
        exit(1)

    namestonumbers = {
        "Myocardial": 0,
        "Abnormal": 1,
        "History": 2,
        "Normal": 3
    }

    convertimagetocifar("cifar-10-batches-py", "cifar10.pkl", namestonumbers)

    # Check if training and test data directories contain images
    if not check_images_in_directory("cifar-10-batches-py") or not check_images_in_directory("cifar-10-batches-py/test"):
        print("Exiting due to missing images.")
        exit(1)

    imagesize = (32, 32)
    batchsize = 25

    imagedata = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2, rotation_range=50, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

    trainingdata = imagedata.flow_from_directory("cifar-10-batches-py", target_size=imagesize, batch_size=batchsize, class_mode="categorical", subset='training')
    validationdata = imagedata.flow_from_directory("cifar-10-batches-py", target_size=imagesize, batch_size=batchsize, class_mode="categorical", subset='validation')
    testdata = imagedata.flow_from_directory("cifar-10-batches-py/test", target_size=imagesize, batch_size=batchsize, class_mode="categorical")

    # Print shapes of data
    print(f"Training data shape: {trainingdata.samples}")
    print(f"Validation data shape: {validationdata.samples}")
    print(f"Test data shape: {testdata.samples}")

    # Print a few labels to verify
    print(f"Training labels: {trainingdata.class_indices}")
    print(f"Test labels: {testdata.class_indices}")

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3), name='conv2d_1'),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation='relu', name='conv2d_2'),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), activation='relu', name='conv2d_3'),
        MaxPooling2D((2, 2)),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    def display_class_activation_map(model, img_path, class_index):
    
        if not model.built:
            model.build(input_shape=(None, 32, 32, 3))

   
        img2 = image.load_img(img_path, target_size=(32, 32))
        x = image.img_to_array(img2)
        x = np.expand_dims(x, axis=0)
        x /= 255.0

        preds = model.predict(x)
        predicted_class = np.argmax(preds[0])
        print(f"Predicted class: {predicted_class}")

   
        conv_layer = next(layer for layer in reversed(model.layers) if isinstance(layer, Conv2D))
        if conv_layer is None:
         raise ValueError("No convolutional layer found in the model.")

    
        model2 = Model(inputs=model.input, outputs=[conv_layer.output, model.output])

        with tf.GradientTape() as tape:
            conv_outputs, predictions = model2(x)
            loss = predictions[:, class_index]

        grads = tape.gradient(loss, conv_outputs)[0]
        pooled_grads = K.mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        for i in range(pooled_grads.shape[-1]):
            conv_outputs[:, :, i] *= pooled_grads[i]

        heatmap = np.mean(conv_outputs, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)


        image2 = image.load_img(img_path)
        image2 = image.img_to_array(image2)
        import cv2

        heatmap = cv2.resize(heatmap, (image2.shape[1], image2.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        ontop_image = cv2.addWeighted(image2.astype(np.uint8), 0.6, heatmap, 0.4, 0)
        ontop_image = cv2.cvtColor(ontop_image, cv2.COLOR_BGR2RGB)

        plt.imshow(ontop_image)
        plt.title("CAM")
        plt.axis("off")
        plt.show()

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(trainingdata, validation_data=validationdata, epochs=10)

    test_loss, test_accuracy = model.evaluate(testdata)
    print(f"Test accuracy: {test_accuracy}")
    model.save("imagerecognitionAi.h5")

    #plt.plot(history.history['accuracy'], label='training accuracy')
    #plt.plot(history.history['val_accuracy'], label='validation accuracy')
    #plt.plot(history.history['loss'], label='training loss')
    #plt.plot(history.history['val_loss'], label='validation loss')
    #plt.xlabel('Epoch')
    #plt.ylabel('Accuracy')
    #plt.legend()
    #plt.show()

    # Display Class Activation Map for a sample image, in this case its just the first image of the first class
    sample_image = "c:/Users/Nitya/Desktop/cifar-10-batches-py/Myocardial/MI(1).png"
    display_class_activation_map(model, sample_image, class_index=0)
