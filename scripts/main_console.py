
# Python 3.6+

import os
import sys
import json

import keras
import numpy as np
import tensorflow as tf

from keras.models import Sequential
from keras.preprocessing import image
from keras.models import load_model
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator


model_name = 'hotel_model.h5'
file_name = 'classification_results.json'  
image_extensions = ('jpeg', 'png', 'jpg', 'tiff', 'gif') 

root_dir = os.path.abspath(os.path.dirname(__name__))


train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
training_set = train_datagen.flow_from_directory('training_images/hotels/training_images', target_size=(64, 64), 
                                                  batch_size=8, class_mode='binary')

test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory('training_images/hotels/test_images', target_size=(64, 64),
                                             batch_size=8, class_mode='binary')



def train_model():

    print('Training the default model. Please wait...')

    classifier = Sequential()

    classifier.add(Convolution2D(32, 2, 3, input_shape=(64, 64, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Flatten())
    classifier.add(Dense(output_dim=128, activation='relu'))
    classifier.add(Dense(output_dim=1, activation='sigmoid'))

    classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])

    classifier.fit_generator(training_set, steps_per_epoch=2000,  epochs=5, validation_data=test_set, validation_steps=200)

    classifier.save(model_name)

    print("Model Trained.\n")

    return 



def predictor(input_type, folder_or_image, classifier): 
    """ 
    Accepts either a folder or an image and a classifier that's the ML model to use for the prediction. 

    If an image is given as input, predicts whether the image is a hotel or not 
    and prints to the terminal
    
    If a folder is supplied, loops through all the files in the folder and 
    creates a .json file containing a list of all images that are hotels and 
    not hotels

    """

    if input_type == 'file':

        image_file = folder_or_image
        image = prepare_image(image_file)

        result = classifier.predict(image)
        outcome = prediction(result)

        print(outcome, '\n')

        return 

    # It's implicit that the input type is a folder from here on

    hotels = [] # list of file names that are hotels
    not_hotels = [] # list of file names that are not hotels

    for folder_name, folders, files in os.walk(folder_or_image):  

        for file in files:

            image = prepare_image(file)
            
            result = classifier.predict(image)
            outcome = prediction(result)

            if outcome == 'hotel':
                hotels.append(file)
            else:
                not_hotels.append(file)

        with open(os.path.join(folder_name, file_name), 'w') as f:        # After each iteration in a folder,
            json.dump({'hotels': hotels, 'not_hotels': not_hotels}, f)    # write result to a json file in the folder

        hotels.clear() # clear the list containing the hotel names for use in the next iterated folder
        not_hotels.clear() # Do the same for the not_hotels list

    return 


def prepare_image(image_file):

    test_image = image.load_img(image_file, target_size=(64, 64))
    test_image = image.img_to_array(image_file)
    test_image = np.expand_dims(image_file, axis=0)

    return test_image


def prediction(result):
    training_set.class_indices

    if result[0][0] >= 0.5:
        prediction = 'hotel'
    else:
        prediction = 'non-hotel'

    return prediction


def main():
    """ The main script """

    print("Team DragonMachine's Image Classifier\n")
    train_model()

    while 1:
    
        model = input("\nEnter a model name to be used for the classification or press 0 to use the default one: ")
        if model == 0:
            break

        if not os.path.isfile(os.path.join(root_dir, model)):
            print("The model name you supplied was not found in the root directory")
            continue
            
        model_name = model
        break
    
    classifier = load_model(model_name)
    print(f"Model '{model}' loaded\n")

    while 1:

        folder_or_image = input('Enter a folder path or image path or "q" to quit: ')

        if folder_or_image.lower() in ('q', 'quit'):
            break

        if not os.path.isdir(folder_or_image):  # if it's not a folder that was supplied, check if it's a file
            if os.path.isfile(folder_or_image): # if it's a file, check that it's a valid image
                if folder_or_image.split('.')[1].lower() not in image_extensions:
                    print(f"Error: An image file is required. Try again. The valid extensions are: {image_extensions}\n")   
                    continue
                
                input_type = 'file'
                predictor(input_type, folder_or_image, classifier)
                continue 
                

            print('Error: Invalid path. Kindly supply a valid folder or image path\n')
            continue

        input_type = 'folder'
        predictor(input_type, folder_or_image, classifier) 
        print(f"Done! The '{file_name}' file has been written to respective folders in {folder_or_image}\n")
        continue

    print("Exiting...")
    exit()


if __name__ == '__main__':
    main()


