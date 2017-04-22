
# coding: utf-8

# In[1]:

import tensorflow as tf


# In[2]:

import csv
import cv2
import numpy as np

lines = []
# with open('data/driving_log.csv', 'r') as csvfile:
#     next(csvfile)
#     reader = csv.reader(csvfile)
#     for line in reader:
#         lines.append(line)


# In[3]:

def combine_lines(files):
    if lines == []:
        for file in files:
            with open(file + '/driving_log.csv', 'r') as csvfile:
                next(csvfile)
                reader = csv.reader(csvfile)
                for line in reader:
                    lines.append(line)


# In[4]:

#files = ['normal', 'normal_back', 'steer_back', 'harder', 'harder_back']
files = ['normal', 'data', 'curve', 'steer_back', 'normal_back', 'normal_more']
combine_lines(files)


# In[5]:

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)


# In[6]:

import sklearn

def generator(samples, batch_size=32, training=True):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        #sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                #print(batch_sample)
                if batch_sample[0].strip()[:4] == 'IMG/':
                    name = 'data/IMG/' + batch_sample[0].split('/')[-1]
                    left_name = 'data/IMG/' + batch_sample[1].split('/')[-1]
                    right_name = 'data/IMG/' + batch_sample[2].split('/')[-1]
                else:   
                    if batch_sample[0].split('/')[-3] == 'normal_5':
                        name = 'normal_more/IMG/' + batch_sample[0].split('/')[-1]
                        left_name = 'normal_more/IMG/' + batch_sample[1].split('/')[-1]
                        right_name = 'normal_more/IMG/' + batch_sample[2].split('/')[-1]
                    else:
                        name = batch_sample[0].split('/')[-3] + '/' + batch_sample[0].split('/')[-2] + '/' + batch_sample[0].split('/')[-1]
                        left_name = batch_sample[1].split('/')[-3] + '/' + batch_sample[1].split('/')[-2] + '/' + batch_sample[1].split('/')[-1]
                        right_name = batch_sample[2].split('/')[-3] + '/' + batch_sample[2].split('/')[-2] + '/' + batch_sample[2].split('/')[-1]

                center_image = cv2.imread(name)
                left_image = cv2.imread(left_name)
                right_image = cv2.imread(right_name)
                
                if center_image == None:
                    #print(batch_sample)
                    print(name)
                if left_image == None:
                    print(left_name)
                # lost some right images for unknown reason
                if right_image == None:
                    #print(batch_sample[0].split('/'))
                    #print(right_name)
                    continue
                
                center_angle = float(batch_sample[3])
                left_angle = center_angle + 0.1
                right_angle = center_angle - 0.1
                
                if training:
                    images.extend([center_image, left_image, right_image, cv2.flip(center_image,1), cv2.flip(left_image,1), cv2.flip(right_image,1)])
                    angles.extend([center_angle, left_angle, right_angle, center_angle * -1.0, left_angle * -1.0, right_angle * -1.0])
                else:
                    images.extend([center_image, left_image, right_image])
                    angles.extend([center_angle, left_angle, right_angle])
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32, training=False)


# In[7]:

#X_train, y_train = next(train_generator)


# In[8]:

# right_image


# In[9]:

# def process_image(img):
#     return img


# In[10]:

# angles = []
# images = []

# for line in lines:
# #     source_path = line[0]
# #     filename = source_path.split('/')[-1]
# #     current_path = 'data_AQ/IMG/' + filename
# #     image = cv2.imread(current_path)
# #     images.append(image)
# #     print(line[0])
    
#     path = 'data/IMG/'
#     image = process_image(cv2.imread(path + line[0].split('/')[-1]))
#     left_image = process_image(cv2.imread(path + line[1].split('/')[-1]))
#     right_image = process_image(cv2.imread(path + line[2].split('/')[-1]))
    
#     correction = 0.2
#     angle = float(line[3])
#     left_angle = angle + correction
#     right_angle = angle - correction
    
#     images.extend([image, left_image, right_image])
#     angles.extend([angle, left_angle, right_angle])


# In[11]:

# augmented_images, augmented_angles = [], []
# for image, angle in zip(images, angles):
#     augmented_images.append(image)
#     augmented_angles.append(angle)
#     augmented_images.append(cv2.flip(image,1))
#     augmented_angles.append(angle * -1.0)


# In[12]:

# X_train = np.array(augmented_images)
# y_train = np.array(augmented_angles)
# dropout_rate = 0.7


# In[13]:

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dropout

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25),(0, 0))))
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(500))
model.add(Dropout(0.7))
model.add(Dense(250))
model.add(Dropout(0.7))
model.add(Dense(100))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
# model.fit(X_train, y_train, batch_size=32, nb_epoch=10, validation_split=0.2, shuffle=True)
history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator,                                      nb_val_samples=len(validation_samples), nb_epoch=10, verbose=1)
model.save('model.h5')


# In[14]:

import matplotlib.pyplot as plt

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()


# In[ ]:



