**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image2]: ./examples/center.jpg "Driving"
[image3]: ./examples/back1.jpg "Recovery Image"
[image4]: ./examples/back2.jpg "Recovery Image"
[image5]: ./examples/back3.jpg "Recovery Image"
[image6]: ./examples/flip.jpg "Normal Image"
[image7]: ./examples/fliped.jpg "Flipped Image"
[image8]: ./examples/plot.jpg "plot"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 filter sizes and depths between 24 and 48 (model.py lines 95-97) followed by two convolutional layer with 3x3 filter sizes and depths of 64 (model.py lines 98-99). Then three fully connected layers with 500, 250 and 100 nodes after the data is flatten (model.py lines 100-105).

The model includes RELU layers to introduce nonlinearity (model.py line 95-99), and the data is normalized in the model using a Keras lambda layer (model.py line 93).

The model only passes cropped images to get rid of the views have nothing to do with driving. (model.py line 94)

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 202, 104). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 24, 82-83). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 107).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving (normal, normal_more), recovering from the left and right sides of the road (steer_back), curve focused (curve) , reverse center lane driving (normal_back) and Udatciy data (data).

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to create a model to drive in the center of lane.

My first step was to use a convolution neural network model similar to the Nvidia self-driving car model. I thought this model might be appropriate because this model is for self-driving too.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that it has two dropout layers between fully connected layers because the complexcity is mostly introduced by fully connected layers.

Then I doubled nodes in fully connected layers because I do not need to worry about overfitting. But it turns out to be computationally expensive and the result is not good. So I decided to restore the nodes numbers but collected more data.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I collected one data set only focusing on curves.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 95-106) consisted of a convolution neural network with the following layers and layer sizes.

Covolutional layer
  filter 5x5, depth 24
Covolutional layer
  filter 5x5, depth 36
Covolutional layer
  filter 5x5, depth 48
Covolutional layer
  filter 3x3, depth 64
Covolutional layer
  filter 3x3, depth 64
Flatten layer
Fully connected layer
  nodes 500
Fully connected layer
  nodes 250
Fully connected layer
  nodes 100
Fully connected layer
  nodes 1

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded one laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

To augment the data sat, I also flipped images and angles thinking that this would remove the bias. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

After the collection process, I had 15736 number of data points. I then preprocessed this data by cropping the images and normalizing the values.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by plot below. I used an adam optimizer so that manually training the learning rate wasn't necessary. For the same reason, I cannot change the learning rate while it is obvious that the learning rate sometimes is too high.
![alt text][image8]
