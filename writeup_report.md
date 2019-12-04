# **Behavioral Cloning** 

## Writeup 


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[centre]: ./report_img/center.jpg "Centre camera image"
[centre_flip]: ./report_img/center_flip.jpg "Flipped centre image"
[centre_grayscale]: ./report_img/center_grayscale.jpg "Centre image in grayscale"
[centre_crop]: ./report_img/center_crop_grayscale.jpg "Cropped centre image"

[left]: ./report_img/left.jpg "Left camera image"
[left_flip]: ./report_img/left_flip.jpg "Flipped left image"
[left_grayscale]: ./report_img/left_grayscale.jpg "left image in grayscale"
[left_crop]: ./report_img/left_crop_grayscale.jpg "Cropped left image"


[right]: ./report_img/right.jpg "right camera image"
[right_flip]: ./report_img/right_flip.jpg "Flipped right image"
[right_grayscale]: ./report_img/right_grayscale.jpg "right image in grayscale"
[right_crop]: ./report_img/right_crop_grayscale.jpg "Cropped right image"

[model_arc]: ./report_img/model_architecture.png "Model Architecture"
## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model 
```sh
python model.py -d folder_contain_csv1 folder_contain_csv2 -m pre_trained_model_file_for_transfer_learning  
```
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model defined in function
```python
def NvidiaModel(input):
	...
```
and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The overall architecture of the model I adopted is illustarted as below:
![Network Architeture][model_arc]
The input images are of size 320x160 with rgb channel. The network first uses lambda layers to convert rgb image to grayscale image and normalize them 
```python
normalized = Lambda(lambda image: tf.image.rgb_to_grayscale(image)/255-0.5)(input)
```
and then uses an Cropping2D layer to discard uninterested area on images:
```python
cropped = Cropping2D(cropping=((50,20), (0,0)))(normalized)
```
The network consists of 3 5x5 convolution layers with stride of 2 and 2 3x3 convolution layers with stride 1, followed by 3 fully connected layer with output size 100, 50, 10 respectively and out layer which output a number representing the steering angle. 


The model includes RELU layers to introduce nonlinearity after each convolution layers and fully connected layers.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
