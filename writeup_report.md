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

[dirt_section]: ./report_img/dirt.jpg "Dirt section"
[sharp_curve]: ./report_img/sharp_curve.jpg "Sharp curve"

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

To reduce the overfitting in model, I tried to collect enough data until the MSE is both less than 0.02 for training dataset and validation dataset. 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. When the trained model had bad performance in certain portion of the track, I collected more data near that portion to improve the performance. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The model architeture used for this project is described in *end to end leaning for self-driving car* published by Nvidia engineers. The modifications made on the original network are
- adding one lambda layer to convet rgb image to grayscale image and normalize it
- adding one cropping layer to remove uninterest area on image.

The detail architeture of model has been present in last secion.

The training dataset used for first attempt was the data already provided.
In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 
To augment the training dataset, all three camera images, left right and centre, were used to train the model. the steering angle for left and right image are derived from steering angle of centre image as below:
```python
left_angle=min(centre_angle + 0.2, 1)
right_angle=max(centre_angle - 0.2, -1)
```
To further augment the training dataset, the original images (for left, centre and right) were flipped horizontally and the steering angles were sign-reversed.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots shown in figures below where the vehicle fell off the track.

![dirt area][dirt_section]
![sharp curve][sharp_curve]
 
To improve the driving behavior in these cases, I recorded more data in these spots and also recorded some data which teaches the car how to recover from left side and right side. I used these data together with the already provided data to train the model. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.


#### 2. Creation of the Training Set & Training Process

To capture good driving behavior, I used the example training dataset(left, centre and right camera images are all used) provided as start point. Here is an example image of center lane driving:
- Left image

![left image][left]

- Centre image

![centre image][centre]

- Right image

![right image][right]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to.


To augment the data sat, I also flipped images and angles. For example, here is an image that has then been flipped:
- Original image

![centre image][centre]

- Flipped image

![centre flipped][centre_flip]


After the collection process, I then preprocessed this data by 

- First convert rgb images to grayscale images:

![centre grayscale][centre_grayscale]

- Second only keep region of interested:

![cropped][centre_crop]



I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The number of epochs was 5. I used an adam optimizer so that manually training the learning rate wasn't necessary.
