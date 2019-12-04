from keras.layers import Input, Lambda, Flatten, \
                        Dense, GlobalAveragePooling2D, Conv2D,Cropping2D
from keras.models import Model, load_model
from keras.utils import plot_model
import argparse
import os
import cv2
import numpy as np
import sklearn
import csv
from sklearn.model_selection import train_test_split
import h5py
from keras import __version__ as keras_version
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
#Model Architure
def NvidiaModel(input):
    normalized = Lambda(lambda image: tf.image.rgb_to_grayscale(image)/255-0.5)(input)
    cropped = Cropping2D(cropping=((50,20), (0,0)))(normalized)
    conv1 = Conv2D(24, (5,5), strides=(2,2), activation='relu')(cropped)
    conv2 = Conv2D(36, (5,5), strides=(2,2), activation='relu')(conv1)
    conv3 = Conv2D(48, (5,5), strides=(2,2), activation='relu')(conv2)
    conv4 = Conv2D(64, (3,3), activation='relu')(conv3)
    conv5 = Conv2D(64, (3,3), activation='relu')(conv4)
    flatten = Flatten()(conv5)
    fc1 = Dense(100, activation='relu')(flatten)
    fc2 = Dense(50, activation='relu')(fc1)
    fc3 = Dense(10, activation='relu')(fc2)
    output = Dense(1)(fc3)
    return output

def generator(samples, batch_size=32):
    img_dir = '/opt/carnd_p3/data/'
    num_samples = len(samples)
    while True:
        for offset in range(0, num_samples, batch_size//6):
            batch_samples = samples[offset:offset+batch_size//6]
            images = []
            angles = []
            for batch_sample in batch_samples:
                #Train model using left/centre/right view image and flip the images to augment the training data
                for idx in range(3):
                    if os.path.isabs(batch_sample[idx]):
                        img_name = batch_sample[idx]
                    else:
                        img_name = os.path.join(img_dir,'IMG',batch_sample[idx].split('/')[-1])
                    img = np.asarray(Image.open(img_name))
          
                    angle = float(batch_sample[3])
                    if idx == 1:
                        angle=min(angle + 0.2, 1)
                    if idx == 2:
                        angle=max(angle - 0.2, -1)
                    flip_img = np.fliplr(img)
                    flip_angle = -angle
                    images.append(img)
                    angles.append(angle)
                    images.append(flip_img)
                    angles.append(flip_angle)
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model')
    parser.add_argument(
        '-d',
        '--dir',
        type=str,
        nargs='*',
        default=['/opt/carnd_p3/data/'],
        help='Image Directory ')
    parser.add_argument(
        '-m',
        '--model',
        type=str,
        nargs='*',
        help='Original Model')
    
    args = parser.parse_args()
    csv_dirs = args.dir
    samples = []
    for csv_dir in csv_dirs:
        with open(os.path.join(csv_dir, 'driving_log.csv')) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                if line[0] == 'center':#discard title line in csv file
                    continue
                samples.append(line)

    train_samples, validation_samples = train_test_split(samples, test_size=0.2)    
    batch_size=128
    train_generator = generator(train_samples, batch_size=batch_size, )
    validation_generator = generator(validation_samples, batch_size=batch_size)
    #Train from pre-trained model (transfer learning)
    if args.model is not None:
        print("Using Pre-trained Model")
        model = load_model(args.model[0])
        for layer in model.layers:
            if 'conv2d' in layer.name:
                layer.trainable=False
    else:#Retrain the model
        input = Input((160, 320, 3))
        output = NvidiaModel(input)
        model = Model(inputs=input, outputs=output)
        model.compile(loss='mse', optimizer='adam')
    history_object = model.fit_generator(train_generator, \
                        steps_per_epoch=np.ceil(len(train_samples)/batch_size*2), \
                        validation_data=validation_generator, \
                        validation_steps=np.ceil(len(validation_samples)/batch_size*2), \
                        epochs=5, verbose=1)
    model.save('TEST.h5')
 