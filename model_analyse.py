from keras.models import load_model
from keras.utils import plot_model
import argparse
import os
import cv2
import numpy as np
import tensorflow as tf

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model')
    parser.add_argument(
        '-m',
        '--model',
        type=str,
        help='Model File')
    args = parser.parse_args()
    model = load_model(args.model, custom_objects={'tf': tf})
    plot_model(model, to_file='model_architure.png', show_shapes=True)
    model.summary()
    del model
    
 