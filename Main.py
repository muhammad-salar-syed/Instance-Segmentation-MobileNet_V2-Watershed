
from keras.layers.merge import Concatenate
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import *
from keras.models import Model
from keras.layers import Input
import glob
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import normalize
import tifffile
from skimage import io,img_as_float,img_as_int
from patchify import patchify, unpatchify
from sklearn.preprocessing import MinMaxScaler
from skimage import measure, color, io

scaler=MinMaxScaler()

img_path = glob.glob('./Images/*')
mask_path = glob.glob('./Masks/*')

Images=[]
Masks=[]
for i in range(len(img_path)):
    I=tifffile.imread(img_path[i])
    M=tifffile.imread(mask_path[i])
    Images.append(I)
    Masks.append(M)
    
image_dataset = np.array(Images)/255.
mask_dataset = np.expand_dims((np.array(Masks)),3) /255.

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(image_dataset, mask_dataset, test_size = 0.10, random_state = 0)

def MbNetV2_Unet(input_shape):
    inputs = Input(shape=input_shape, name="input_image")
    
    encoder = MobileNetV2(input_tensor=inputs, weights="imagenet", include_top=False, alpha=0.35)
    skip_connection_names = ["input_image", "block_1_expand_relu", "block_3_expand_relu", "block_6_expand_relu"]
    encoder_output = encoder.get_layer("block_13_expand_relu").output
    
    f = [32,64,128,256,512]
    x = encoder_output
    for i in range(1, len(skip_connection_names)+1, 1):
        x_skip = encoder.get_layer(skip_connection_names[-i]).output
        x = UpSampling2D((2, 2))(x)
        x = Concatenate()([x, x_skip])
        
        x = Conv2D(f[-i], (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        
        x = Conv2D(f[-i], (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        
    x = Conv2D(1, (1, 1), padding="same")(x)
    x = Activation("sigmoid")(x)
    
    model = Model(inputs, x)
    return model

input_shape=(256,256,3)
model=MbNetV2_Unet(input_shape)
model.summary()

import segmentation_models as sm
metrics = [sm.metrics.IOUScore(threshold=0.5)]

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=metrics)
history = model.fit(X_train, y_train, 
                    batch_size = 16, 
                    verbose=1, 
                    epochs=15, 
                    validation_data=(X_test, y_test), 
                    shuffle=False)

model.save('CT_MV2_unet.hdf5')
