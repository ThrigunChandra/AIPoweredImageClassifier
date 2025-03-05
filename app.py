import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import  load_model
import streamlit as st
import numpy as np 
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'



st.header('Image Classification Model')
model = load_model('D:/Code/Projects/M3/Image_Classify.keras')

data_cat = ['Audi', 'Mahindra Scorpio', 'Rolls Royce', 'Toyota Innova', 'cats', 'dogs']
img_height = 180
img_width = 180

try:
    image = st.file_uploader('Please upload a image', type=["jpg", "png", "webp"])
    image_load = tf.keras.utils.load_img(image, target_size=(img_height,img_width))
    img_arr = tf.keras.utils.array_to_img(image_load)
    img_bat=tf.expand_dims(img_arr,0)
    predict = model.predict(img_bat)
    score = tf.nn.softmax(predict)
    st.image(image, width=200)
    st.write('The image is ' + data_cat[np.argmax(score)])
    st.write('With accuracy of ' + str(np.max(score)*100))
except:
    print("Starting Exception")
