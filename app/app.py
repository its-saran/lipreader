import streamlit as st
import os
import imageio
import tensorflow as tf
from utils import load_data, num_to_char, load_model
import tempfile
import cv2
import numpy as np

st.set_page_config(layout='wide')

with st.sidebar:
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title('Lip Reader')
    
    options = os.listdir(os.path.join('..', 'data', 's1'))
    selected_video = st.selectbox('Choose video', options)


st.title('Lip Reader')

col1, col2 = st.columns(2)

if options:
    with col1:
        file_path = os.path.join('..', 'data', 's1', selected_video)
        os.system(f'ffmpeg -i {file_path} -vcodec libx264 sample/test_video.mp4 -y')

        video = open('sample/test_video.mp4', 'rb')
        video_bytes = video.read()
        st.video(video_bytes)

    with col2:
        processing_message = st.empty()
        processing_message.info("Processing video... This might take a moment.")

        video, annotations = load_data(tf.convert_to_tensor(file_path))

        model = load_model()
        yhat = model.predict(tf.expand_dims(video, axis=0))

        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()

        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')

        processing_message.empty()

        st.text_area("Decoded Prediction", converted_prediction, height=150)


