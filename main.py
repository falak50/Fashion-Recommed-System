import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import cv2

# pkl file import by numpy array ---------------------------------
feature_list = np.array(pickle.load(open("featurevector.pkl", 'rb')))
# print(feature_list)
filename = pickle.load(open("filenames.pkl", 'rb'))

# model select-----------------------------------------------------------------
model = ResNet50(weights='imagenet', include_top=False,
                 input_shape=(224, 224, 3))
model.trainable = False

model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])
# model.summary()

# header------------------------------------------------------------
st.sidebar.header('Welcome To Our')
st.sidebar.subheader('Fashion Recommed System')
st.sidebar.text('Fashion Suggestions for users')
# side---------------------------------------------------------------
st.title('Fashion Recommed System')
st.write('Similar picture idetification')

# header------------------------------------------------------------

# all function --------------------------------


def save_upload_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0


def extrack_feature(img_path, model):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = np.array(img)
    expand_img = np.expand_dims(img, axis=0)
    pre_img = preprocess_input(expand_img)
    result = model.predict(pre_img).flatten()
    normalized = result / norm(result)
    return normalized


def recommend(features, feature_list):
    neighbors = NearestNeighbors(
        n_neighbors=6, algorithm="brute", metric="euclidean")
    neighbors.fit(feature_list)
    distance, indices = neighbors.kneighbors([features])

    return indices


# file upload and save the file
upload_file = st.file_uploader("Upload your image for searching")
# print(upload_file)

if upload_file is not None:
    if save_upload_file(upload_file):
        # display the file
        display_image = Image.open(upload_file)
        resized_img = display_image.resize((300, 350))
        st.write('Uploaded Image')
        st.image(resized_img)
        st.write('Find Image')
        # feature extruct of out new picture
        features = extrack_feature(os.path.join(
            'uploads', upload_file.name), model)

        # st.test(features)
        # recommendetion
        indices = recommend(features, feature_list)
        # show the image
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.image(filename[indices[0][0]])
        with col2:
            st.image(filename[indices[0][1]])
        with col3:
            st.image(filename[indices[0][2]])
        with col4:
            st.image(filename[indices[0][3]])
        with col5:
            st.image(filename[indices[0][4]])
    else:
        st.header("May be have some error ocuured in file upload")
