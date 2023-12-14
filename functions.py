import os
import streamlit as st 
import warnings
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import json_normalize
warnings.filterwarnings("ignore")
import base64

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing import image
from tensorflow.keras import models
from tensorflow.keras.models import load_model
from tensorflow.keras import models
from tensorflow.keras import layers 
from PIL import Image 
from skimage.io import imread
import cv2
K.clear_session()

def is_image(file):
    try:
        # Attempt to open the image
        img = Image.open(file)
        return True
    except Exception as e:
        return False

def init_page():
    df = pd.read_csv('recipe_database.csv')
    labels = list(df['category'].unique())
    print(len(labels))
    return df,labels

def load_the_model():
    K.clear_session()
    path_to_model='./model_resnet_2.h5'
    print("Loading the model..")
    model_inception = load_model(path_to_model)
    print("Done!")
    return model_inception

def predict_image(picture, model, labels):
    # Open the uploaded image using PIL
    img = Image.open(picture)
    
    # Resize the image to the target size
    img_ = img.resize((299, 299))

    # Convert the image to an array
    img_array = image.img_to_array(img_)
    
    # Expand the dimensions to match the model's expected input shape
    img_processed = np.expand_dims(img_array, axis=0) 

    # Normalize the pixel values to be in the range [0, 1]
    img_processed /= 255.

    # Make the prediction
    prediction = model.predict(img_processed)
    
     # Get the index of the predicted class
    index = np.argmax(prediction)
    
    plt.title("Prediction - {}".format(labels[index]))
    plt.imshow(img_array)
    return labels[index]

def get_recipe(recipe_db, pred_ans):

    if len(recipe_db[recipe_db['category'] == pred_ans]) < 5:
        return recipe_db[recipe_db['category'] == pred_ans]
    else:
        return recipe_db[recipe_db['category'] == pred_ans].sample(5)


def format_recipe(recipes):
    #st.write(recipes)
    recipes_reset = recipes.reset_index(drop=True)
    recipes_reset.index += 1
    recipes_reset = recipes_reset[['dish_name', 'energy_kcal', 'source', 'source_url','servings']]


    # Extract the 'source' and 'link' columns
    source_list = recipes_reset['source'].tolist()
    link_list = recipes_reset['source_url'].tolist()
    ener_serv = round(recipes_reset['energy_kcal']/recipes_reset['servings'])

    # Create HTML links
    html_links = []
    for source, link in zip(source_list, link_list):
        href = f"<a href='{link}'>{source}</a>"
        html_links.append(href)
    
    # Create a new DataFrame with hyperlinks
    recipes_with_hyperlinks = pd.DataFrame({
        'Dish Name': recipes_reset['dish_name'],
        'Total Energy (kcal)': recipes_reset['energy_kcal'],
        'Servings': recipes_reset['servings'],
        'Energy Per Serving (kcal)': ener_serv,
        'Source': html_links
    })
    return recipes_with_hyperlinks

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

""""
# Sample DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 22],
    'More Info': ['Details about Alice', 'Details about Bob', 'Details about Charlie']
}

df = pd.DataFrame(data)

# Display the table with checkboxes
selected_rows = []
for index, row in df.iterrows():
    checkbox_value = st.checkbox(f"Select {row['Name']}", key=index)
    selected_rows.append(checkbox_value)

# Check if any rows are selected
for index, selected in enumerate(selected_rows):
    if selected:
        # Get the selected row's data
        selected_data = df.iloc[index]

        # Create an expander for more details
        with st.expander(f"More Info for {selected_data['Name']}"):
            st.write(selected_data['More Info'])

    """