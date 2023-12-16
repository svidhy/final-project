import streamlit as st 
import functions as fn
from PIL import Image
import os
from keras.preprocessing.image import load_img
from tempfile import NamedTemporaryFile
import base64
import requests
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder
from agstyler import PINLEFT, PRECISION_TWO, draw_grid

fn.set_background('./images/bg8.jpeg')

st.markdown('<script src="https://developer.edamam.com/attribution/badge.js"></script>', unsafe_allow_html=True)
st.title("Welcome to NutriSnap!")

recipe_db, labels = fn.init_page()
incep_model = fn.load_the_model()

uploaded_file = st.file_uploader("Please upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    if fn.is_image(uploaded_file):
        image = st.image(uploaded_file, caption='Uploaded Image', width=200)
        pred = fn.predict_image(uploaded_file, incep_model, labels)
        st.write("Is the image uploaded", pred.capitalize(), '?')
        recipes = fn.get_recipe(recipe_db, pred)
        recipes.reset_index(drop=True, inplace=True)

        confirm_button = st.button("Yes")
        no_button = st.button("No")

        if confirm_button:
            # Replace the button with an empty container
            confirm_button_container = st.empty()
            recipe = fn.format_recipe(recipes)
            # Display the Edamam badge
            # Display the Edamam badge and the CSS
            st.write(recipe.to_html(escape=False, index=False), unsafe_allow_html=True)
        elif no_button:
            st.write("Sorry, we will work on this!")
            # Clear the button from the layout
            confirm_button_container = st.empty()
            #confirm_button_container.text("Thank you for confirming!")
        
    else:
        st.warning("File is not a valid image.")


