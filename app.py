import streamlit as st
import requests
import json
from io import BytesIO
from PIL import Image

st.title('Face Expression Prediction')

image_file = st.file_uploader("Upload Image")

if image_file is not None:
    image = Image.open(image_file)  # Open the image file
    #resized_image = image.resize((50, 50))
    
    st.image(resized_image, caption='Uploaded Image', width=100,height=100)  # Display the resized image

    image = image_file.getvalue()

    response = requests.post(
        "https://dpatel9923-face-expression-prediction1.hf.space/prediction",
        files = {
            "image": BytesIO(image)
        }
    )

    label = json.loads(response._content)
    st.write(f"Expression of Face is {label['class']}")
