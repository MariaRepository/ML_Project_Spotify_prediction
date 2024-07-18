
import folium.map
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import streamlit as st



import streamlit as st

generos = {
    'acoustic': 0,
    'alt-rock': 1,
    'blues': 2,
    'cantopop': 3,
    'chill': 4,
    'classical': 5,
    'country': 6,
    'dance': 7,
    'dancehall': 8,
    'death-metal': 9,
    'deep-house': 10,
    'disco': 11,
    'drum-and-bass': 12,
    'dub': 13,
    'electro': 14,
    'electronic': 15,
    'folk': 16,
    'forro': 17,
    'funk': 18,
    'garage': 19,
    'groove': 20,
    'guitar': 21,
    'hard-rock': 22,
    'hardcore': 23,
    'hardstyle': 24,
    'hip-hop': 25,
    'house': 26,
    'indie-pop': 27,
    'jazz': 28,
    'k-pop': 29,
    'metal': 30,
    'party': 31,
    'piano': 32,
    'pop': 33,
    'pop-film': 34,
    'progressive-house': 35,
    'punk': 36,
    'rock': 37,
    'salsa': 38,
    'sertanejo': 39,
    'singer-songwriter': 40,
    'soul': 41,
    'spanish': 42
}

# Configuración de la página
st.set_page_config(page_title="Valerapp", page_icon="🤖", layout="wide")

# Sección de presentación
with st.container():
    st.subheader("Data Science Project:")
    st.title("Wanna evaluate your composition popularity?")
    st.write("Your composition is great as it is, but let's see what Spotify may score.")
    
    col1, col2 = st.columns([3, 3])
    
    with col1:
        image_path = "image1.jfif"
        st.image(image_path, caption="Spotify babel", width=500)
    
    with col2:
        st.write("Introduce your composition GENRE.")
        st.write("[Saber más >](https://valerapp.com/)")
        
        # Selector de género musical en el sidebar
        st.sidebar.header('Selecciona un género musical:')
        seleccion = st.sidebar.selectbox('', options=list(generos.keys()))
        
        # Mostrar el código del género seleccionado
        if seleccion in generos:
            codigo = generos[seleccion]
            st.write(f'Seleccionaste: {seleccion} - Código: {codigo}')
        else:
            st.warning('Por favor, selecciona un género musical.')

# Sliders para introducir valores
with st.container():
    st.subheader("Introduce valores para evaluar tu composición:")
    
    # Duración
    duration_number = st.slider("Selecciona duración ms (10-15):", min_value=10, max_value=15)
    if st.button("Confirmar duración"):
        st.write(f"Has seleccionado duración: {duration_number}")

    # Instrumentalidad
    instrumental_number = st.slider("Selecciona instrumentalidad (0-1):", min_value=0.0, max_value=1.0)
    if st.button("Confirmar instrumentalidad"):
        st.write(f"Has seleccionado instrumentalidad: {instrumental_number}")

    # Danceability
    dance_number = st.slider("Selecciona danceability (0-1):", min_value=0.0, max_value=1.0)
    if st.button("Confirmar danceability"):
        st.write(f"Has seleccionado danceability: {dance_number}")

    # Energía
    energy_number = st.slider("Selecciona energía (0-1):", min_value=0.0, max_value=1.0)
    if st.button("Confirmar energía"):
        st.write(f"Has seleccionado energía: {energy_number}")

    # Speechiness
    speech_number = st.slider("Selecciona speechiness (0-1):", min_value=0.0, max_value=1.0)
    if st.button("Confirmar speechiness"):
        st.write(f"Has seleccionado speechiness: {speech_number}")

    # Tempo
    tempo_number = st.slider("Selecciona tempo (0-1):", min_value=0.0, max_value=1.0)
    if st.button("Confirmar tempo"):
        st.write(f"Has seleccionado tempo: {tempo_number}")

