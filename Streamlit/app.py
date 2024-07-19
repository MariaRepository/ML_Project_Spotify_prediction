import streamlit as st
import pandas as pd
import pickle
import numpy as np
from PIL import Image

# Dictionary of musical genres and their codes
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

# Page configuration
st.set_page_config(page_title="DS Spotify Project", page_icon="🎹", layout="wide")

# Load the model
model_path = r"C:\Users\mfm-8\Online_Env\ML_Project_Spotify_prediction\results_notebook\Models_results\models_results\mejor_modelo_lgb_reg_short.pkl"
best_lgb_reg_short = pickle.load(open(model_path, 'rb'))

# Feature names used in the model
selected_features = [
    'duration_ms',
    'instrumentalness',
    'danceability',
    'liveness',
    'energy',
    'speechiness',
    'tempo',
    'year_2018', 'year_2019', 'year_2020', 'year_2021', 'year_2022', 'year_2023',
    'genre'
]

# Initialize input values
input_values = {
    'duration_ms': None,
    'instrumentalness': None,
    'danceability': None,
    'liveness': None,
    'energy': None,
    'speechiness': None,
    'tempo': None,
    'year_2018': 0,
    'year_2019': 0,
    'year_2020': 0,
    'year_2021': 0,
    'year_2022': 0,
    'year_2023': 0,
    'genre': None
}

# Presentation section
with st.container():
    st.subheader("Data Science Project:")
    st.title("Wanna evaluate your composition's popularity❤️?")
    st.write("Your composition is great as it is, but let's see what Spotify may score.")

    # Genre selection section
    st.subheader("Select your genre:")
    seleccion = st.selectbox('', options=list(generos.keys()))
    
    # Display the code of the selected genre
    if seleccion in generos:
        codigo = generos[seleccion]
        #st.write(f'Genre selected: {seleccion} - Code: {codigo}')
        input_values['genre'] = codigo
    else:
        st.warning('Please, select a musical genre.')

    col1, col2 = st.columns([5, 5])
    
    with col1:
        image_path = "C:/Users/mfm-8/Online_Env/ML_Project_Spotify_prediction/Others/image1.jfif"
        image_path_2 = "C:/Users/mfm-8/Online_Env/ML_Project_Spotify_prediction/Others/1.jpg"

        st.image(image_path, caption="Spotify babel Tower made with IA", width=700)
        st.image(image_path_2, caption="All rights reserved 2024", width=700)

    # Main section for input values
    with col2:
        st.subheader("Enter values to evaluate your composition:")

        # Duration
        input_values['duration_ms'] = st.slider("Select duration in ms (10-15):", min_value=10, max_value=15)
        
        # Instrumentalness
        input_values['instrumentalness'] = st.slider("Select instrumentalness (0-1):", min_value=0.0, max_value=1.0)
        
        # Danceability
        input_values['danceability'] = st.slider("Select danceability (0-1):", min_value=0.0, max_value=1.0)
        
        # Liveness
        input_values['liveness'] = st.slider("Select liveness (0-1):", min_value=0.0, max_value=1.0)
        
        # Energy
        input_values['energy'] = st.slider("Select energy (0-1):", min_value=0.0, max_value=1.0)
        
        # Speechiness
        input_values['speechiness'] = st.slider("Select speechiness (0-1):", min_value=0.0, max_value=1.0)
        
        # Tempo
        input_values['tempo'] = st.slider("Select tempo (0-1):", min_value=0.0, max_value=1.0)
        
        # Year (assuming you want to select which year the composition is from)
        year = st.selectbox("Select the year of the composition:", ['2018', '2019', '2020', '2021', '2022', '2023'])
        for y in ['2018', '2019', '2020', '2021', '2022', '2023']:
            input_values[f'year_{y}'] = 1 if y == year else 0

        # Button to confirm all inputs and make prediction
        if st.button("Confirm All"):
            #st.write(f"Inputs confirmed: {input_values}")
            
            # Create a DataFrame for prediction
            input_df = pd.DataFrame([input_values])
            
            # Make prediction
            try:
                prediction = best_lgb_reg_short.predict(input_df[selected_features])[0]
                prediction_rounded = round(prediction, 2)
                
                # Display the prediction with larger font
                st.markdown(f"<h2 style='text-align: center; font-size: 36px;'>🎹 Prediction: {prediction_rounded}<br><br>🎧 Trust yourself, we are more than a NUMBER<br><br>❤️</h2>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error making prediction: {e}")
            st.markdown("""
                <div style='font-size: 24px; text-align: center; color: #f39c12;'>
                    Wanna learn more about composition? Check our courses at <a href="http://www.compositionmaster.com" target="_blank">www.compositionmaster.com</a>
                </div>
            """, unsafe_allow_html=True)

# Custom CSS for blue sliders and button
st.markdown("""
    <style>
    /* Custom style for sliders */
    .stSlider .st-bw {
        background-color: #1f77b4;
    }
    .stSlider input[type="range"]::-webkit-slider-thumb {
        background: #1f77b4;
        border: 2px solid #1f77b4;
    }
    .stSlider input[type="range"]::-moz-range-thumb {
        background: #1f77b4;
        border: 2px solid #1f77b4;
    }
    /* Custom style for buttons */
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
        font-size: 16px;
    }
    .stButton > button:hover {
        background-color: #1565c0;
    }
    </style>
""", unsafe_allow_html=True)