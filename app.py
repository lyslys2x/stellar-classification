import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline

model = joblib.load("model.pkl")

# Page configuration
st.set_page_config(
    page_title="IdentiStar - Stellar Classification",
    page_icon="⭐",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for styling
st.markdown("""
<style>
    
    .header-container {
        background: linear-gradient(90deg, #16213e 0%, #AD99CF 100%);
        padding: 1rem 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    
    .header-title {
        font-size: 2.5rem;
        color: white;
        text-align: left;
        margin: 0;
    }
    
    .nav-links {
        display: flex;
        gap: 2rem;
        margin-top: 1rem;
    }
    
    .nav-link {
        color: white;
        text-decoration: none;
        font-weight: 500;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        transition: background-color 0.3s;
    }
    
    .nav-link:hover {
        background-color: rgba(255,255,255,0.1);
    }
    
    .panel {
        background: linear-gradient(135deg, #2d1b69 0%, #1e3c72 100%);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.4);
        border: 1px solid rgba(255,255,255,0.1);
        height: auto;
    }
    
    .panel-title {
        color: white;
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    
    .stTextInput > div > div > input {
        background-color: #f0f0f0;
        border: none;
        border-radius: 8px;
        padding: 0.75rem;
        color: #333;
    }
    
    .stTextInput > div > div > input:focus {
        box-shadow: 0 0 0 2px rgba(255,255,255,0.3);
    }
    
    .prediction-result {
        text-align: center;
        color: white;
        margin-top: 1rem;
    }
    
    .prediction-title {
        font-size: 2rem;
        font-weight: bold;
        margin: 1rem 0;
        color: #ffd700;
    }
    
    .prediction-description {
        font-size: 1rem;
        line-height: 1.6;
        margin-top: 1rem;
    }
    
    .star-background {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: -1;
    }
    
    .star {
        position: absolute;
        background: white;
        border-radius: 50%;
        animation: twinkle 3s infinite;
    }
    
    @keyframes twinkle {
        0%, 100% { opacity: 0.3; }
        50% { opacity: 1; }
    }

    .stImage img {
        border-radius: 10px !important;
        border: 2px solid rgba(255, 255, 255, 0.2);
    }
    
    .stImage {
        border-radius: 10px;
        overflow: hidden;
    }
</style>
""", 
unsafe_allow_html=True)

# Generate starry background
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("https://www.mccormick.northwestern.edu/images/news/2023/07/what-does-a-twinkling-star-sound-like-take-a-listen-social.jpg");
        background-size: cover
    }}
    .stApp::before {{
        content: "";
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background: rgba(60, 40, 20, 0.25); /* Adjust alpha for lighter/darker */
        pointer-events: none;
        z-index: 0;
    }}
    </style>
    """, unsafe_allow_html=True)

# Header
st.markdown("""
<div class="header-container">
    <h1 class="header-title">IdentiStar</h1>
    <div class="nav-links">
        <a href="#" class="nav-link">Home</a>
        <a href="#" class="nav-link">Stars</a>
        <a href="#" class="nav-link">Star Classifier</a>
        <a href="#" class="nav-link">About</a>
    </div>
</div>
""", unsafe_allow_html=True)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<h2 class="panel-title">Enter values for Stellar Classification</h2>', unsafe_allow_html=True)
    
    # Using number inputs to control the range of value the user is entering
    alpha_selected = st.number_input("Enter Alpha", min_value=0, max_value=120, step=1)
    delta_selected = st.number_input("Enter Delta", min_value=0, max_value=120, step=1)
    red_shift_selected = st.number_input("Enter Red Shift", min_value=0, max_value=120, step=1)
    UV_filter_selected = st.number_input("Enter UV Filter", min_value=0, max_value=120, step=1)
    green_filter_selected = st.number_input("Enter Green Filter", min_value=0, max_value=120, step=1)
    red_filter_selected = st.number_input("Enter Red Filter", min_value=0, max_value=120, step=1)
    ir_filter_selected = st.number_input("Enter Infrared Filter", min_value=0, max_value=120, step=1)
    near_ir_filter_selected = st.number_input("Enter Near Infrared Filter", min_value=0, max_value=120, step=1)
    
    # Predict button
    if st.button("Classify Stellar Object", type="primary"):
        st.session_state.show_prediction = True
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<h2 class="panel-title">Predicted Stellar Object</h2>', unsafe_allow_html=True)
    
    # Show prediction if button was clicked
    if hasattr(st.session_state, 'show_prediction') and st.session_state.show_prediction:
        input_data = {
            "alpha": alpha_selected,
            "delta": delta_selected,
            "red_shift": red_shift_selected,
            "UV_filter": UV_filter_selected,
            "green_filter": green_filter_selected,
            "red_filter": red_filter_selected,
            "IR_filter": ir_filter_selected,
            "near_IR_filter": near_ir_filter_selected,
        }

        df_input = input_df = pd.DataFrame([input_data])
        prediction = model.predict(df_input)[0]

        if prediction == "QSO":
            st.markdown('<div class="prediction-result">', unsafe_allow_html=True)
            st.markdown('<h3 class="prediction-title">Quasar Object</h3>', unsafe_allow_html=True)
            # Display quasar image using st.image with URL
            st.image("https://upload.wikimedia.org/wikipedia/commons/3/38/Artist%27s_rendering_ULAS_J1120%2B0641.jpg", 
                    caption="Quasar Visualization", 
                    use_container_width=True)
            
            # Prediction result
            st.markdown('<p class="prediction-description">Active Galactic Nucleus (AGN), a galaxy with an extremely bright core caused by the light emitted as matter falls into a central supermassive black hole</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        elif prediction == "STAR":
            st.markdown('<div class="prediction-result">', unsafe_allow_html=True)
            st.markdown('<h3 class="prediction-title">Star</h3>', unsafe_allow_html=True)
            # Display quasar image using st.image with URL
            st.image("https://static.scientificamerican.com/dam/m/1299c71322768b82/original/star_tyc_3203-450-1_in_lacerta_lizard_constellation.jpg?m=1747336540.73&w=1200", 
                    caption="Quasar Visualization", 
                    use_container_width=True)
            
            # Prediction result
            st.markdown('<p class="prediction-description">Active Galactic Nucleus (AGN), a galaxy with an extremely bright core caused by the light emitted as matter falls into a central supermassive black hole</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        elif prediction == "GALAXY":
            st.markdown('<div class="prediction-result">', unsafe_allow_html=True)
            st.markdown('<h3 class="prediction-title">Galaxy</h3>', unsafe_allow_html=True)
            # Display quasar image using st.image with URL
            st.image("https://i.natgeofe.com/n/e484088d-3334-4ab6-9b75-623f7b8505c9/1086.jpg", 
                    caption="Quasar Visualization", 
                    use_container_width=True)
            
            # Prediction result
            st.markdown('<p class="prediction-description">Active Galactic Nucleus (AGN), a galaxy with an extremely bright core caused by the light emitted as matter falls into a central supermassive black hole</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="prediction-result">', unsafe_allow_html=True)
            st.markdown(f'<h3 class="prediction-title">{prediction} Object</h3>', unsafe_allow_html=True)
            st.markdown('<p class="prediction-description">This is a different type of stellar object.</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<p style="color: white; text-align: center; font-style: italic;">Enter values and click "Classify Stellar Object" to see the prediction</p>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
        


# Footer
st.markdown("""
<div style="text-align: center; color: white; margin-top: 2rem; padding: 1rem; opacity: 0.7;">
    <p>© 2024 IdentiStar - Advanced Stellar Classification System</p>
</div>
""", unsafe_allow_html=True)

# st.title("Stellar Object Classifier")

# #html_file ="index.html"
# #css_file = "style.css"
# # Inject custom CSS
# #def load_css(file_name):
# #   with open(file_name) as f:
# #        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# # Inject custom HTML#
# # def load_html(file_name):
#     #with open(file_name, 'r') as f:
#         #html = f.read()
#         #st.markdown(html, unsafe_allow_html=True)

# # calling input/form elements
# # Using number inputs to control the range of value the user is entering
# alpha_selected = st.number_input("Enter Alpha", min_value=0, max_value=120, step=1)
# delta_selected = st.number_input("Enter Delta", min_value=0, max_value=120, step=1)
# red_shift_selected = st.number_input("Enter Red Shift", min_value=0, max_value=120, step=1)
# UV_filter_selected = st.number_input("Enter UV Filter", min_value=0, max_value=120, step=1)
# green_filter_selected = st.number_input("Enter Green Filter", min_value=0, max_value=120, step=1)
# red_filter_selected = st.number_input("Enter Red Filter", min_value=0, max_value=120, step=1)
# ir_filter_selected = st.number_input("Enter Infrared Filter", min_value=0, max_value=120, step=1)
# near_ir_filter_selected = st.number_input("Enter Near Infrared Filter", min_value=0, max_value=120, step=1)

# # Check if inputs are empty before entering them into the model

# # Predict button

# if st.button("Predict Stellar Object"):

#     input_data = {
#         "alpha": alpha_selected,
#         "delta": delta_selected,
#         "red_shift": red_shift_selected,
#         "UV_filter": UV_filter_selected,
#         "green_filter": green_filter_selected,
#         "red_filter": red_filter_selected,
#         "IR_filter": ir_filter_selected,
#         "near_IR_filter": near_ir_filter_selected,
#     }

#     df_input = input_df = pd.DataFrame([input_data])
#     prediction = model.predict(df_input)[0]
#     st.success(f"Predicted Stellar Object: {prediction}")

# st.markdown(
#     f"""
#     <style>
#     .stApp {{
#         background-image: url("https://www.mccormick.northwestern.edu/images/news/2023/07/what-does-a-twinkling-star-sound-like-take-a-listen-social.jpg");
#         background-size: cover
#     }}
#     .stApp::before {{
#         content: "";
#         position: fixed;
#         top: 0; left: 0; right: 0; bottom: 0;
#         background: rgba(60, 40, 20, 0.25); /* Adjust alpha for lighter/darker */
#         pointer-events: none;
#         z-index: 0;
#     }}
#     </style>
#     """,
#     unsafe_allow_html= True
# )
