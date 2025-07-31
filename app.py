import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import time
import random
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Fix for NumPy 2.0+ compatibility - add bool8 alias
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

# Load model
model = joblib.load("model.pkl")

# Page configuration
st.set_page_config(
    page_title="IdentiStar - Complete Stellar Explorer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    /* Global styles */
    * {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Main content spacing */
    .main .block-container {
        padding-top: 140px;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    
    .css-1d391kg .sidebar-content {
        padding: 1rem;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(90deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
        padding: 0;
        border-radius: 0;
        margin: 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        overflow: hidden;
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        z-index: 1000;
        border-bottom: 2px solid rgba(255,255,255,0.1);
    }
    
    .navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem 2rem;
        background: linear-gradient(90deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    .navbar-brand {
        display: flex;
        flex-direction: column;
        align-items: flex-start;
    }
    
    .header-title {
        font-size: 2.2rem;
        color: #ffffff;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        font-weight: 700;
        letter-spacing: 1px;
    }
    
    .navbar-subtitle {
        color: rgba(255,255,255,0.9);
        font-size: 0.95rem;
        margin-top: 0.25rem;
        font-style: italic;
        font-weight: 300;
    }
    
    .navbar-stats {
        display: flex;
        align-items: center;
    }
    
    .stat-item {
        display: flex;
        flex-direction: column;
        align-items: center;
        background: rgba(255,255,255,0.1);
        padding: 0.75rem 1rem;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.2);
        backdrop-filter: blur(10px);
        min-width: 80px;
    }
    
    .stat-number {
        color: #ffd700;
        font-size: 1.4rem;
        font-weight: bold;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }
    
    .stat-label {
        color: rgba(255,255,255,0.9);
        font-size: 0.75rem;
        margin-top: 0.25rem;
        font-weight: 500;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .navbar {
            flex-direction: column;
            gap: 1rem;
            padding: 1rem;
        }
        
        .navbar-menu {
            justify-content: center;
            flex-wrap: wrap;
        }
        
        .nav-link {
            padding: 0.5rem 1rem;
            font-size: 0.8rem;
        }
        
        .header-title {
            font-size: 1.8rem;
        }
        
        .main .block-container {
            padding-top: 160px;
        }
    }
    
    /* Content cards and panels */
    .panel {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.1);
        margin-bottom: 1.5rem;
        backdrop-filter: blur(10px);
    }
    
    .panel-title {
        color: #ffffff;
        font-size: 1.6rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
        text-align: center;
        letter-spacing: 0.5px;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.3);
        border: 1px solid rgba(255,255,255,0.1);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #8080ff 0%, #9999ff 100%);
        padding: 2rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(240, 147, 251, 0.3);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .info-box {
        background: rgba(255, 255, 255, 0.05);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .info-box h4 {
        color: #ffffff;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    
    .info-box p {
        color: rgba(255,255,255,0.9);
        line-height: 1.6;
        margin: 0;
    }
    
    .game-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 6px 20px rgba(79, 172, 254, 0.3);
        border: 1px solid rgba(255,255,255,0.1);
        transition: transform 0.3s ease;
    }
    
    .game-card:hover {
        transform: translateY(-2px);
    }
    
    /* Form inputs */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        background-color: rgba(255,255,255,0.1) !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        border-radius: 8px !important;
        padding: 0.75rem !important;
        color: #ffffff !important;
        backdrop-filter: blur(10px);
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.3) !important;
        border-color: rgba(102, 126, 234, 0.5) !important;
    }
    
    .stTextInput > div > div > input::placeholder,
    .stNumberInput > div > div > input::placeholder {
        color: rgba(255,255,255,0.6) !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
    }
    
    /* Prediction results */
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
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }
    
    .prediction-description {
        font-size: 1rem;
        line-height: 1.6;
        margin-top: 1rem;
        color: rgba(255,255,255,0.9);
    }

    /* Images */
    .stImage img {
        border-radius: 12px !important;
        border: 2px solid rgba(255, 255, 255, 0.2) !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3) !important;
        align-items: center !important;
    }
    
    .stImage {
        border-radius: 12px;
        overflow: hidden;
    }
    
    /* Headers and text */
    h1, h2, h3 {
        color: #ffffff !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px !important;
    }
    
    h1 {
        font-size: 2.5rem !important;
        margin-bottom: 1.5rem !important;
    }
    
    h2 {
        font-size: 2rem !important;
        margin-bottom: 1rem !important;
    }
    
    h3 {
        font-size: 1.5rem !important;
        margin-bottom: 0.75rem !important;
    }
    
    p {
        color: rgba(255,255,255,0.9) !important;
        line-height: 1.6 !important;
    }
    
    /* Dataframes */
    .stDataFrame {
        background: rgba(255,255,255,0.05) !important;
        border-radius: 8px !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
    }
    
    /* Success/Error messages */
    .stSuccess {
        background: rgba(76, 175, 80, 0.1) !important;
        border: 1px solid rgba(76, 175, 80, 0.3) !important;
        border-radius: 8px !important;
        color: #4CAF50 !important;
    }
    
    .stError {
        background: rgba(244, 67, 54, 0.1) !important;
        border: 1px solid rgba(244, 67, 54, 0.3) !important;
        border-radius: 8px !important;
        color: #F44336 !important;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255,255,255,0.1);
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(102, 126, 234, 0.5);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(102, 126, 234, 0.7);
    }
    
    /* Graph and chart styling */
    .stPlotlyChart {
        background: rgba(255,255,255,0.05) !important;
        border-radius: 16px !important;
        padding: 1rem !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2) !important;
        backdrop-filter: blur(10px) !important;
        margin: 1rem 0 !important;
        overflow: hidden !important;
    }
    
    .stPlotlyChart > div {
        border-radius: 12px !important;
        overflow: hidden !important;
        width: 100% !important;
        height: auto !important;
    }
    
    /* Ensure Plotly charts have proper sizing */
    .stPlotlyChart .js-plotly-plot {
        width: 100% !important;
        height: auto !important;
        overflow: hidden !important;
    }
    
    .stPlotlyChart .plotly {
        width: 100% !important;
        height: auto !important;
        overflow: hidden !important;
    }
    

    
    
    /* Metric containers */
    .stMetric {
        background: rgba(255,255,255,0.05) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2) !important;
        backdrop-filter: blur(10px) !important;
        margin: 0.5rem 0 !important;
    }
    
    /* All plot containers */
    [data-testid="stPlotlyChart"],
    [data-testid="stDataFrame"],
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.05) !important;
        border-radius: 16px !important;
        padding: 1rem !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2) !important;
        backdrop-filter: blur(10px) !important;
        margin: 1rem 0 !important;
        overflow: visible !important;
    }
    
    /* Chart titles and labels */
    .stPlotlyChart .js-plotly-plot .plotly .main-svg {
        border-radius: 12px !important;
    }
S
    }
</style>
""", unsafe_allow_html=True)

# Generate starry background
st.markdown(
    f"""
    <style>
    .stApp {{
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 25%, #16213e 50%, #0f3460 75%, #533483 100%);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
    }}
    
    @keyframes gradientShift {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}
    
    .stApp::before {{
        content: "";
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background: 
            radial-gradient(2px 2px at 20px 30px, #ffffff, transparent),
            radial-gradient(2px 2px at 40px 70px, rgba(255,255,255,0.8), transparent),
            radial-gradient(1px 1px at 90px 40px, #ffffff, transparent),
            radial-gradient(1px 1px at 130px 80px, rgba(255,255,255,0.6), transparent),
            radial-gradient(2px 2px at 160px 30px, #ffffff, transparent);
        background-repeat: repeat;
        background-size: 200px 100px;
        animation: twinkle 4s ease-in-out infinite;
        pointer-events: none;
        z-index: 0;
        opacity: 0.3;
    }}
    
    @keyframes twinkle {{
        0%, 100% {{ opacity: 0.3; }}
        50% {{ opacity: 0.6; }}
    }}
    </style>
    """, unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("star_classification_dataset.csv")
    # Basic cleaning
    df['alpha'] = pd.to_numeric(df['alpha'], errors='coerce')
    df = df.dropna(subset=['alpha'])
    df = df.dropna()
    
    # Fix for numpy bool8 deprecation - use explicit boolean conversion
    mask = (df != -9999.000000).all(axis=1)
    df = df[mask]
    
    return df

# Load data
df = load_data()

# Sidebar navigation
st.sidebar.title("🚀 Navigation")

# Initialize selected_page in session state if not exists
if 'selected_page' not in st.session_state:
    st.session_state.selected_page = "🏠 Home Dashboard"

page = st.sidebar.selectbox(
    "Choose your exploration mode:",
    ["🏠 Home Dashboard", "🔬 Stellar Classification", "📊 Data Explorer"],
    index=["🏠 Home Dashboard", "🔬 Stellar Classification", "📊 Data Explorer"].index(st.session_state.selected_page)
)

# Update session state when page changes
if page != st.session_state.selected_page:
    st.session_state.selected_page = page



# Initialize session state for games and classification history
if 'quiz_score' not in st.session_state:
    st.session_state.quiz_score = 0
if 'total_questions' not in st.session_state:
    st.session_state.total_questions = 0
if 'quiz_history' not in st.session_state:
    st.session_state.quiz_history = []
if 'classification_history' not in st.session_state:
    st.session_state.classification_history = []

if page == "🏠 Home Dashboard":
    st.header("🌌 Welcome to IdentiStar!")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Total Objects</h3>
            <h2>{len(df):,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Galaxies</h3>
            <h2>{len(df[df['class'] == 'GALAXY']):,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Stars</h3>
            <h2>{len(df[df['class'] == 'STAR']):,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Quasars</h3>
            <h2>{len(df[df['class'] == 'QSO']):,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick start section
    st.subheader("🚀 Quick Start")
    
    col1, = st.columns(1)
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h4>🔬 Try Stellar Classification</h4>
            <p>Input stellar object properties and get instant classifications with confidence scores.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔬 Start Classification", type="primary"):
            st.session_state.selected_page = "🔬 Stellar Classification"
            st.rerun()
    
    # Interactive 3D scatter plot
    st.subheader("🌌 Interactive 3D Stellar Map")
    
    # Feature selection for 3D plot
    col1, col2, col3 = st.columns(3)
    with col1:
        x_feature = st.selectbox("X-axis:", ['red_shift', 'UV_filter', 'green_filter', 'red_filter', 'IR_filter'])
    with col2:
        y_feature = st.selectbox("Y-axis:", ['UV_filter', 'red_shift', 'green_filter', 'red_filter', 'IR_filter'])
    with col3:
        z_feature = st.selectbox("Z-axis:", ['green_filter', 'red_shift', 'UV_filter', 'red_filter', 'IR_filter'])
    
    # Sample data for performance
    sample_size = st.slider("Sample size for visualization:", 1000, 10000, 5000)
    df_sample = df.sample(sample_size, random_state=42)
    
    # Create 3D scatter plot
    fig_3d = px.scatter_3d(
        df_sample, 
        x=x_feature, 
        y=y_feature, 
        z=z_feature,
        color='class',
        title=f"3D Stellar Classification Map: {x_feature} vs {y_feature} vs {z_feature}",
        color_discrete_map={'GALAXY': '#1f77b4', 'STAR': '#ff7f0e', 'QSO': '#2ca02c'}
    )
    
    fig_3d.update_layout(
        scene=dict(
            xaxis_title=x_feature.replace('_', ' ').title(),
            yaxis_title=y_feature.replace('_', ' ').title(),
            zaxis_title=z_feature.replace('_', ' ').title(),
        ),
        height=600
    )
    
    st.plotly_chart(fig_3d, use_container_width=True)
    
    # Recent activity
    st.subheader("📈 Recent Activity")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Redshift distribution by class
        fig_redshift = px.histogram(
            df_sample, 
            x='red_shift', 
            color='class',
            title="Redshift Distribution by Stellar Class",
            nbins=50
        )
        st.plotly_chart(fig_redshift, use_container_width=True)
    
    with col2:
        # Filter magnitudes distribution
        fig_filters = px.box(
            df_sample, 
            x='class', 
            y='UV_filter',
            title="UV Filter Magnitude by Class"
        )
        st.plotly_chart(fig_filters, use_container_width=True)
    
    # Recent classification history (if any)
    if st.session_state.classification_history:
        st.subheader("📚 Your Recent Classifications")
        
        # Show last 3 classifications
        recent_classifications = st.session_state.classification_history[-3:]
        
        col1, col2, col3 = st.columns(3)
        for i, entry in enumerate(reversed(recent_classifications)):
            with [col1, col2, col3][i]:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{entry['prediction']}</h3>
                    <h4>{entry['confidence']:.1f}%</h4>
                    <p style="font-size: 0.8rem; opacity: 0.8;">{entry['timestamp']}</p>
                </div>
                """, unsafe_allow_html=True)

elif page == "🔬 Stellar Classification":
    st.header("🔬 Live Stellar Classification")
    
    # Interactive input section
    st.markdown("""
    <div class="info-box">
        <h4>💡 How it works:</h4>
        <p>Enter the observed values for a stellar object and watch our AI classify it in real-time!</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h2 class="panel-title">Enter values for Stellar Classification</h2>', unsafe_allow_html=True)
        
        # Using number inputs to control the range of value the user is entering
        alpha_selected = st.number_input("Enter Alpha", min_value=0.0, max_value=360.0, step=0.01)
        delta_selected = st.number_input("Enter Delta", min_value=-90.0, max_value=90.0, step=0.01)
        red_shift_selected = st.number_input("Enter Red Shift", min_value=0.0, max_value=8.0, step=0.01)
        UV_filter_selected = st.number_input("Enter UV Filter", min_value=15.0, max_value=30.0, step=0.01)
        green_filter_selected = st.number_input("Enter Green Filter", min_value=14.0, max_value=30.0, step=0.01)
        red_filter_selected = st.number_input("Enter Red Filter", min_value=13.0, max_value=30.0, step=0.01)
        ir_filter_selected = st.number_input("Enter Infrared Filter", min_value=12.0, max_value=30.0, step=0.01)
        near_ir_filter_selected = st.number_input("Enter Near Infrared Filter", min_value=12.5, max_value=30.0, step=0.01)
        
        # Predict button
        if st.button("Classify Stellar Object", type="primary"):
            st.session_state.show_prediction = True
    
    with col2:
        st.markdown('<h2 class="panel-title">Predicted Stellar Object</h2>', unsafe_allow_html=True)
        
        # Show prediction if button was clicked
        if hasattr(st.session_state, 'show_prediction') and st.session_state.show_prediction:
            input_data = {
                "UV_filter": UV_filter_selected,
                "green_filter": green_filter_selected,
                "red_filter": red_filter_selected,
                "IR_filter": ir_filter_selected,
                "near_IR_filter": near_ir_filter_selected,
                "alpha": alpha_selected,
                "delta": delta_selected,
                "red_shift": red_shift_selected,
            }

            df_input = pd.DataFrame([input_data])
            prediction = model.predict(df_input)[0]
            prediction_proba = model.predict_proba(df_input)[0]
            
            # Add to classification history
            history_entry = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'prediction': prediction,
                'confidence': max(prediction_proba) * 100,
                'input_data': input_data.copy()
            }
            st.session_state.classification_history.append(history_entry)

            # Display results with confidence
            st.markdown(f"""
            <div class="prediction-card">
                <h2>Predicted Class: {prediction}</h2>
                <h3>Confidence: {max(prediction_proba)*100:.1f}%</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence breakdown
            classes = model.classes_
            confidence_df = pd.DataFrame({
                'Class': classes,
                'Confidence': prediction_proba * 100
            })
            
            # fig_confidence = px.bar(
            #     confidence_df, 
            #     x='Class', 
            #     y='Confidence',
            #     title="Classification Confidence by Class",
            #     color='Confidence',
            #     color_continuous_scale='viridis'
            # )
            # st.plotly_chart(fig_confidence, use_container_width=True)

            if prediction == "QSO":
                st.markdown('<div class="prediction-result">', unsafe_allow_html=True)
                st.image("https://upload.wikimedia.org/wikipedia/commons/3/38/Artist%27s_rendering_ULAS_J1120%2B0641.jpg", 
                        caption="Quasar Visualization", 
                        use_container_width=True)
                
                st.markdown('<p class="prediction-description">Active Galactic Nucleus (AGN), a galaxy with an extremely bright core caused by the light emitted as matter falls into a central supermassive black hole</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            elif prediction == "STAR":
                st.markdown('<div class="prediction-result">', unsafe_allow_html=True)
                st.image("https://static.scientificamerican.com/dam/m/1299c71322768b82/original/star_tyc_3203-450-1_in_lacerta_lizard_constellation.jpg?m=1747336540.73&w=1200", 
                        caption="Star Visualization", 
                        use_container_width=True)
                
                st.markdown('<p class="prediction-description">A massive ball of plasma held together by gravity, primarily composed of hydrogen and helium, that emits light through nuclear fusion</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            elif prediction == "GALAXY":
                st.markdown('<div class="prediction-result">', unsafe_allow_html=True)
                st.image("https://i.natgeofe.com/n/e484088d-3334-4ab6-9b75-623f7b8505c9/1086.jpg", 
                        caption="Galaxy Visualization", 
                        use_container_width=True)
                
                st.markdown('<p class="prediction-description">A vast collection of stars, gas, dust, and dark matter bound together by gravity, containing billions to trillions of stars</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="prediction-result">', unsafe_allow_html=True)
                st.markdown(f'<h3 class="prediction-title">{prediction} Object</h3>', unsafe_allow_html=True)
                st.markdown('<p class="prediction-description">This is a different type of stellar object.</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<p style="color: white; text-align: center; font-style: italic;">Enter values and click "Classify Stellar Object" to see the prediction</p>', unsafe_allow_html=True)
    
    # Classification History Section
    if st.session_state.classification_history:
        st.subheader("📚 Classification History")
        
        # History controls
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown("""
            <div class="info-box">
                <h4>📊 Your Classification Journey</h4>
                <p>Review your past classifications and track your exploration of stellar objects.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)  # Add some spacing
            if st.button("🗑️ Clear History", type="secondary"):
                st.session_state.classification_history = []
                st.rerun()
        
        # Display history in reverse chronological order
        for i, entry in enumerate(reversed(st.session_state.classification_history[-10:])):  # Show last 10 entries
            with st.expander(f"🔍 {entry['prediction']} - {entry['timestamp']} (Confidence: {entry['confidence']:.1f}%)"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📊 Input Parameters:**")
                    input_data = entry['input_data']
                    st.markdown(f"""
                    - **Alpha:** {input_data['alpha']:.2f}°
                    - **Delta:** {input_data['delta']:.2f}°
                    - **Redshift:** {input_data['red_shift']:.4f}
                    - **UV Filter:** {input_data['UV_filter']:.2f}
                    - **Green Filter:** {input_data['green_filter']:.2f}
                    - **Red Filter:** {input_data['red_filter']:.2f}
                    - **IR Filter:** {input_data['IR_filter']:.2f}
                    - **Near IR Filter:** {input_data['near_IR_filter']:.2f}
                    """)
                
                with col2:
                    st.markdown("**🎯 Classification Result:**")
                    st.markdown(f"""
                    - **Predicted Class:** {entry['prediction']}
                    - **Confidence:** {entry['confidence']:.1f}%
                    - **Timestamp:** {entry['timestamp']}
                    """)
                    
                    # Show the appropriate image for the prediction
                    if entry['prediction'] == "QSO":
                        st.image("https://upload.wikimedia.org/wikipedia/commons/3/38/Artist%27s_rendering_ULAS_J1120%2B0641.jpg", 
                                caption="Quasar", width=150)
                    elif entry['prediction'] == "STAR":
                        st.image("https://static.scientificamerican.com/dam/m/1299c71322768b82/original/star_tyc_3203-450-1_in_lacerta_lizard_constellation.jpg?m=1747336540.73&w=1200", 
                                caption="Star", width=150)
                    elif entry['prediction'] == "GALAXY":
                        st.image("https://i.natgeofe.com/n/e484088d-3334-4ab6-9b75-623f7b8505c9/1086.jpg", 
                                caption="Galaxy", width=150)
        
        # History statistics
        if len(st.session_state.classification_history) > 1:
            st.subheader("📈 Classification Statistics")
            
            # Calculate statistics
            predictions = [entry['prediction'] for entry in st.session_state.classification_history]
            confidences = [entry['confidence'] for entry in st.session_state.classification_history]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Classifications", len(st.session_state.classification_history))
            
            with col2:
                st.metric("Average Confidence", f"{sum(confidences)/len(confidences):.1f}%")
            
            with col3:
                most_common = max(set(predictions), key=predictions.count)
                st.metric("Most Common Class", most_common)
            
            with col4:
                unique_classes = len(set(predictions))
                st.metric("Classes Discovered", unique_classes)
            
            # Classification distribution chart
            prediction_counts = {}
            for pred in predictions:
                prediction_counts[pred] = prediction_counts.get(pred, 0) + 1
            
            fig_history = px.pie(
                values=list(prediction_counts.values()),
                names=list(prediction_counts.keys()),
                title="Your Classification Distribution",
                color_discrete_map={'GALAXY': '#1f77b4', 'STAR': '#ff7f0e', 'QSO': '#2ca02c'}
            )
            st.plotly_chart(fig_history, use_container_width=True)
    

elif page == "📊 Data Explorer":
    st.header("📊 Interactive Data Explorer")
    
    # Data overview
    st.subheader("📈 Dataset Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Class distribution pie chart
        class_counts = df['class'].value_counts()
        fig_pie = px.pie(
            values=class_counts.values, 
            names=class_counts.index,
            title="Distribution of Stellar Classes"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Feature correlation heatmap
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlation_matrix = df[numeric_cols].corr()
        
        fig_heatmap = px.imshow(
            correlation_matrix,
            title="Feature Correlation Heatmap",
            color_continuous_scale='RdBu',
            aspect='auto'
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Interactive feature analysis
    st.subheader("🔍 Feature Analysis")
    
    feature = st.selectbox("Select feature to analyze:", 
                          ['red_shift', 'UV_filter', 'green_filter', 'red_filter', 'IR_filter', 'near_IR_filter'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution by class
        fig_dist = px.histogram(
            df, 
            x=feature, 
            color='class',
            title=f"{feature.replace('_', ' ').title()} Distribution by Class",
            nbins=30
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with col2:
        # Box plot
        fig_box = px.box(
            df, 
            x='class', 
            y=feature,
            title=f"{feature.replace('_', ' ').title()} by Class"
        )
        st.plotly_chart(fig_box, use_container_width=True)
    
    # Statistical summary
    st.subheader("📊 Statistical Summary")
    st.dataframe(df.groupby('class')[feature].describe())


elif page == "🌌 Cosmic Stories":
    st.header("🌌 Cosmic Stories & Educational Content")
    
    st.markdown("""
    <div class="info-box">
        <h4>🌟 Welcome to the Cosmic Classroom!</h4>
        <p>Learn about the fascinating objects in our universe and how we classify them.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Educational content
    st.subheader("📚 What are these objects?")
    
    object_type = st.selectbox("Choose an object type to learn about:", 
                              ["Stars", "Galaxies", "Quasars"])
    
    if object_type == "Stars":
        st.markdown("""
        ### ⭐ Stars
        **What they are:** Massive balls of plasma held together by gravity, primarily composed of hydrogen and helium.
        
        **Key characteristics:**
        - Emit light through nuclear fusion in their cores
        - Have varying sizes, temperatures, and colors
        - Most stars in our galaxy are red dwarfs
        - Our Sun is a yellow dwarf star
        
        **In our dataset:** Stars typically have low redshift values and specific magnitude patterns across different filters.
        """)
        
        # Show star examples
        star_examples = df[df['class'] == 'STAR'].sample(3)
        st.subheader("🌟 Example Stars from Our Dataset")
        st.dataframe(star_examples[['red_shift', 'UV_filter', 'green_filter', 'red_filter']])
    
    elif object_type == "Galaxies":
        st.markdown("""
        ### 🌌 Galaxies
        **What they are:** Vast collections of stars, gas, dust, and dark matter bound together by gravity.
        
        **Key characteristics:**
        - Contain billions to trillions of stars
        - Come in various shapes (spiral, elliptical, irregular)
        - Our Milky Way is a spiral galaxy
        - Most distant objects visible to telescopes
        
        **In our dataset:** Galaxies show a wide range of redshift values and have distinct spectral energy distributions.
        """)
        
        # Show galaxy examples
        galaxy_examples = df[df['class'] == 'GALAXY'].sample(3)
        st.subheader("🌌 Example Galaxies from Our Dataset")
        st.dataframe(galaxy_examples[['red_shift', 'UV_filter', 'green_filter', 'red_filter']])
    
    elif object_type == "Quasars":
        st.markdown("""
        ### ⚡ Quasars (QSOs)
        **What they are:** Extremely bright active galactic nuclei powered by supermassive black holes.
        
        **Key characteristics:**
        - Among the most luminous objects in the universe
        - Powered by accretion disks around supermassive black holes
        - Often found at very high redshifts (very distant)
        - Emit across the entire electromagnetic spectrum
        
        **In our dataset:** Quasars typically have high redshift values and very bright magnitudes in UV filters.
        """)
        
        # Show quasar examples
        quasar_examples = df[df['class'] == 'QSO'].sample(3)
        st.subheader("⚡ Example Quasars from Our Dataset")
        st.dataframe(quasar_examples[['red_shift', 'UV_filter', 'green_filter', 'red_filter']])
    
    # Interactive redshift explanation
    st.subheader("🔴 Understanding Redshift")
    
    redshift_value = st.slider("Adjust redshift value:", 0.0, 5.0, 1.0)
    
    # Calculate distance and age (simplified)
    distance_approx = redshift_value * 3.26  # billion light years (simplified)
    age_approx = redshift_value * 2.5  # billion years ago (simplified)
    
    st.markdown(f"""
    **Redshift {redshift_value:.2f} means:**
    - **Distance:** Approximately {distance_approx:.1f} billion light years away
    - **Age:** Light left the object about {age_approx:.1f} billion years ago
    - **Universe was:** About {13.8 - age_approx:.1f} billion years old when light was emitted
    
    *Note: These are simplified calculations for educational purposes.*
    """)
    
    # Show objects at this redshift
    nearby_objects = df[
        (df['red_shift'] >= redshift_value - 0.1) & 
        (df['red_shift'] <= redshift_value + 0.1)
    ]
    
    if not nearby_objects.empty:
        st.subheader(f"🔍 Objects at Similar Redshift ({redshift_value:.2f} ± 0.1)")
        redshift_dist = nearby_objects['class'].value_counts()
        fig_redshift_dist = px.pie(
            values=redshift_dist.values,
            names=redshift_dist.index,
            title=f"Object Types at Redshift ~{redshift_value:.2f}"
        )
        st.plotly_chart(fig_redshift_dist, use_container_width=True)

# Footer
st.markdown("""
<div style="text-align: center; color: white; margin-top: 2rem; padding: 1rem; opacity: 0.7;">
    <p>© 2024 IdentiStar - Complete Stellar Classification System</p>
</div>
""", unsafe_allow_html=True)
