import streamlit as st

def configure_page():
    st.set_page_config(page_title="Información Financiera", layout="wide", page_icon="📈", initial_sidebar_state="expanded")

def apply_custom_css():
    st.markdown("""
    <style>
    .reportview-container {
        background: #0e1117;
        color: #ffffff;
    }
    .sidebar .sidebar-content {
        background: #1e2130;
    }
    .Widget>label {
        color: #ffffff;
    }
    .stTextInput>div>div>input {
        background: #262730;
        color: #ffffff;
    }
    .stSelectbox>div>div>select {
        background: #262730;
        color: #ffffff;
    }
    .stDateInput>div>div>input {
        background: #262730;
        color: #ffffff;
    }
    .stTab {
        background-color: #1e2130;
        color: #ffffff;
        border-radius: 5px 5px 0 0;
    }
    .stTab[data-baseweb="tab"] {
        height: 50px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .stTab[aria-selected="true"] {
        background-color: #0e1117;
        border-bottom: 2px solid #FF6F61;
    }
    .stButton>button {
        background-color: #FF6F61;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 24px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #FF4F3D;
    }
    .card {
        background-color: #1e2130;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    h1, h2, h3 {
        color: #FF6F61;
    }
    </style>
    """, unsafe_allow_html=True)

plotly_config = {
    'template': 'plotly_dark',
}