import streamlit as st

def apply_custom_theme():
    """Injects custom CSS to heavily modify the Streamlit UI into a premium, AI-focused dark theme."""
    custom_css = """
    <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap');

        /* Global Typography and Background */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            background-color: #0A0A0F; /* Deep rich dark background */
            color: #E2E8F0; /* Clean white-gray text */
        }

        /* Sidebar Styling (Glassmorphism) */
        [data-testid="stSidebar"] {
            background-color: rgba(15, 15, 20, 0.65) !important;
            backdrop-filter: blur(16px) saturate(180%);
            -webkit-backdrop-filter: blur(16px) saturate(180%);
            border-right: 1px solid rgba(255, 255, 255, 0.05);
        }

        /* Top Header Styling */
        header[data-testid="stHeader"] {
            background-color: rgba(10, 10, 15, 0.8) !important;
            backdrop-filter: blur(10px);
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        }

        /* Primary Buttons */
        .stButton>button {
            background: linear-gradient(135deg, #00C9FF 0%, #92FE9D 100%);
            color: #0A0A0F !important;
            font-weight: 600;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0, 201, 255, 0.3);
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0, 201, 255, 0.5);
            color: #0A0A0F !important;
            border: none;
        }

        /* Secondary / Default Buttons */
        .stButton>button[kind="secondary"] {
            background: rgba(255, 255, 255, 0.05);
            color: #E2E8F0 !important;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: none;
        }
        .stButton>button[kind="secondary"]:hover {
            background: rgba(255, 255, 255, 0.1);
            border-color: rgba(255, 255, 255, 0.2);
            color: #FFFFFF !important;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        }

        /* Inputs and Text Areas */
        .stTextInput>div>div>input, 
        .stTextArea>div>div>textarea,
        .stSelectbox>div>div>div {
            background-color: rgba(20, 20, 25, 0.7) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 8px;
            color: #E2E8F0 !important;
            transition: all 0.2s ease;
        }

        .stTextInput>div>div>input:focus, 
        .stTextArea>div>div>textarea:focus,
        .stSelectbox>div>div>div:focus {
            border-color: #00C9FF !important;
            box-shadow: 0 0 0 1px #00C9FF !important;
        }

        /* Headings */
        h1, h2, h3, h4 {
            color: #FFFFFF !important;
            font-weight: 700;
            letter-spacing: -0.02em;
        }

        h1 {
            background: -webkit-linear-gradient(0deg, #FFFFFF, #A0AEC0);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        /* Info/Success/Error Banners */
        .stAlert {
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
        }
        
        .st-emotion-cache-1pxazr7 { /* Success */
            background-color: rgba(146, 254, 157, 0.1) !important;
            color: #92FE9D !important;
            border-color: rgba(146, 254, 157, 0.2) !important;
        }
        
        .st-emotion-cache-1v0mbdj { /* Error */
            background-color: rgba(255, 75, 75, 0.1) !important;
            color: #FF4B4B !important;
            border-color: rgba(255, 75, 75, 0.2) !important;
        }

        /* Dataframes & Tables */
        .stDataFrame {
            background-color: rgba(15, 15, 20, 0.4);
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }

        /* Code Blocks */
        code {
            font-family: 'JetBrains Mono', monospace;
            background-color: rgba(255, 255, 255, 0.08) !important;
            color: #00C9FF !important;
            border-radius: 4px;
            padding: 0.2em 0.4em;
        }
        
        pre {
            background-color: #050508 !important;
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 8px;
        }
        
        /* Metric Cards */
        [data-testid="stMetric"] {
            background-color: rgba(20, 20, 25, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 1rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s ease;
        }
        [data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            border-color: rgba(0, 201, 255, 0.3);
        }

        /* Scrollbars */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #0A0A0F;
        }
        ::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.2);
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(255, 255, 255, 0.3);
        }

        /* Main Container Padding */
        .main .block-container {
            padding-top: 2rem;
            max-width: 1200px;
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)
