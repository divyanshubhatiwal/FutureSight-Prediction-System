import os
import streamlit as st

# Custom imports 
from pages.multipage import MultiPage
from modules import data_upload, machine_learning, data_quality, data_visualize # import your modules here

# Create an instance of the app 
app = MultiPage()

# Configure page settings
st.set_page_config(
    page_title="Futuresight",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title of the main page
st.title("📊 Futuresight Prediction System")
st.caption("Transform your data into insights with powerful visualization and ML")

# Add all your application here
app.add_page("Upload Data", data_upload.app)
app.add_page("Data Quality & Metadata", data_quality.app, requires=["Upload Data"])
app.add_page("Machine Learning", machine_learning.app, requires=["Upload Data", "Data Quality & Metadata"])
app.add_page("Data Analysis", data_visualize.app, requires=["Upload Data"])

# The main app
app.run()
