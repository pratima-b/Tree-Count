import streamlit as st
from treeview import treeview_app
from green_cover import green_cover_app

# Set up the Streamlit page
st.set_page_config(page_title="Tree & Green Cover Analysis", page_icon="🌳", layout="wide")

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Go to", ["TreeView", "Green Cover Analysis"])

# Load the selected app based on the user's choice
if app_mode == "TreeView":
    treeview_app()
elif app_mode == "Green Cover Analysis":
    green_cover_app()
