import streamlit as st

st.set_page_config(
    page_title="Instrument Classifier",
    page_icon="🎵",
)

st.write("# Instrument Prediction! 👋")

st.sidebar.success("Select a page to learn more.")

st.markdown(
    """
    Using the freesound.org library, I've built a neural network to classify audio files as instruments.
    This only works for simple files of single instruments, but will be expanded to multi-instrument transcription.

    **👈 Select a tab from the sidebar** to learn more!
"""
)