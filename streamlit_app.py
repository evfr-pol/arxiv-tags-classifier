from model import Prediction

import streamlit as st

@st.cache_resource
def load_model():
    model_preds = Prediction(768, 8, 0.2, './base_model.pt')
    return model_preds


model = load_model()
st.title("Article Themes by Title and Abstract")

title = st.text_input("Title", placeholder="Write article's title...")

abstract = st.text_area(
    "Abstract", 
    height=250,
    placeholder="Write article's abstract..."
)

label_full_names = {
    'cs': 'Computer Science',
    'econ': 'Economics',
    'eess': 'Electrical Engineering',
    'math': 'Mathematics',
    'q-bio': 'Quantitative Biology',
    'q-fin': 'Quantitative Finance',
    'stat': 'Statistics',
    'phys': 'Physics',
}

if st.button("Predict"):
    if title:
        labels = model.get_labels(title, abstract)
        if labels:
            result_text = ', '.join([label_full_names.get(label, label) for label in labels])
            st.success(result_text)
        else:
            st.warning("No labels predicted.")
    else:
        st.error("Please, fill title fields")
