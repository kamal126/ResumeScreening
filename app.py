import streamlit as st
import pickle
import re
import joblib
import numpy as np

# Load model
model = joblib.load('resume_classifier_model.pkl')

# Load vectorizer
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Optional: Label encoder mapping (reverse map)
category_map = {
    0: 'Data Science',
    1: 'HR',
    2: 'Advocate',
    3: 'Arts',
    4: 'Web Designing',
    5: 'Mechanical Engineer',
    6: 'Sales',
    7: 'Health and fitness',
    8: 'Civil Engineer',
    9: 'Java Developer',
    10: 'Business Analyst',
    11: 'SAP Developer',
    12: 'Automation Testing',
    13: 'Electrical Engineering',
    14: 'Operations Manager',
    15: 'Python Developer',
    16: 'DevOps Engineer',
    17: 'Network Security Engineer',
    18: 'PMO',
    19: 'Database',
    20: 'Hadoop',
    21: 'ETL Developer',
    22: 'DotNet Developer',
    23: 'Blockchain',
    24: 'Testing'
}

def clean_resume(text):
    text = re.sub('http\S+\s*', ' ', text)
    text = re.sub('RT|cc', ' ', text)
    text = re.sub('#\S+', '', text)
    text = re.sub('@\S+', ' ', text)
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)
    text = re.sub(r'[^\x00-\x7f]',r' ', text)
    # Update these lines like this:
    text = re.sub(r'http\S+\s*', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    return text

# Streamlit UI
st.title("🧠 Resume Category Classifier")
st.markdown("Paste your resume text below:")

user_input = st.text_area("Resume Text", height=300)

if st.button("Predict Category"):
    if user_input.strip() == "":
        st.warning("Please enter some resume content!")
    else:
        cleaned_text = clean_resume(user_input)
        vectorized_input = vectorizer.transform([cleaned_text])
        prediction = model.predict(vectorized_input)[0]

        category = category_map.get(prediction, f"Category #{prediction}")
        st.success(f"✅ Predicted Resume Category: **{category}**")


import fitz  # PyMuPDF

def extract_text_from_pdf(uploaded_file):
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text

uploaded_file = st.file_uploader("Upload PDF Resume", type=['pdf'])
if uploaded_file:
    extracted_text = extract_text_from_pdf(uploaded_file)
    st.text_area("Extracted Text", value=extracted_text, height=300)
    # You can pass this text to prediction
    if st.button("Predict Category from PDF"):
        cleaned_text = clean_resume(extracted_text)
        vectorized_input = vectorizer.transform([cleaned_text])
        prediction = model.predict(vectorized_input)[0]

        category = category_map.get(prediction, f"Category #{prediction}")
        st.success(f"✅ Predicted Resume Category from PDF: **{category}**")