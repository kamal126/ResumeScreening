from joblib import load
import re
import streamlit as st
import fitz  # PyMuPDF

# ------------------------------
# Load model and vectorizer
# ------------------------------
model = load('resume_classifier_model.pkl')
vectorizer = load('tfidf_vectorizer.pkl')

# ------------------------------
# Category mapping
# ------------------------------
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

# ------------------------------
# Helper functions
# ------------------------------
def clean_resume(text):
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'RT|cc', ' ', text)
    text = re.sub(r'#\S+', '', text)
    text = re.sub(r'@\S+', ' ', text)
    text = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)
    text = re.sub(r'[^\x00-\x7f]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def predict_category(text):
    cleaned_text = clean_resume(text)
    vectorized_input = vectorizer.transform([cleaned_text])
    prediction = model.predict(vectorized_input)[0]
    return category_map.get(prediction, f"Category #{prediction}")

def extract_text_from_pdf(uploaded_file):
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("🧠 Resume Category Classifier")
st.markdown("Paste your resume text or upload a PDF to predict the category.")

# ----- Text input -----
user_input = st.text_area("Paste Resume Text Here", height=300)

if st.button("Predict Category from Text"):
    if not user_input.strip():
        st.warning("Please enter some resume content!")
    else:
        category = predict_category(user_input)
        st.success(f"✅ Predicted Resume Category: **{category}**")

# ----- PDF upload -----
uploaded_file = st.file_uploader("Or upload PDF Resume", type=['pdf'])

if uploaded_file:
    with st.spinner("Extracting text from PDF..."):
        extracted_text = extract_text_from_pdf(uploaded_file)
    st.text_area("Extracted Text from PDF", value=extracted_text, height=300)

    if st.button("Predict Category from PDF"):
        category = predict_category(extracted_text)
        st.success(f"✅ Predicted Resume Category from PDF: **{category}**")
