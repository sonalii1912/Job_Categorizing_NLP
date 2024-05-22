import streamlit as st
import pickle
import re
import nltk
import re

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load the models
with open('clf.pkl', 'rb') as file:
    clf = pickle.load(file)

with open('tfidf.pkl', 'rb') as file:
    tfidfd = pickle.load(file)

# Define function to clean resume text
def cleanResume(txt):
    cleanText = re.sub(r'http\S+\s', ' ', txt)
    cleanText = re.sub(r'RT|cc', ' ', cleanText)
    cleanText = re.sub(r'#\S+\s*', ' ', cleanText)
    cleanText = re.sub(r'@\S+', '  ', cleanText)  
    cleanText = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText) 
    cleanText = re.sub(r'\s+', ' ', cleanText)
    return cleanText

# Streamlit app
def main():
    st.title("Resume Screening App")
    st.markdown("""
    This app predicts the job category based on the uploaded resume.
    """)

    # Upload resume
    uploaded_file = st.file_uploader('Upload Resume', type=['txt', 'pdf'])

    if uploaded_file is not None:
        try:
            resume_bytes = uploaded_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            resume_text = resume_bytes.decode('latin-1')

        cleaned_resume = cleanResume(resume_text)
        input_features = tfidfd.transform([cleaned_resume])
        prediction_id = clf.predict(input_features)[0]

        # Map category ID to category name
        category_mapping = {
            15: "Java Developer",
            23: "Testing",
            8: "DevOps Engineer",
            20: "Python Developer",
            24: "Web Designing",
            12: "HR",
            13: "Hadoop",
            3: "Blockchain",
            10: "ETL Developer",
            18: "Operations Manager",
            6: "Data Science",
            22: "Sales",
            16: "Mechanical Engineer",
            1: "Arts",
            7: "Database",
            11: "Electrical Engineering",
            14: "Health and fitness",
            19: "PMO",
            4: "Business Analyst",
            9: "DotNet Developer",
            2: "Automation Testing",
            17: "Network Security Engineer",
            21: "SAP Developer",
            5: "Civil Engineer",
            0: "Advocate"
        }

        category_name = category_mapping.get(prediction_id, "Unknown")

        # Display prediction
        st.success(f"Predicted Category: {category_name}")

    # Add custom CSS for styling
    st.markdown("""
    <style>
    body {
        font-family: Arial, sans-serif;
        background-color: #f0f0f0; /* Light grey background */
    }
    .stApp {
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
        background-color: #ffffff; /* White container background */
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background-color: #007bff; /* Blue button background */
        color: #ffffff; /* White button text */
        border-radius: 5px;
        padding: 8px 15px;
        font-size: 16px;
    }
    .stTextInput>div>div>div>input {
        font-size: 16px;
        padding: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Add a footer
    st.markdown("""
    ---
    Created by Aparna Hatte
    """)

# Run the app
if __name__ == "__main__":
    main()
