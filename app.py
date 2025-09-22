import streamlit as st
import PyPDF2
import os
from dotenv import load_dotenv
import google.generativeai as genai
import io
import re

# Load environment variables from .env file
load_dotenv()

# Set up page configuration and title
st.set_page_config(page_title="AI Resume Critiquer", layout="centered")
st.title("AI Resume Critiquer")
st.markdown("Upload your resume as a PDF and get AI-powered feedback tailored to your needs!")

# Get the API key from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Check if the API key is set before proceeding
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    st.error("API Key not found. Please set the GOOGLE_API_KEY environment variable. You can find instructions in the app.py file.")
    st.info("💡 To fix this, create a `.env` file in the same directory as this script and add `GOOGLE_API_KEY='your_api_key_here'` inside it.")
    st.stop()

# --- Utility Functions ---

def get_text_from_pdf(pdf_file):
    """
    Extracts text from a PDF file object.
    """
    text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return None
    return text

def get_ai_feedback(prompt, resume_text):
    """
    Sends the resume text and user prompt to the Gemini Pro model and returns the feedback.
    """
    try:
        # Use the gemini-1.5-pro model as gemini-pro is deprecated for this use case
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(f"{prompt}\n\nResume Text:\n{resume_text}")
        return response.text
    except Exception as e:
        st.error(f"Error generating feedback: {e}")
        return None

# --- Streamlit UI ---

uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
job_description = st.text_area(
    "Paste the job description (optional)",
    placeholder="Example: 'Seeking a Python Developer with 5+ years of experience in web application development...'"
)

# Define a list of common resume critiques
critiques = [
    "Overall critique",
    "Tailor to a job description",
    "Grammar and spelling check",
    "Strengths and weaknesses",
    "Action verbs analysis"
]

# Use a selectbox for a professional look and feel
selected_critique = st.selectbox(
    "Choose the type of feedback you want:",
    options=critiques,
    index=0  # Default to "Overall critique"
)

# Button to trigger the analysis
submit_button = st.button("Get AI Feedback", use_container_width=True)

if submit_button and uploaded_file is not None:
    # Use a spinner while processing
    with st.spinner("Analyzing your resume..."):
        resume_bytes = uploaded_file.read()
        pdf_file = io.BytesIO(resume_bytes)
        resume_text = get_text_from_pdf(pdf_file)

        if resume_text:
            # Construct the prompt based on the user's selection
            base_prompt = "You are a world-class resume critiquer. Provide detailed, constructive feedback on the following resume. Be professional, direct, and actionable."
            
            if selected_critique == "Overall critique":
                prompt = f"{base_prompt} Provide a comprehensive overall critique of the resume, including its structure, clarity, and effectiveness."
            elif selected_critique == "Tailor to a job description":
                if job_description:
                    prompt = f"{base_prompt} Compare the resume to the following job description and suggest specific changes to better tailor the resume to the role. Job Description:\n{job_description}"
                else:
                    st.warning("Please provide a job description for this critique type.")
                    st.stop()
            elif selected_critique == "Grammar and spelling check":
                prompt = f"{base_prompt} Conduct a thorough grammar and spelling check. List any errors found with a suggestion for correction."
            elif selected_critique == "Strengths and weaknesses":
                prompt = f"{base_prompt} Identify the key strengths and weaknesses of this resume."
            elif selected_critique == "Action verbs analysis":
                prompt = f"{base_prompt} Analyze the action verbs used in the resume. Suggest stronger, more impactful alternatives to common verbs."

            feedback = get_ai_feedback(prompt, resume_text)

            if feedback:
                st.subheader("AI Feedback")
                st.markdown(feedback)
            else:
                st.warning("Could not retrieve AI feedback. Please try again.")

elif submit_button and uploaded_file is None:
    st.error("Please upload a PDF file to get feedback.")
