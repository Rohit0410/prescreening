import streamlit as st
import google.generativeai as genai
from llama_index.core import SimpleDirectoryReader
import os
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')
import requests
# Configure API Key
api_key = "AIzaSyBtO-JpfCUZ8Zz_uLrjY1SavDvSUICvbqY"  # Replace with your actual API key
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-flash')

# Function to extract text from JD/Resume files
def process_document(uploaded_file, doc_type):
    """Extract text from uploaded JD or Resume file."""
    if uploaded_file is not None:
        try:
            temp_path = os.path.join("temp", uploaded_file.name)
            os.makedirs("temp", exist_ok=True)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            document = SimpleDirectoryReader(input_files=[temp_path]).load_data()
            return " ".join([i.text.replace("\n", " ") for i in document])

        except Exception as e:
            st.error(f"Error processing {doc_type}: {e}")
            return None

# Function to generate follow-up questions
def generate_followup_question(jd_text, user_response):
    """Generate a follow-up question based on the candidate's previous answer."""
    followup_prompt = f"""
    You are an AI interviewer conducting a job interview for the following JD:
    {jd_text}
    
    The candidate answered the previous question as follows:
    "{user_response}"
    
    Based on this response, ask a relevant follow-up question that probes deeper into their experience or skills.
    """
    return model.generate_content(followup_prompt).text

# Function to evaluate responses
def evaluate_responses(jd_text, responses):
    """Evaluate the candidate's responses and provide a final score out of 100 with feedback."""
    evaluation_prompt = f"""
    You are an AI recruiter evaluating a candidate for the following JD:
    {jd_text}
    
    The candidate's responses are as follows:
    {responses}
    
    Rate their overall performance out of 100 and provide feedback on strengths and weaknesses.
    """
    return model.generate_content(evaluation_prompt).text



def score_resume(jd_content,resume_content):
    """
    Sends a request to the JD-CV scoring API to evaluate resumes based on a given job description.
    
    Args:
        jd_url (str): URL of the job description document.
        jd_content (str): Raw text content of the job description.
        resumes (str): URL(s) of resumes to be scored.
        resume_content (str): Raw text content of the resume.

    Returns:
        dict: JSON response from the API containing the scoring results.
    """
    url = "http://api.dev.linkcxo.com/v1/jdcv/score_resumes"

    # Ensure at least one JD input is provided
    if not jd_content:
        return {"error": "Either 'jd_file' or 'jd_content' must be provided."}

    # Ensure at least one resume input is provided
    if not resume_content:
        return {"error": "Either 'resumes' or 'resume_content' must be provided."}
    
    payload = {
        'jd_file': "",
        'jd_content': jd_content,
        'resumes': "",
        'resume_content': resume_content
    }
    
    try:
        response = requests.post(url, data=payload)
        response.raise_for_status()  # Raise error if request fails
        print(response)
        return response.json()['resume_1.pdf']
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return None

# Streamlit UI
st.title("💼 AI Pre-Screening Chatbot")

jd_file = st.file_uploader("Upload Job Description (JD)", type=["docx", "txt", "pdf"])
resume_file = st.file_uploader("Upload Resume", type=["docx", "txt", "pdf"])

if jd_file and resume_file:
    with st.spinner("Processing JD & Resume..."):
        jd_details = process_document(jd_file, "JD")
        resume_details = process_document(resume_file, "Resume")
    final_doc = jd_details+resume_details
    result = score_resume(jd_details,resume_details)

    if jd_details and resume_details:
        st.subheader(f"""Your profile is {result}% matching with the Job Description""")
        st.subheader("🤖 AI Pre-Screening Interview")

        # Initialize session state
        if "conversation_history" not in st.session_state:
            st.session_state.conversation_history = []
            st.session_state.question_count = 0
            st.session_state.current_question = model.generate_content(
                f"Based on this JD, ask the first interview question:\n{final_doc}"
            ).text

        # Display AI's current question
        if st.session_state.question_count < 5:
            st.write(f"**AI:** {st.session_state.current_question}")

            # Capture user response
            user_response = st.text_area("Your Answer:", key="user_response")

            if st.button("Submit Response"):
                if user_response.strip():
                    # Store response
                    st.session_state.conversation_history.append((st.session_state.current_question, user_response))
                    st.session_state.question_count += 1

                    # Generate next question if under 5
                    if st.session_state.question_count < 5:
                        st.session_state.current_question = generate_followup_question(final_doc, user_response)
                        st.rerun()

        else:
            # After 5 questions, evaluate responses
            st.subheader("📊 Final Evaluation")
            responses = "\n".join([f"Q: {q}\nA: {a}" for q, a in st.session_state.conversation_history])
            score_feedback = evaluate_responses(jd_details, responses)

            st.write(score_feedback)

            # Clear session state for a new interview
            if st.button("Start New Interview"):
                for key in ["conversation_history", "current_question", "question_count"]:
                    del st.session_state[key]
                st.rerun()
