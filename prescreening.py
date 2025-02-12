import streamlit as st
import google.generativeai as genai
from llama_index.core import SimpleDirectoryReader
import os
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')
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

# Streamlit UI
st.title("💼 AI Pre-Screening Chatbot")

jd_file = st.file_uploader("Upload Job Description (JD)", type=["docx", "txt", "pdf"])
resume_file = st.file_uploader("Upload Resume", type=["docx", "txt", "pdf"])

if jd_file and resume_file:
    with st.spinner("Processing JD & Resume..."):
        jd_details = process_document(jd_file, "JD")
        resume_details = process_document(resume_file, "Resume")

    if jd_details and resume_details:
        st.subheader("🤖 AI Pre-Screening Interview")

        # Initialize session state
        if "conversation_history" not in st.session_state:
            st.session_state.conversation_history = []
            st.session_state.question_count = 0
            st.session_state.current_question = model.generate_content(
                f"Based on this JD, ask the first interview question:\n{jd_details}"
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
                        st.session_state.current_question = generate_followup_question(jd_details, user_response)
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
