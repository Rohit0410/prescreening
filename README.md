"""
AI Pre-Screening Chatbot using Streamlit and Gemini 1.5 Flash

This application allows recruiters to conduct AI-driven pre-screening interviews
by analyzing job descriptions (JD) and resumes. It generates interview questions,
evaluates candidate responses, and provides feedback.

Modules:
    - streamlit (st): For creating the web interface.
    - google.generativeai (genai): For generating interview questions and evaluations.
    - llama_index.core (SimpleDirectoryReader): For extracting text from uploaded documents.
    - os: For file handling.
    - nltk: For natural language processing (downloads stopwords, tokenizers, etc.).

Usage:
    1. Upload a Job Description (JD) document.
    2. Upload a Resume document.
    3. The AI will conduct a five-question interview.
    4. User responds to each question in the text area.
    5. AI evaluates responses and provides a final score with feedback.

Functions:
    - process_document(uploaded_file, doc_type)
        Extracts text from an uploaded document (JD or Resume).

    - generate_followup_question(jd_text, user_response)
        Generates a follow-up interview question based on the user's response.

    - evaluate_responses(jd_text, responses)
        Evaluates the candidate's responses, provides a score, and gives feedback.

Session State Variables:
    - st.session_state.conversation_history: Stores the history of questions and answers.
    - st.session_state.question_count: Tracks the number of questions asked.
    - st.session_state.current_question: Holds the current interview question.

Output:
    - AI-generated interview questions
    - User responses input via text area
    - AI-generated evaluation score and feedback

Author: Rohit Chauhan
Version: 1.0
"""

