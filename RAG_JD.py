import os
import google.generativeai as genai
# import docx2txt
import nltk 
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words('english'))
import re
import random
import shutil
from llama_index.core import SimpleDirectoryReader
api_list = "AIzaSyBtO-JpfCUZ8Zz_uLrjY1SavDvSUICvbqY"
# api_key=random.choices(api_list)
genai.configure(api_key=api_list)
import requests
import logging
import numpy as np
import pandas as pd
import json
from flask import jsonify,Flask,request
import traceback
from sentence_transformers import SentenceTransformer
import faiss
embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

app = Flask(__name__)
def preprocessing(document):
    text = document.replace('\n', ' ').replace('\t', ' ').lower()
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    tokens = word_tokenize(text)
    tokens = [re.sub(r'[^a-zA-Z\s]', '', token) for token in tokens]
    tokens = [token for token in tokens if token and token not in stop_words]
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

def get_gemini_response(prompt):
    # input_text = preprocessing(prompt) 
    model=genai.GenerativeModel('gemini-1.5-flash')
    response=model.generate_content([prompt],generation_config = genai.GenerationConfig(
        temperature=0.3
    ))
    return response.text

def download_file(url):
    try:
        # Send a GET request to download the file
        resume_folder_path1 = "downloaded_file.pdf"
        resume_folder_path2 = "downloaded_file.docx"
        g=url.split("?")
        file_extension = os.path.splitext(g[0])[1]
        print(file_extension)
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check if the request was successful

        # Save the file content to the specified path
        if file_extension=='.docx':
            with open(resume_folder_path2, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)

            print(f"File downloaded successfully: {resume_folder_path2}")
            return resume_folder_path2
        elif file_extension==".pdf":
            with open(resume_folder_path1, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)

            print(f"File downloaded successfully: {resume_folder_path1}")
            return resume_folder_path1
        print('Done')

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        print('failed')
        return None

def input_pdf_setup(uploaded_file):
    if uploaded_file is not None:
        data = SimpleDirectoryReader(input_files=[uploaded_file]).load_data()
        data1 = " ".join([doc.text for doc in data])
        print("DATA",data1)
        document_resume = " ".join([doc.text.replace('\n', ' ').replace('\t', ' ').replace('\xa0', ' ') for doc in data])
        # final = preprocessing(document_resume)
        return data1,document_resume
    else:
        raise FileNotFoundError("No file uploaded")

# Step 1: Semantic Chunking
def semantic_chunking(document, max_tokens=100, overlap=20):
    sentences = document.split('. ')
    chunks = []
    current_chunk = []
    current_tokens = 0

    for i, sentence in enumerate(sentences):
        sentence_tokens = len(sentence.split())
        if current_tokens + sentence_tokens > max_tokens:
            chunks.append(". ".join(current_chunk))
            current_chunk = sentences[max(0, i - overlap):i]
            current_tokens = sum(len(s.split()) for s in current_chunk)
        
        current_chunk.append(sentence)
        current_tokens += sentence_tokens

    if current_chunk:
        chunks.append(". ".join(current_chunk))
    return chunks

# Step 2: Indexing with FAISS
def build_faiss_index(chunks):
    embeddings = embedding_model.encode(chunks, convert_to_tensor=False)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, embeddings

# Step 3: Querying the Index
def retrieve_chunks(query, index, chunks, top_k=3):
    query_embedding = embedding_model.encode([query], convert_to_tensor=False)
    distances, indices = index.search(query_embedding, top_k)
    retrieved_chunks = [chunks[idx] for idx in indices[0]]
    return retrieved_chunks

# Step 4: Generating a Response
def generate_response(query, retrieved_chunks):
    context = "\n\n".join(retrieved_chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    response = get_gemini_response(prompt)
    return response

# @app.route('/v1/jd_autofilling', methods=['POST'])
def scoring1(uploaded_file):
#     try:
#         if 'jd_file' not in request.form:
#             return jsonify({'error': 'Please provide a JD file.'}), 400

#         jd = request.form['jd_file']
#         print('resume',jd)

#         # resume_folder_path = r'temp5/'
        
#         # resume_folder_path = r"downloaded_file.pdf"
#         # os.makedirs(resume_folder_path, exist_ok=True)

#         resume_file_path = download_file(jd)
#         print('jd_file_path',resume_file_path)

#         data1,input_text = input_pdf_setup(resume_file_path)
#         print(input_text)
#         if input_text is None:
#             return jsonify({'error': 'Error processing the JD file.'}), 500

    input_text = input_pdf_setup(uploaded_file)
    # Step 1: Semantic Chunking
    print(input_text[0])
    print("Step 1: Semantic Chunking...")
    chunks = semantic_chunking(input_text[0], max_tokens=50, overlap=5)
    # for i, chunk in enumerate(chunks):
    #     print(f"Chunk {i + 1}:\n{chunk}\n")

    # Step 2: Build FAISS Index
    print("Step 2: Building FAISS Index...")
    index, embeddings = build_faiss_index(chunks)

    # Step 3: Retrieve Relevant Chunks
    print("Step 3: Retrieving Relevant Chunks...")
    skills = "what are the skills mentioned in this documents?"
    Experience = "recreate all the working experiences"

    retrieved_chunks_skills = retrieve_chunks(skills, index, chunks)
    retrieved_chunks_Experience = retrieve_chunks(Experience, index, chunks)
    # retrieved_chunks_sal_Currancy = retrieve_chunks(query_sal_currancy, index, chunks)
    print("Retrieved Chunks:")
    # for i, chunk in enumerate(retrieved_chunks):
    #     print(f"Chunk {i + 1}:\n{chunk}\n")

    # Step 4: Generate Response
    print("Step 4: Generating Response...")
    try:
        answer_skill = generate_response(skills, retrieved_chunks_skills)
        # answer_skill = str(re.search(r"\d+", answer_skill).group())
    except Exception as e:
        answer_skill=""

    try:
        answer_Experience = generate_response(Experience, retrieved_chunks_Experience)
        # answer_Experience = str(re.search(r"\d+", answer_Experience).group())
    except Exception as e:
        answer_Experience=""



    print("Generated Answer1:")
    print(answer_skill)
    print("Generated Answer2:")
    print(answer_Experience)
 
    print(answer_skill,answer_Experience)
    return answer_skill,answer_Experience
#     except Exception as e:
#         return logging.error(f"An error occurred: {e}. Line: {traceback.format_exc()}")
        
scoring1(r"D:\Rohit\jdcv_score_app\jdcv_score_app\resume_Details\downloaded_file.pdf")

