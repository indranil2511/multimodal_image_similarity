import base64
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import asyncio
from embedding import color_histogram, get_image_embedding
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from io import BytesIO
import copy
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

from dotenv import load_dotenv

from prompt import get_prompt

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def ensure_event_loop():
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:  # No running event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop

# Use this function to run your async functions
def run_async_function():
    loop = ensure_event_loop()


def get_conversional_chain():
    """
    Creates a conversational chain for question answering.

    Returns:
        A LangChain question-answering chain.
    """

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. If the answer is not available in the context, just say, "answer is not available in the context", don't provide the wrong answer.

    Context:
    {context}?

    Question:
    {question}

    Answer:
    """
    print('chain initiated')
    model = ChatGoogleGenerativeAI(model=os.getenv("MODEL"))
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    print('chain executed')
    return chain


def user_input(user_question, image_list):
    model = ChatGoogleGenerativeAI(model="gemini-pro-vision",  max_output_tokens=1024)
    """
    Processes user input and generates a response using the conversational chain,
    providing both the user's question and the processed PDF text as context.

    Args:
        user_question: The user's question.
        processed_pdf_text: The processed text extracted from the uploaded PDF files.

    Returns:
        The generated response from the conversational chain.
    """
    content = []
    for img in image_list:
        obj = {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img}"},
        }
        deep_obj = copy.deepcopy(obj)
        content.append(deep_obj)
    prompt = get_prompt(user_question)
    text_obj = {"type": "text", "text": prompt}
    content.append(text_obj)
    msg = HumanMessage(
            content=content
        )
    output_message = model.invoke([msg])
    print(f'output_message ',output_message.content)
    # Load and preprocess images
    embeddings = [get_image_embedding(img) for img in st.session_state["image_byte"]]

    similarity_matrix = cosine_similarity(embeddings)
    # Print similarity score
    st.write(f'Similarity score between image1 and image2: {similarity_matrix[0][1]}')

    color_hist1 = color_histogram(st.session_state["image_byte"][0])
    color_hist2 = color_histogram(st.session_state["image_byte"][1])
    color_similarity = cosine_similarity([color_hist1], [color_hist2])

    st.write(f'Color similarity score between image1 and image2: {color_similarity[0][0]}')

    st.write("Reply: ", output_message.content)

    st.image(st.session_state["image_byte"][0])
    st.image(st.session_state["image_byte"][1])




def main():
    """
    Main function for the Streamlit app.
    """

    st.header("Chat with Alex")

    user_question = st.text_input("Ask a Question from the PDF Files")

    # Run the async function
    run_async_function()

    if user_question:
        if st.session_state.get("iamge_list"):
            with st.spinner('Processing...'):
                user_input(user_question, st.session_state["iamge_list"])
        else:
            st.error("Please upload files first.")

    with st.sidebar:
        st.title("Menu:")
        image_list = []
        image_byte = []
        images = st.file_uploader("Upload Files & Click Submit to Proceed", accept_multiple_files=True)
        for img in images:
            pics = copy.deepcopy(img)
            image_list.append(get_base64_image(pics.read()))
            image_byte.append(Image.open(BytesIO(img.read())))
        st.session_state["iamge_list"] = image_list
        st.session_state["image_byte"] = image_byte

def get_base64_image(image_byte):
    b64_string = base64.b64encode(image_byte).decode('utf-8')
    return b64_string
