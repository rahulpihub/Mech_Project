import streamlit as st
import sqlite3
import numpy as np
import cv2
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from transformers import AutoTokenizer
import google.generativeai as genai
import os
import io
import tempfile
import uuid  

# Set the API key for Google Generative AI
IMAGE_API_KEY = 'AIzaSyDdRNV7YuLzJ8n-nYkIzmGMZPR8l8wVlks'

# Function to load an image from a file
def load_image(image_file):
    img = Image.open(image_file)  
    return img

# Function to analyze the image and detect components using the AI model
def analyze_image(image, api_key, data):
    try:
        # Prepare the prompt for the AI model with the list of components to detect
        data_str = ", ".join(data)
        prompt = f"""
        you are now a expert in mechanical image objects component detection,
        You need to access the following list of components: {data_str}.
        Detect whether the image contains any of these components.
        List each component found in the image on a new line.
        Do not include descriptions or any extra information, only the component names.
        The names should be presented as a list, each on a separate line.
        """
        
        # Configure the Generative AI with the provided API key
        genai.configure(api_key=api_key)
        image_model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        
        # Generate content (component detection) using the model
        response = image_model.generate_content([image, prompt])
        generated_text = response.text
        
        # Process the generated text to extract component names
        components = [line.strip() for line in generated_text.strip().splitlines() if line.strip()]
        return components
    except Exception as e:
        # Handle any errors that occur during analysis
        st.error(f"An error occurred: {str(e)}")
        return []

# Configure Google Generative AI model
genai.configure(api_key=IMAGE_API_KEY)

# Create a Chat model for Google Generative AI
text_model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=IMAGE_API_KEY,
    temperature=0.2,
    convert_system_message_to_human=True
)

# Connect to the SQLite database
conn = sqlite3.connect('diseal.db')
cursor = conn.cursor()

# Set the title of the Streamlit app
st.title("Mechanical Vision AI 🤖 ")

# Create a file uploader for images
uploaded_image = st.file_uploader("Upload an image to search", type=["jpg", "png"])

if uploaded_image is not None:
    # Load and display the uploaded image
    image = load_image(uploaded_image)
    input_image = np.array(Image.open(uploaded_image))
    st.image(input_image, caption="Uploaded Image", channels="RGB")

    # Generate unique IDs for storing data
    unique_id = str(uuid.uuid4())  
    persist_directory = f"./chroma_db/{unique_id}"  
    os.makedirs(persist_directory, exist_ok=True)  # Create directory if it doesn't exist

    unique_id_sep = str(uuid.uuid4())
    persist_directory_sep = f"./chroma_separatekb_db/{unique_id_sep}"
    os.makedirs(persist_directory_sep, exist_ok=True)  # Create separate directory

    # Query the database to retrieve images and PDFs
    cursor.execute("SELECT image, pdf, common_pdf FROM knowledge_base")
    rows = cursor.fetchall()

    found_pdf = None
    found_common_pdf = None
    found_image = None
    similarity_threshold = 0.9  # Set similarity threshold for image comparison

    # Loop through each row in the database to find similar images
    for row in rows:
        db_image_blob = row[0]
        db_pdf_blob = row[1]
        db_common_pdf_blob = row[2]

        nparr = np.frombuffer(db_image_blob, np.uint8)
        db_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # Decode the image from blob

        # Convert both images to grayscale for similarity comparison
        image1 = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        image2 = cv2.cvtColor(db_image, cv2.COLOR_BGR2GRAY)

        # Calculate the structural similarity index (SSIM)
        score, _ = ssim(image1, image2, full=True)

        # If the score exceeds the threshold, a similar image is found
        if score > similarity_threshold:
            st.success("Similar image found!")
            print(score) # Log the score
            found_pdf = db_pdf_blob
            found_common_pdf = db_common_pdf_blob
            found_image = db_image
            break  # Exit the loop once a match is found


    # If a matching image and PDF are found, display them and provide download options
    if found_pdf and found_image is not None:
        st.image(found_image, channels="BGR", caption="Relevant Image")

        # Create in-memory buffers for the PDFs to be downloaded
        pdf_buffer = io.BytesIO(found_pdf)
        common_pdf_buffer = io.BytesIO(found_common_pdf)

        # Provide buttons for downloading the relevant and common PDFs
        st.download_button("Download Uploaded Image's Relevant Data", pdf_buffer, file_name="related_document.pdf", mime="application/pdf")
        st.download_button("Download Uploaded Image's Whole Data", common_pdf_buffer, file_name="common_document.pdf", mime="application/pdf")

        st.subheader("Detecting the components with our AI  🤖...")

        # Create temporary PDF files for analysis
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf_file_whole:
            temp_pdf_file_whole.write(found_common_pdf)
            temp_pdf_path_whole = temp_pdf_file_whole.name

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf_file_separate:
            temp_pdf_file_separate.write(found_pdf)
            temp_pdf_path_separate = temp_pdf_file_separate.name

        # Load and split the PDFs for processing
        pdf_loader = PyPDFLoader(temp_pdf_path_whole)
        pagesall = pdf_loader.load_and_split()

        pdf_loader = PyPDFLoader(temp_pdf_path_separate)
        pagessep = pdf_loader.load_and_split()

        # Print the content of all pages in the PDFs (for debugging)
        for i, page in enumerate(pagesall):
            print(f"Page {i + 1} Content:\n{page.page_content}\n")

        for i, page in enumerate(pagessep):
            print(f"Page {i + 1} Content:\n{page.page_content}\n")

        # Split the content of the PDF pages into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        context = "\n\n".join(str(p.page_content) for p in pagesall)
        texts = text_splitter.split_text(context)

        text_splitter_sep = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        context_sep = "\n\n".join(str(p.page_content) for p in pagessep)
        texts_sep = text_splitter_sep.split_text(context_sep)

        # Determine the number of text chunks created
        num_texts = len(texts)
        k_value = min(5, num_texts)  # Limit to a maximum of 5 chunks

        num_texts_sep = len(texts_sep)
        k_value_sep = min(5, num_texts_sep)  # Same for separate documents

        # Create embeddings for the text using Hugging Face model
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", clean_up_tokenization_spaces=True)

        # Create Chroma vector stores for the texts
        vector_index = Chroma.from_texts(texts, embeddings, persist_directory=persist_directory).as_retriever(search_kwargs={"k": k_value})
        vector_index_sep = Chroma.from_texts(texts_sep, embeddings, persist_directory=persist_directory_sep).as_retriever(search_kwargs={"k": k_value_sep})

        # Create RetrievalQA chains for querying the AI model
        qa_chain = RetrievalQA.from_chain_type(
            text_model,
            retriever=vector_index,
            return_source_documents=True
        )

        qa_chain_sep = RetrievalQA.from_chain_type(
            text_model,
            retriever=vector_index_sep,
            return_source_documents=True
        )

        # Define prompts for the AI model to get component names and object title
        prompt = "List all the mechanical components available in the engine along with their names and I don't want any description. Then in each and every string you don't want to mention integers."
        result = qa_chain.invoke({"query": prompt})
        components_list = result["result"].split('\n')  # Split the result into a list

        prompt_title = "What is the full name of this mechanical object? "
        result_title = qa_chain.invoke({"query": prompt_title})
        title_list = result_title["result"].split('\n')  # Get the title of the mechanical object

        prompt_sep = "List all the mechanical components available in the engine along with their names and I don't want any description. Then in each and every string you don't want to mention integers."
        result_sep = qa_chain_sep.invoke({"query": prompt_sep})
        components_list_sep = result_sep["result"].split('\n')  # Separate components list

        prompt_title_sep = "What is the full name of this mechanical object? "
        result_title_sep = qa_chain_sep.invoke({"query": prompt_title_sep})
        title_list_sep = result_title_sep["result"].split('\n')  # Title for the separate components

        # Display the results in the Streamlit app
        st.write("**Mechanical Object Name**")
        for title in title_list:
            st.write(title)

        st.write("**Mechanical All Components List**")
        for component in components_list:
            st.write(component)

        st.subheader("Detecting the mechanical component at this image perspective...")

        st.write("**Detected Components List**")
        for component in components_list_sep:
            st.write(component)

        # Analyze the uploaded image to detect components
        detected_components = analyze_image(image, IMAGE_API_KEY, components_list_sep)

        # Filter out empty components from the list
        filtered_components_list_sep = [component.strip() for component in components_list_sep if component.strip()]
        print(filtered_components_list_sep)
        print("\n")
        
        # Filter detected components
        filtered_detected_components = [component for component in detected_components if component]
        print(filtered_detected_components)
        print("\n")

        # Identify defects by comparing detected components with expected components
        defect = [component for component in filtered_components_list_sep if component not in filtered_detected_components]
        print(defect)

        # Display detected components
        st.write("Detected Components Of The Image:")
        for detect in detected_components:
            st.write(f"- {detect}")

        # Assuming 'defect' is populated with detected defect components
        st.write("Defected Components Of The Image:")

        if defect:  # Check if there are any defects
            for d in defect:
        # Create an Amazon search URL for the defect component
                amazon_search_url = f"https://www.amazon.in/s?k={d.replace(' ', '+')}&ref=nb_sb_noss"
        
        # Display defect component with a clickable link to Amazon search results
                st.markdown(f"- {d} [🔗 First Product on Amazon]({amazon_search_url})")
        else:
            st.write("No defect components detected.")


        # Clean up temporary PDF files
        os.remove(temp_pdf_path_whole)
        os.remove(temp_pdf_path_separate)
    else:
        # Inform the user if no similar image was found
        st.error("No similar image found.")
