import os
import json
from flask import Flask, request, render_template, jsonify, send_file
from gtts import gTTS
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
import PyPDF2  # For PDF text extraction

app = Flask(__name__)

# Load the GROQ API key from the config.json file
try:
    with open('config.json', 'r') as file:
        config = json.load(file)
    os.environ["GROQ_API_KEY"] = config.get("GROQ_API_KEY", "")
    if not os.environ["GROQ_API_KEY"]:
        raise ValueError("GROQ_API_KEY is missing in config.json.")
except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
    raise Exception(f"Error loading config.json: {e}")

# Mapping of topics to corresponding PDF files
topic_to_pdf = {
    "half adder": "module half.pdf",
    "full adder": "module FullAdder.pdf",
    # Add more topics and corresponding PDFs as needed
}

# Function to extract text from a PDF file using PyPDF2
def extract_text_from_pdf(file_path):
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
            return text
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {e}")

# Prompt template for VLSI-related queries
system_prompt = PromptTemplate(
    input_variables=["context", "query"],
    template=(
        "You are a highly knowledgeable and articulate VLSI (Very Large Scale Integration) expert with vast expertise in "
        "digital and analog IC design, fabrication, and testing. Your goal is to provide concise, accurate answers within "
        "200 words based on the given context. Engage the user in a professional and interactive manner, ensuring clarity "
        "in explanations.\n\nContext: {context}\n\nQuery: {query}\nResponse:"
    )
)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    user_query = request.json.get('query')
    topic = request.json.get('topic')

    if not user_query or not topic:
        return jsonify({"error": "Query or topic parameter is missing"}), 400

    # Retrieve the corresponding PDF file for the topic
    pdf_filename = topic_to_pdf.get(topic)
    if not pdf_filename:
        return jsonify({"error": "Topic not found"}), 404

    pdf_path = os.path.join("pdf", pdf_filename)

    try:
        # Extract text from the PDF file
        text = extract_text_from_pdf(pdf_path)

        # Split the text into manageable chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_text(text)
        documents = [Document(page_content=t) for t in texts]

        # Create embeddings and vector store
        embeddings = HuggingFaceEmbeddings()
        persist_directory = "doc_db"
        vectordb = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_directory
        )

        # Set up the retriever
        retriever = vectordb.as_retriever()

        # Initialize the LLM from GROQ
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

        # Create a QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )

        # Prepare the context and format the prompt
        document_content = "\n".join([doc.page_content for doc in documents])
        updated_context = f"Document Content:\n{document_content}"
        vlsi_prompt = system_prompt.format(context=updated_context, query=user_query)

        # Generate the response
        response = qa_chain.invoke({"query": vlsi_prompt})
        result_text = response["result"]
        concise_response = " ".join(result_text.split()[:200])

        # Save the TTS audio file
        audio_file = "static/response.mp3"
        if os.path.exists(audio_file):
            os.remove(audio_file)
        tts = gTTS(text=concise_response, lang='en')
        tts.save(audio_file)

        return jsonify({"response": concise_response, "audio_url": f"/static/response.mp3"})

    except Exception as e:
        app.logger.error(f"Error while processing query: {e}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/static/<path:filename>')
def serve_audio(filename):
    return send_file(f"static/{filename}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
