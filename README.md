# SmartPDFs 📚
( By Aman Jain :) )  
**SmartPDFs** is an AI-powered PDF assistant that allows users to **upload PDFs, summarize content, and interactively ask questions** using advanced language models (Llama 3 / HuggingFace models). It stores embeddings for efficient retrieval and keeps conversational context.

---

## Features and working

- Upload multiple PDFs and extract text automatically.
- Generate concise summaries of PDF content.
- Ask questions about the documents with context-aware responses.
- Maintains conversation history for multi-turn interactions.
- Sidebar displays PDF summary for quick reference.
- Built with **Streamlit** for a clean, interactive interface.

---

## User Interface

![SmartPDFs Screenshot](assets/SmartPDFs_with_llama.JPG)

---

## Tech Stack

- **Frontend:** Streamlit  
- **PDF Processing:** PyPDF2  
- **Text Embeddings:** HuggingFaceInstructEmbeddings  
- **Vector Store:** FAISS  
- **Language Models:** Llama 3 via Ollama, ChatOllama  
- **Memory:** LangChain ConversationBufferMemory  
- **Utilities:** LangChain, Python 3.9+

---

# Note: 
OpenAi Chat model can also be used instead of lllama3/Hugginh Face but it is paid per tokerns!
Also, OpenAi Embeddings can be used (must faster) but again paid.

---

## Architecture of the Project:

![SmartPDFs Screenshot](assets/Architecture.JPG)

---

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/SmartPDFs.git
cd SmartPDFs
```
2. Create and activate a virtual environment:
python -m venv venv
 - Windows: 
 ```bash
   .\venv\Scripts\activate
   ```
- macOS/Linux: 
```bash
   source venv/bin/activate
   ```

3. Install dependencies:
```bash
   pip install -r requirements.txt
```
4. Pull the Llama 3 model (using Ollama):
- Download and install: [https://ollama.com/download](https://ollama.com/download) and then:
```bash
    ollama pull llama3
```
5. Create a .env file for environment variables (if required by LangChain models):
 - OPENAI_API_KEY=your_api_key_here, ...

---

## Usage: 
```bash
streamlit run app.py
```
- Upload PDFs in the sidebar.
- Click Process to extract text, generate embeddings, and summarize content.
- Ask questions in the main interface.
- PDF summary is always accessible in the sidebar.


## Project Structure
SmartPDFs/  
├── app.py            # Main Streamlit app  
├── Templates.py      # HTML/CSS templates for UI  
├── requirements.txt  # Python dependencies  
├── README.md  
└── venv/             # Python virtual environment