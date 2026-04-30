<img width="941" height="472" alt="image" src="https://github.com/user-attachments/assets/4ea2a86d-5238-4fd9-80ef-9b278acaef5f" />


# End-to-End Medical Chatbot

A Flask-based medical chatbot that answers questions from PDFs stored in the `Data/` folder. The project uses Pinecone for retrieval and returns answers grounded in the uploaded documents by default.

## What This Project Does

- Reads medical PDFs from `Data/`
- Splits text into chunks and builds a vector index in Pinecone
- Answers user questions using the most relevant document snippets
- Uses Groq chat when enabled, then falls back to a safe document-grounded response

## Current Behavior

- Default mode is document-grounded only
- The app only reads these five environment variables
- Groq chat is enabled when `USE_GROQ_CHAT=true` and `GROQ_API_KEY` is present
- If a question is not covered by the uploaded PDFs, the bot says it could not find the answer in the documents

## Requirements

- Python 3.10 or newer
- A Pinecone account and API key
- PDF files placed in the `Data/` folder

Optional:

- The Groq settings listed below if you want the live chat fallback

## Setup

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Create a `.env` file in the project root:

```ini
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=medicalbot

# Required for live chat fallback
# GROQ_API_KEY=your_groq_key
# USE_GROQ_CHAT=true
# GROQ_MODEL_NAME=llama-3.1-8b-instant
```

The app is now configured for only these 5 environment variables:

- `PINECONE_API_KEY`
- `PINECONE_INDEX_NAME`
- `GROQ_API_KEY`
- `USE_GROQ_CHAT`
- `GROQ_MODEL_NAME`

## Build the Index

1. Put your PDF files into the `Data/` folder.
2. Run the index builder:

```powershell
python store_index.py
```

This reads the PDFs, chunks the text, creates deterministic local embeddings, and uploads the vectors to Pinecone.

## Run the App

```powershell
python app.py
```

Open:

- http://127.0.0.1:5000

## How Answers Work

1. Your question is converted into a deterministic local embedding.
2. The app first searches the extracted PDF text directly for the best matching document chunk.
3. If needed, Pinecone returns the closest document chunks.
4. The app extracts the best matching snippet from those chunks.
5. If `USE_GROQ_CHAT=true`, the app tries Groq first and falls back to a safe document-guided response if the model is unavailable.

## Troubleshooting

- If `store_index.py` says no PDFs were found, make sure `Data/` contains one or more `.pdf` files.
- If Pinecone cannot connect, verify `PINECONE_API_KEY` and that your Pinecone index exists.
- If the app returns a document snippet that looks unrelated, add better source PDFs or rerun `python store_index.py` after updating the `Data/` folder.
- If you want the app to stay strictly in-document, set `USE_GROQ_CHAT=false`.

## Project Structure

- `app.py` - Flask app and question answering route
- `store_index.py` - Builds the Pinecone index from the PDFs
- `src/helper.py` - PDF loading, chunking, and embedding helpers
- `src/prompt.py` - Prompt used when Groq chat is enabled
- `templates/chat.html` - Frontend chat page
- `static/style.css` - Frontend styling

## Notes

- This repository is now configured for a simple, fast start in VS Code.
- The main behavior is retrieval from your uploaded medical documents, not general web search.
- If you want, I can also add a short `smoke-test` script to verify `.env`, `Data/`, and Pinecone connectivity before running the app.
