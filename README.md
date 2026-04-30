# End-to-End Medical Chatbot

A Flask-based medical chatbot that answers questions from PDFs stored in the `Data/` folder. The project uses Pinecone for retrieval and returns answers grounded in the uploaded documents by default.

## What This Project Does

- Reads medical PDFs from `Data/`
- Splits text into chunks and builds a vector index in Pinecone
- Answers user questions using the most relevant document snippets
- Optionally uses Groq chat first, OpenAI second, and a document snippet fallback last

## Current Behavior

- Default mode is document-grounded only
- The bot does not need any online chat API key to run the local retrieval flow
- Groq chat is optional and enabled by default when `GROQ_API_KEY` is present
- If a question is not covered by the uploaded PDFs, the bot says it could not find the answer in the documents

## Requirements

- Python 3.10 or newer
- A Pinecone account and API key
- PDF files placed in the `Data/` folder

Optional:

- Groq API key, only if you want to enable online chat fallback
- OpenAI API key, only if you want a second online fallback

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

# Optional, only needed for Pinecone serverless index creation
# PINECONE_ENVIRONMENT=us-east1-gcp
# PINECONE_CLOUD=aws

# Optional, only if you want Groq chat fallback
# GROQ_API_KEY=your_groq_key
# USE_GROQ_CHAT=true
# GROQ_MODEL_NAME=llama-3.1-8b-instant

# Optional, second online fallback
# OPENAI_API_KEY=your_openai_key
# USE_OPENAI_CHAT=true
# OPENAI_MODEL_NAME=gpt-4o-mini
```

The code also accepts `PINECONE_KEY`, `GROQ_KEY`, and `OPENAI_KEY` as aliases.

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

- http://127.0.0.1:8080

## How Answers Work

1. Your question is converted into a deterministic local embedding.
2. The app first searches the extracted PDF text directly for the best matching document chunk.
3. If needed, Pinecone returns the closest document chunks.
4. The app extracts the best matching snippet from those chunks.
5. If `USE_GROQ_CHAT=true`, the app tries Groq first, then OpenAI if Groq fails, and finally falls back to the document snippet.

## Troubleshooting

- If `store_index.py` says no PDFs were found, make sure `Data/` contains one or more `.pdf` files.
- If Pinecone cannot connect, verify `PINECONE_API_KEY` and, for serverless index creation, `PINECONE_ENVIRONMENT`.
- If the app returns a document snippet that looks unrelated, add better source PDFs or rerun `python store_index.py` after updating the `Data/` folder.
- If you want the app to stay strictly in-document, keep both `USE_GROQ_CHAT=false` and `USE_OPENAI_CHAT=false`.

## Project Structure

- `app.py` - Flask app and question answering route
- `store_index.py` - Builds the Pinecone index from the PDFs
- `src/helper.py` - PDF loading, chunking, and embedding helpers
- `src/prompt.py` - Optional prompt used only when online chat fallback is enabled
- `templates/chat.html` - Frontend chat page
- `static/style.css` - Frontend styling

## Notes

- This repository is now configured for a simple, fast start in VS Code.
- The main behavior is retrieval from your uploaded medical documents, not general web search.
- If you want, I can also add a short `smoke-test` script to verify `.env`, `Data/`, and Pinecone connectivity before running the app.
