# Quick Run

Use this page when you return to the project on another day. For the normal daily startup, you only need to activate the virtual environment and run the app.

## Daily Run

```powershell
.\.venv\Scripts\activate
python app.py
```

If the virtual environment is already active, you only need:

```powershell
python app.py
```

## If You Changed PDFs

If you add, remove, or replace files in `Data/`, rebuild the index first, then start the app:

```powershell
python store_index.py
python app.py
```

## Notes

- Keep your `.env` file in the project root.
- Keep `USE_GROQ_CHAT=true` if you want Groq to be used first.
- Keep `USE_OPENAI_CHAT=true` if you want OpenAI as the second fallback.
- If you want a document-only chatbot, set both chat flags to `false`.
