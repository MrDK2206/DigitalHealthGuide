import os
import re
import logging
from difflib import get_close_matches
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, url_for
from openai import OpenAI
from pinecone import Pinecone

from src.helper import embed_texts, load_pdf_file, text_split


load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.logger.setLevel(logging.INFO)

SITE_NAME = "DigiHealthGuide"
LOGO_URL = "https://i.ibb.co/4RJcK36H/Untitled-design.png"
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "Data"

_GREETING_WORDS = {"hi", "hii", "hey", "hello", "hullo", "yo"}
_QUERY_EXPANSIONS = {
    "fever": {"temperature", "febrile", "pyrexia", "chills", "hot", "flu"},
    "remedy": {"treatment", "treat", "management", "relief", "care", "medicine", "medication"},
    "remedies": {"treatment", "treat", "management", "relief", "care", "medicine", "medication"},
    "headache": {"migraine", "head pain", "cephalalgia", "pain"},
    "pain": {"ache", "discomfort", "soreness", "painful"},
    "cough": {"sputum", "phlegm", "cold", "respiratory"},
}

_GENERAL_MEDICAL_GUIDANCE = {
    "fever": (
        "A fever can happen with infections, inflammation, or dehydration. Track the temperature, stay hydrated, and seek urgent care if there is confusion, trouble breathing, stiff neck, severe weakness, or a very high fever."
    ),
    "headache": (
        "Headaches are often related to stress, dehydration, lack of sleep, eye strain, or illness. Rest in a quiet room, drink water, and consider over-the-counter pain relief only if it is safe for you. Get urgent help for a sudden severe headache, headache with weakness, confusion, fever, stiff neck, vision changes, or after a head injury."
    ),
    "pain": (
        "Pain should be judged by location, severity, and duration. If pain is severe, worsening, associated with chest pressure, breathlessness, fainting, or neurological symptoms, get immediate medical help."
    ),
    "cough": (
        "A cough can come from a viral infection, allergies, irritation, or asthma. Watch for fever, shortness of breath, blood, chest pain, or symptoms lasting more than a few weeks."
    ),
}

_COMMON_SYMPTOM_GUIDANCE = {
    "stomach": "Stomach pain can be caused by indigestion, gas, infection, food intolerance, or acidity. Use light meals, hydrate well, and avoid oily or spicy food temporarily. Seek urgent care for severe persistent pain, vomiting blood, black stools, high fever, or dehydration.",
    "abdominal": "Abdominal pain can be due to gastric irritation, infection, constipation, or other internal causes. Rest, hydrate, and monitor progression. Get urgent evaluation if pain is severe, localized, worsening, or associated with fever, repeated vomiting, blood in stool, or fainting.",
    "throat": "A sore throat is often caused by viral infection or irritation. Warm fluids, rest, and hydration usually help. Seek care if symptoms last more than a few days, or if there is high fever, breathing difficulty, drooling, or severe swallowing pain.",
    "back": "Back pain commonly comes from strain, posture issues, or muscle spasm. Use gentle movement, heat packs, and avoid heavy lifting. Seek urgent care for weakness, numbness, bladder or bowel changes, fever, or pain after trauma.",
    "tooth": "Tooth pain can occur with cavity, gum infection, or sensitivity. Rinse with warm salt water and avoid very hot or cold foods. Get dental care soon, and urgent care if there is facial swelling, fever, or trouble opening the mouth.",
    "cold": "Common cold symptoms are usually self-limited. Rest, hydration, and symptom control can help. Seek care for breathing trouble, persistent high fever, chest pain, confusion, or symptoms that worsen instead of improving.",
    "stone": "Pain from kidney stone can be severe and may radiate from the flank to the lower abdomen or groin. Hydrate and seek urgent medical evaluation for severe pain, fever, vomiting, blood in urine, or reduced urine output.",
}

_GENERAL_MEDICAL_TERMS = {
    "remedy",
    "remedies",
    "treatment",
    "treat",
    "care",
    "medicine",
    "medication",
    "symptom",
    "symptoms",
    "pain",
    "fever",
    "cough",
    "headache",
    "solution",
    "solutions",
}

_NO_ANSWER_PATTERNS = (
    "could not find",
    "couldn't find",
    "not found in the uploaded documents",
    "not in the uploaded documents",
    "do not contain the answer",
    "don't contain the answer",
)

_SITE_NAV = [
    {"label": "Home", "endpoint": "home", "page": "home"},
    {"label": "About", "endpoint": "about", "page": "about"},
    {"label": "How It Works", "endpoint": "how_it_works", "page": "how_it_works"},
    {"label": "Capabilities", "endpoint": "capabilities", "page": "capabilities"},
    {"label": "Developers", "endpoint": "developers", "page": "developers"},
    {"label": "Chat", "endpoint": "chat_page", "page": "chat"},
]

_USER_STEPS = [
    {
        "title": "Ask a question",
        "text": "Type a medical question in plain language, the same way you would ask a helper or support assistant.",
    },
    {
        "title": "Get document-backed help",
        "text": "The answer is retrieved from the uploaded PDFs and returned in a friendly, concise format.",
    },
    {
        "title": "Know the limits",
        "text": "If the documents do not contain the answer, the bot says so instead of guessing.",
    },
]

_USER_BENEFITS = [
    "Simple chat interface with a medical theme.",
    "Answers grounded in the uploaded source content.",
    "Helpful fallback text when the source does not contain the answer.",
    "Easy to use on desktop and mobile.",
]

_USER_LIMITS = [
    "Not a replacement for a doctor or emergency care.",
    "Does not diagnose or prescribe treatment.",
    "Will not invent missing facts.",
]

_DEVELOPER_DETAILS = {
    "architecture": [
        "Flask serves the UI pages and the /get API route.",
        "PDFs from Data/ are loaded, read, and split into chunks.",
        "Chunk retrieval happens locally first, then Pinecone if available.",
        "Groq is the LLM for generating responses.",
    ],
    "files": [
        "app.py controls routing, retrieval, and response generation.",
        "src/helper.py handles PDF loading, chunking, embeddings, and IDs.",
        "src/prompt.py stores the chat system prompt for fallback generation.",
        "store_index.py builds or refreshes the Pinecone index.",
        "templates/ and static/ contain the presentation layer.",
    ],
    "runtime": [
        "The app runs on port 5000 locally by default.",
        "Render deployment uses gunicorn through the Procfile.",
        "The API is JSON-free on the frontend and returns plain text for chat replies.",
    ],
    "env": [
        "PINECONE_API_KEY",
        "PINECONE_INDEX_NAME",
        "GROQ_API_KEY",
        "USE_GROQ_CHAT (default: true)",
        "GROQ_MODEL_NAME (default: llama-3.1-8b-instant)",
    ],
}

_PROJECT_STEPS = [
    {
        "title": "Load the PDFs",
        "text": "The app reads the medical documents from the Data/ folder and prepares them for retrieval.",
    },
    {
        "title": "Find the best context",
        "text": "A user question is matched against the document chunks first so the answer stays grounded in source material.",
    },
    {
        "title": "Return a safe answer",
        "text": "If the documents do not contain the answer, the assistant falls back to careful guidance instead of guessing.",
    },
]

_CAN_DO = [
    "Answer questions using the uploaded PDF content.",
    "Show a clear, user-friendly medical chat interface.",
    "Provide fallback guidance when the source documents do not contain the answer.",
    "Run as a Flask website on desktop and mobile browsers.",
]

_CANNOT_DO = [
    "Replace a licensed clinician or emergency services.",
    "Guarantee a diagnosis or prescribe treatment.",
    "Invent facts that are not present in the source content.",
]


def _first_env_value(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def _env_bool(name: str, default: str = "true") -> bool:
    return os.getenv(name, default).lower() in {"1", "true", "yes", "on"}


def _site_context(active_page: str, page_title: str, page_description: str, **extra):
    return {
        "active_page": active_page,
        "nav_items": [
            {**item, "href": url_for(item["endpoint"])}
            for item in _SITE_NAV
        ],
        "site_name": SITE_NAME,
        "logo_url": LOGO_URL,
        "page_title": page_title,
        "page_description": page_description,
        "backend_status": _backend_status(),
        **extra,
    }


def _is_greeting(query: str) -> bool:
    terms = {term for term in re.findall(r"[a-z0-9]+", query.lower()) if term}
    if not terms:
        return False

    if terms.issubset(_GREETING_WORDS):
        return True

    non_greeting_terms = terms.difference(_GREETING_WORDS)
    if terms.intersection(_GREETING_WORDS) and len(non_greeting_terms) <= 2:
        return not non_greeting_terms.intersection(_GENERAL_MEDICAL_TERMS)

    return False


def _normalized_query_terms(query: str) -> set[str]:
    raw_terms = {term for term in re.findall(r"[a-z0-9]+", query.lower()) if term}
    normalized_terms = set(raw_terms)
    vocabulary = set(_GENERAL_MEDICAL_GUIDANCE) | _GENERAL_MEDICAL_TERMS

    for term in raw_terms:
        if term in vocabulary:
            continue

        compact = re.sub(r"(.)\1+", r"\1", term)
        if compact in vocabulary:
            normalized_terms.add(compact)
            continue

        matches = get_close_matches(term, vocabulary, n=1, cutoff=0.78)
        if matches:
            normalized_terms.add(matches[0])

    return normalized_terms


def _general_medical_fallback(query: str) -> str:
    if _is_greeting(query):
        return "Hello. I can help with basic medical guidance. Ask about symptoms like fever, headache, stomach pain, cough, throat pain, or treatment options."

    query_terms = _normalized_query_terms(query)
    for key, response in _GENERAL_MEDICAL_GUIDANCE.items():
        if key in query_terms:
            return response

    for key, response in _COMMON_SYMPTOM_GUIDANCE.items():
        if key in query_terms:
            return response

    if "ache" in query_terms or "pain" in query_terms:
        if "stomach" in query_terms or "abdominal" in query_terms or "abdomen" in query_terms or "gastric" in query_terms:
            return _COMMON_SYMPTOM_GUIDANCE["stomach"]
        if "stone" in query_terms or "kidney" in query_terms:
            return _COMMON_SYMPTOM_GUIDANCE["stone"]
        return (
            "General body ache can happen with viral illness, dehydration, overexertion, poor sleep, or stress. Rest, hydrate, and monitor symptoms. "
            "Seek urgent care if pain is severe, one-sided with weakness, associated with breathing trouble, chest pain, high fever, confusion, or persistent vomiting."
        )

    return (
        "I can still help with basic medical guidance even when the documents do not contain this exact answer. "
        "Please share the main symptom, duration, age group, and red flags like high fever, breathing difficulty, severe pain, confusion, dehydration, bleeding, or repeated vomiting."
    )


def _looks_like_general_medical_question(query: str) -> bool:
    query_terms = _normalized_query_terms(query)
    return bool(
        query_terms.intersection(_GENERAL_MEDICAL_GUIDANCE)
        or query_terms.intersection(_GENERAL_MEDICAL_TERMS)
        or query_terms.intersection(_COMMON_SYMPTOM_GUIDANCE)
        or {"ache", "pain"}.intersection(query_terms)
    )


@lru_cache(maxsize=1)
def _load_local_chunks() -> list[dict]:
    if not DATA_DIR.exists():
        return []

    docs = load_pdf_file(data_dir=str(DATA_DIR))
    if not docs:
        return []

    return text_split(docs, chunk_size=800, overlap=80)


def _extract_best_snippet(query: str, contexts: list[str]) -> str:
    query_terms = {term for term in re.findall(r"[a-z0-9]+", query.lower()) if len(term) > 2}

    if not contexts:
        return "I could not find this in the uploaded documents."

    best_sentence = ""
    best_score = -1
    for context in contexts:
        sentences = re.split(r"(?<=[.!?])\s+|\n+", context)
        for sentence in sentences:
            sentence_terms = set(re.findall(r"[a-z0-9]+", sentence.lower()))
            score = len(query_terms.intersection(sentence_terms))
            if score > best_score and sentence.strip():
                best_score = score
                best_sentence = sentence.strip()

    if not best_sentence:
        best_sentence = contexts[0].strip()

    return best_sentence


def _expand_query_terms(query: str) -> set[str]:
    terms = {term for term in _normalized_query_terms(query) if len(term) > 2}
    expanded = set(terms)
    for term in terms:
        expanded.update(_QUERY_EXPANSIONS.get(term, set()))
    return expanded


def _is_no_answer_response(answer: str) -> bool:
    answer_lower = answer.lower()
    return any(pattern in answer_lower for pattern in _NO_ANSWER_PATTERNS)


def _rank_local_chunks(query: str, limit: int = 5) -> list[str]:
    chunks = _load_local_chunks()
    if not chunks:
        return []

    query_terms = _expand_query_terms(query)
    scored_chunks = []

    for chunk in chunks:
        text = chunk.get("text", "").strip()
        if not text:
            continue

        chunk_terms = set(re.findall(r"[a-z0-9]+", text.lower()))
        overlap = len(query_terms.intersection(chunk_terms))
        if overlap == 0:
            continue

        scored_chunks.append((overlap, len(text), text))

    scored_chunks.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    return [text for _, _, text in scored_chunks[:limit]]


def _backend_status() -> dict:
    pinecone_key = _first_env_value("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "medicalbot")

    if not pinecone_key:
        msg = "Set PINECONE_API_KEY and PINECONE_INDEX_NAME to enable live document retrieval."
        app.logger.warning(msg)
        return {
            "ready": False,
            "message": msg,
        }

    try:
        pc = Pinecone(api_key=pinecone_key)
        index_names = pc.list_indexes().names()
        if index_name not in index_names:
            msg = f"Pinecone connected, but the '{index_name}' index is missing. Run python store_index.py first."
            app.logger.warning(msg)
            return {
                "ready": False,
                "message": msg,
            }
    except Exception as exc:
        msg = f"Pinecone is unavailable: {str(exc)}"
        app.logger.error(msg)
        return {"ready": False, "message": msg}

    success_msg = f"Connected to the '{index_name}' Pinecone index."
    app.logger.info(success_msg)
    return {"ready": True, "message": success_msg}


@lru_cache(maxsize=1)
def _pinecone_index():
    status = _backend_status()
    if not status["ready"]:
        return None

    pinecone_key = _first_env_value("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "medicalbot")
    pc = Pinecone(api_key=pinecone_key)
    return pc.Index(index_name)


def _build_contexts(query: str, top_k: int = 3) -> list[str]:
    local_contexts = _rank_local_chunks(query)
    if local_contexts:
        return local_contexts

    pinecone_index = _pinecone_index()
    if pinecone_index is None:
        return []

    q_emb = embed_texts([query])[0]
    resp = pinecone_index.query(vector=q_emb, top_k=top_k, include_metadata=True)

    contexts = []
    matches = resp.get("matches", []) if isinstance(resp, dict) else getattr(resp, "matches", [])
    for match in matches:
        meta = match.get("metadata", {}) if isinstance(match, dict) else getattr(match, "metadata", {})
        score = match.get("score", 0.0) if isinstance(match, dict) else getattr(match, "score", 0.0)
        ctx = meta.get("text_preview") if isinstance(meta, dict) else None
        if not ctx:
            ctx = meta.get("text") if isinstance(meta, dict) else None
        if ctx and score >= 0.15:
            contexts.append(ctx)

    return contexts


def _generate_from_context(client: OpenAI, model_name: str, query: str, combined_context: str) -> str:
    from src.prompt import system_prompt

    user_msg = (
        "Answer only from the retrieved context. If the context does not contain the answer, say that you could not find it in the uploaded documents.\n\n"
        f"CONTEXT:\n{combined_context}\n\nQuestion: {query}"
    )
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt.format(context=combined_context)},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=500,
        temperature=0.2,
    )
    return completion.choices[0].message.content.strip()


def _generate_general_response(client: OpenAI, model_name: str, query: str) -> str:
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful medical information assistant. Give concise, non-diagnostic guidance and include a safety note when appropriate."
                ),
            },
            {"role": "user", "content": query},
        ],
        max_tokens=250,
        temperature=0.2,
    )
    return completion.choices[0].message.content.strip()


def _answer_query(query: str, top_k: int = 3) -> str:
    if _is_greeting(query):
        return _general_medical_fallback(query)

    try:
        contexts = _build_contexts(query, top_k=top_k)
    except Exception as e:
        app.logger.error(f"Error building contexts: {e}", exc_info=True)
        contexts = []
    
    combined_context = "\n---\n".join(contexts)

    if not combined_context and _looks_like_general_medical_question(query):
        return _general_medical_fallback(query)

    groq_enabled = _env_bool("USE_GROQ_CHAT", "true")
    groq_key = _first_env_value("GROQ_API_KEY")

    if not groq_enabled or not groq_key:
        app.logger.warning("Groq is disabled or API key is missing. Using fallback response.")
        return _general_medical_fallback(query)

    try:
        groq_client = OpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1")
    except Exception as e:
        app.logger.error(f"Failed to initialize Groq client: {e}")
        return _general_medical_fallback(query)

    if not combined_context:
        try:
            return _generate_general_response(
                groq_client,
                os.getenv("GROQ_MODEL_NAME", "llama-3.1-8b-instant"),
                query,
            )
        except Exception as e:
            app.logger.warning(f"Groq fallback failed: {e}")
            return _general_medical_fallback(query)

    try:
        answer = _generate_from_context(
            groq_client,
            os.getenv("GROQ_MODEL_NAME", "llama-3.1-8b-instant"),
            query,
            combined_context,
        )
        if _is_no_answer_response(answer) and _looks_like_general_medical_question(query):
            return _general_medical_fallback(query)
        return answer
    except Exception as e:
        app.logger.warning(f"Groq context generation failed: {e}")
        snippet = _extract_best_snippet(query, contexts)
        return f"Based on the uploaded documents: {snippet}"


@app.route("/")
def home():
    return render_template(
        "home.html",
        **{
            **_site_context(
                "home",
                "DigiHealthGuide | Home",
                "A simple user-first medical chatbot home page with a live demo and a short explanation of what the bot does.",
                highlights=[
                    "User-friendly medical chat assistant",
                    "Answers based on uploaded PDFs",
                    "Professional but simple presentation",
                ],
                user_steps=_USER_STEPS,
                user_benefits=_USER_BENEFITS,
                user_limits=_USER_LIMITS,
                demo_prompts=[
                    "What warning signs mean I should seek urgent care?",
                    "Summarize the treatment guidance in simple words.",
                    "What symptoms are described in the uploaded documents?",
                ],
            ),
        },
    )


@app.route("/about")
def about():
    return render_template(
        "about.html",
        **_site_context(
            "about",
            "About DigiHealthGuide",
            "Learn what the project is for, who should use it, and how it fits into a professional portfolio.",
            audience=[
                "Students building a document-grounded AI demo.",
                "Teams that need an internal medical knowledge assistant.",
                "Recruiters reviewing a polished end-to-end Flask project.",
            ],
            resume_points=[
                "Built an end-to-end Flask application with retrieval-based medical Q&A.",
                "Added a structured, multi-page interface for product-style presentation.",
                "Designed a safe fallback flow when the answer is not in the source content.",
            ],
        ),
    )


@app.route("/how-it-works")
def how_it_works():
    return render_template(
        "how_it_works.html",
        **_site_context(
            "how_it_works",
            "How DigiHealthGuide works",
            "See the retrieval pipeline from PDF ingestion to the final grounded answer.",
            steps=_PROJECT_STEPS,
        ),
    )


@app.route("/capabilities")
def capabilities():
    return render_template(
        "capabilities.html",
        **_site_context(
            "capabilities",
            "What DigiHealthGuide can and cannot do",
            "A clear scope statement for demos, resumes, and production expectations.",
            can_do=_CAN_DO,
            cannot_do=_CANNOT_DO,
        ),
    )


@app.route("/developers")
def developers():
    return render_template(
        "developers.html",
        **_site_context(
            "developers",
            "DigiHealthGuide for Developers",
            "Technical architecture, runtime behavior, environment variables, and implementation flow for developers who want to understand the project in depth.",
            developer_details=_DEVELOPER_DETAILS,
        ),
    )


@app.route("/chat")
def chat_page():
    return render_template(
        "chat.html",
        **_site_context(
            "chat",
            "Chat with DigiHealthGuide",
            "Ask questions against the uploaded medical documents using the live chat interface.",
            sample_prompts=[
                "What symptoms are mentioned for the condition in the PDFs?",
                "Give me a short summary of the treatment guidance.",
                "What warning signs mean I should seek urgent care?",
            ],
        ),
    )


@app.route("/get", methods=["POST"])
def chat():
    msg = request.values.get("msg", "").strip()
    if not msg:
        return "Message is required.", 400

    try:
        app.logger.info(f"Processing chat query: {msg[:100]}")
        answer = _answer_query(msg)
        app.logger.info(f"Chat answer generated: {answer[:100]}")
        return answer
    except Exception as exc:
        error_msg = f"Failed to generate a response: {str(exc)}"
        app.logger.error(error_msg, exc_info=True)
        return error_msg, 500


@app.route("/api/status")
def api_status():
    status = _backend_status()
    return jsonify(
        {
            "site": SITE_NAME,
            "backend": status,
            "documents_found": len(_load_local_chunks()),
        }
    )


if __name__ == "__main__":
    app.logger.info(f"Starting app on port 5000")
    app.logger.info(f"Groq enabled: {_env_bool('USE_GROQ_CHAT', 'true')}")
    app.logger.info(f"Backend status: {_backend_status()}")
    
    app.run(host="0.0.0.0", port=5000, debug=False)
