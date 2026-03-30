import json
import logging
import sys
import time
from itertools import cycle
from pathlib import Path
from typing import Any, TypedDict

import requests
import streamlit as st

from backend.config import get_settings


class ChatMessage(TypedDict):
    role: str
    content: str


PAGE_TITLE = "J.A.R"
BACKEND_URL = get_settings().backend_url
MAX_QUESTION_LENGTH = 500
BASE_DIR = Path(__file__).resolve().parent.parent
PHOTO_PATH = BASE_DIR / "assets" / "photo.png"
GITHUB_URL = "https://github.com/PavoToXx"
LINKEDIN_URL = "https://www.linkedin.com/in/josephdominguez-/"

DEFAULT_METRICS = {"latency_ms": None, "tokens_used": None}
STREAM_ENDPOINT = "/ask/stream"
TRANSLATIONS = {
    "es": {
        "title": "J.R.A",
        "subtitle": "RAG system sobre mi trayectoria profesional",
        "sidebar_title": "Sobre Joseph",
        "bio": "Joseph es un apasionado de la nube y en ML, centrado en la creación de productos de IA útiles, flujos de trabajo fiables y sistemas de recuperación prácticos.",
        "backend_ready": "Conexión exitosa",
        "backend_missing": "Conexión fallida",
        "suggested": "Preguntas sugeridas",
        "chat_placeholder": "Haz una pregunta sobre el ambito profesional de Joseph...",
        "thinking_states": [
            "J.R.A esta pensando.",
            "J.R.A esta pensando..",
            "J.R.A esta pensando...",
        ],
        "rate_limit": "Has alcanzado el limite de preguntas por hora. Intenta de nuevo en unos minutos.",
        "generic_error": "Ocurrio un error inesperado. Por favor intenta de nuevo.",
        "empty_warning": "Escribe una pregunta antes de enviarla.",
        "length_warning": f"Tu pregunta debe tener como maximo {MAX_QUESTION_LENGTH} caracteres.",
        "metrics_title": "Metricas",
        "latency": "Latencia",
        "tokens": "Tokens",
        "mode_toggle": "Modo nocturno",
        "language_label": "Idioma",
        "language_options": ["ES", "EN"],
        "github": "GitHub",
        "linkedin": "LinkedIn",
        "suggested_questions": [
            "En que proyectos ha trabajado?",
            "Cuales son sus skills?",
            "Que experiencia tiene?",
            "Que esta aprendiendo ahora?",
        ],
    },
    "en": {
        "title": "J.R.A",
        "subtitle": "RAG system about my professional journey",
        "sidebar_title": "About Joseph",
        "bio": "Joseph is a cloud and ML practitioner focused on building useful AI products, reliable pipelines, and practical retrieval systems.",
        "backend_ready": "Connection Successful",
        "backend_missing": "Connection Failed",
        "suggested": "Suggested questions",
        "chat_placeholder": "Ask about Joseph's professional background...",
        "thinking_states": [
            "J.R.A is thinking.",
            "J.R.A is thinking..",
            "J.R.A is thinking...",
        ],
        "rate_limit": "You have reached the hourly question limit. Please try again in a few minutes.",
        "generic_error": "An unexpected error occurred. Please try again.",
        "empty_warning": "Enter a question before sending it.",
        "length_warning": f"Your question must be {MAX_QUESTION_LENGTH} characters or fewer.",
        "metrics_title": "Metrics",
        "latency": "Latency",
        "tokens": "Tokens",
        "mode_toggle": "Dark mode",
        "language_label": "Language",
        "language_options": ["ES", "EN"],
        "github": "GitHub",
        "linkedin": "LinkedIn",
        "suggested_questions": [
            "What projects has he worked on?",
            "What are his key skills?",
            "What experience does he have?",
            "What is he learning right now?",
        ],
    },
}

LOGGER = logging.getLogger(__name__)


st.set_page_config(page_title=PAGE_TITLE, layout="wide")


def apply_custom_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        .* [data-testid="stAppViewContainer"] {
            font-family: 'Inter', Open Sans;
            background: radial-gradient(circle at 0% 0%, #091819 0%, #07354e 45%, #04122c 100%);
        }

        [data-testid="stMainBlockContainer"] {
            padding-top: 1rem;
        }

        .jr-header {
            border-radius: 28px;
            padding: 1.2rem 1.25rem 1rem 1.25rem;
            margin-bottom: 1rem;
        }

        .jr-kicker {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            margin-bottom: 0.4rem;
        }

        .jr-subtitle {
            margin-top: 0.35rem;
            margin-bottom: 0;
        }

        
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None
    if "language" not in st.session_state:
        st.session_state.language = "en"
    if "language_selector" not in st.session_state:
        st.session_state.language_selector = st.session_state.language
    if "last_response_metrics" not in st.session_state:
        st.session_state.last_response_metrics = DEFAULT_METRICS.copy()


def sync_language_from_selector() -> None:
    st.session_state.language = st.session_state.language_selector


def get_copy() -> dict[str, Any]:
    return TRANSLATIONS[st.session_state.language]


def update_metrics(latency_ms: int | None, tokens_used: int | None) -> None:
    st.session_state.last_response_metrics = {
        "latency_ms": latency_ms,
        "tokens_used": tokens_used,
    }


def render_sidebar() -> None:
    copy = get_copy()
    st.sidebar.title(copy["sidebar_title"])
    st.sidebar.radio(
        copy["language_label"],
        ("es", "en"),
        key="language_selector",
        format_func=lambda language_code: language_code.upper(),
        horizontal=True,
        on_change=sync_language_from_selector,
    )
    copy = get_copy()

    if PHOTO_PATH.exists():
        st.sidebar.image(str(PHOTO_PATH), width='stretch')
    else:
        st.sidebar.info("Add your photo to `frontend/assets/photo.png` to display it here.")

    st.sidebar.markdown(copy["bio"])
    st.sidebar.link_button(copy["github"], GITHUB_URL, width='stretch')
    st.sidebar.link_button(copy["linkedin"], LINKEDIN_URL, width='stretch')
    if BACKEND_URL:
        st.sidebar.caption(copy["backend_ready"])
    else:
        st.sidebar.warning(copy["backend_missing"])

    st.sidebar.markdown("---")
    st.sidebar.subheader(copy["metrics_title"])
    latency_value = st.session_state.last_response_metrics["latency_ms"]
    tokens_value = st.session_state.last_response_metrics["tokens_used"]
    st.sidebar.metric(copy["latency"], f"{latency_value} ms" if latency_value is not None else "--")
    st.sidebar.metric(copy["tokens"], str(tokens_value) if tokens_value is not None else "--")


def parse_sse_event(raw_event: str) -> tuple[str | None, str | None]:
    event_name = None
    data_lines: list[str] = []

    for line in raw_event.splitlines():
        if line.startswith("event:"):
            event_name = line.removeprefix("event:").strip()
        elif line.startswith("data:"):
            data_lines.append(line.removeprefix("data:").strip())

    if not data_lines:
        return event_name, None

    return event_name, "\n".join(data_lines)


def stream_backend_answer(question: str, placeholder) -> str:
    copy = get_copy()
    if not BACKEND_URL:
        LOGGER.error("BACKEND_URL is not configured.")
        placeholder.markdown(copy["generic_error"])
        return copy["generic_error"]

    accumulated_text = ""
    thinking_states = cycle(copy["thinking_states"])
    update_metrics(None, None)

    for _ in range(3):
        placeholder.markdown(next(thinking_states))
        time.sleep(0.16)

    try:
        with requests.post(
            f"{BACKEND_URL}{STREAM_ENDPOINT}",
            json={"question": question},
            headers={"Accept": "text/event-stream"},
            timeout=30,
            stream=True,
        ) as response:
            if response.status_code == 429:
                LOGGER.warning("Rate limit reached for chat request.")
                placeholder.markdown(copy["rate_limit"])
                return copy["rate_limit"]

            if response.status_code != 200:
                LOGGER.error("Unexpected backend status. status=%s", response.status_code)
                placeholder.markdown(copy["generic_error"])
                return copy["generic_error"]

            event_lines: list[str] = []
            for line in response.iter_lines(decode_unicode=True):
                if line == "":
                    event_name, data = parse_sse_event("\n".join(event_lines))
                    event_lines = []

                    if not event_name or not data:
                        continue

                    payload = json.loads(data)
                    if event_name == "chunk":
                        chunk = payload.get("content", "")
                        if chunk:
                            accumulated_text += chunk
                            placeholder.markdown(accumulated_text)
                    elif event_name == "meta":
                        update_metrics(payload.get("latency_ms"), payload.get("tokens_used"))
                    elif event_name == "error":
                        message = payload.get("message", copy["generic_error"])
                        placeholder.markdown(message)
                        return message
                    continue

                event_lines.append(line)
    except requests.RequestException:
        LOGGER.exception("Connection error during backend request.")
        placeholder.markdown(copy["generic_error"])
        return copy["generic_error"]
    except json.JSONDecodeError:
        LOGGER.exception("Invalid JSON payload received from streaming endpoint.")
        placeholder.markdown(copy["generic_error"])
        return copy["generic_error"]

    if not accumulated_text.strip():
        LOGGER.error("Streaming backend returned an empty response.")
        placeholder.markdown(copy["generic_error"])
        return copy["generic_error"]

    return accumulated_text


def render_chat_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def add_message(role: str, content: str) -> None:
    st.session_state.messages.append({"role": role, "content": content})


def validate_question(prompt: str) -> str | None:
    copy = get_copy()
    normalized_prompt = prompt.strip()

    if not normalized_prompt:
        st.warning(copy["empty_warning"])
        return None

    if len(normalized_prompt) > MAX_QUESTION_LENGTH:
        st.warning(copy["length_warning"])
        return None

    return normalized_prompt


def render_header() -> None:
    copy = get_copy()
    st.markdown(
        f"""
        <div class="jr-header">
            <h1>{copy["title"]}</h1>
            <div class="jr-kicker">Joseph Retrieval Assistant</div>
            <p class="jr-subtitle">{copy["subtitle"]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_suggested_questions() -> None:
    copy = get_copy()
    st.caption(copy["suggested"])
    columns = st.columns(2)

    for index, question in enumerate(copy["suggested_questions"]):
        with columns[index % 2]:
            if st.button(question, width='stretch', key=f"suggested_{index}"):
                st.session_state.pending_prompt = question


def handle_chat_turn(prompt: str) -> None:
    validated_prompt = validate_question(prompt)
    if validated_prompt is None:
        return

    add_message("user", validated_prompt)
    with st.chat_message("user"):
        st.markdown(validated_prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        answer = stream_backend_answer(validated_prompt, placeholder)

    add_message("assistant", answer)


def main() -> None:
    initialize_session_state()
    apply_custom_css()
    render_sidebar()
    render_header()
    render_suggested_questions()
    render_chat_history()

    prompt = st.chat_input(get_copy()["chat_placeholder"])
    queued_prompt = st.session_state.pending_prompt

    if prompt is not None:
        handle_chat_turn(prompt)
    elif queued_prompt:
        st.session_state.pending_prompt = None
        handle_chat_turn(queued_prompt)


if __name__ == "__main__":
    main()
