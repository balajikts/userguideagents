"""Streamlit chat UI for the Electronics User-Guide Assistant.

Run: streamlit run ui/streamlit_app.py   (API_URL defaults to http://localhost:8000)
"""

import os
import sys
from pathlib import Path

import httpx
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from render import answer_markdown, cited_sources, source_label  # noqa: E402

API_URL = os.environ.get("API_URL", "http://localhost:8000")
SUGGESTIONS = {
    ":material/headphones: Reset Sony WH-1000XM5": "How do I factory reset my Sony WH-1000XM5?",
    ":material/tv: Subtitles on LG C3": "How do I turn on subtitles on my LG OLED C3?",
    ":material/router: TP-Link Wi-Fi password": "How do I change the Wi-Fi password on a TP-Link Archer AX55?",
}

st.set_page_config(page_title="Device Guide", page_icon=":material/menu_book:")
st.title(":material/menu_book: Device Guide")
st.caption("Step-by-step answers from official manuals, with a source for every step.")

if "messages" not in st.session_state:
    st.session_state.messages = []  # {"role", "content", "answer"?}
    st.session_state.session_id = None


def render_answer(answer: dict) -> None:
    st.markdown(answer_markdown(answer))
    for w in answer.get("warnings", []):
        st.warning(w, icon=":material/warning:")
    sources = cited_sources(answer) if answer.get("status") != "needs_clarification" else []
    if sources:
        with st.expander(f"Sources ({len(sources)})", type="compact"):
            for s in sources:
                st.markdown(f"**{s['id']}** · [{source_label(s)}]({s['url']}) · trust {s['trust_score']:.2f}")


def ask(question: str) -> dict:
    # Forward the browser's IP (set by Caddy) so the API rate-limits per user, not per UI container.
    client_ip = st.context.headers.get("X-Forwarded-For") or st.context.ip_address
    client_ip = client_ip if isinstance(client_ip, str) else ""
    r = httpx.post(
        f"{API_URL}/ask",
        json={"question": question, "session_id": st.session_state.session_id},
        headers={"X-Forwarded-For": client_ip} if client_ip else {},
        timeout=120,
    )
    if r.status_code == 429:
        return {"status": "rejected", "summary": "You're sending questions too quickly. Please wait a minute."}
    r.raise_for_status()
    body = r.json()
    ans = body["answer"]
    # Keep the session only while the assistant is waiting on a clarification.
    st.session_state.session_id = body["session_id"] if ans["status"] == "needs_clarification" else None
    return ans


with st.sidebar:
    if st.button("New conversation", icon=":material/refresh:", width="stretch"):
        st.session_state.messages = []
        st.session_state.session_id = None
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if "answer" in msg:
            render_answer(msg["answer"])
        else:
            st.markdown(msg["content"])

prompt = st.chat_input("Ask how to do something with your device", submit_mode="disable")
if not st.session_state.messages and not prompt:
    picked = st.pills("Try asking:", list(SUGGESTIONS), label_visibility="collapsed")
    if picked:
        prompt = SUGGESTIONS[picked]

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.status(":shimmer[Checking the manuals]", type="compact") as status:
            try:
                answer = ask(prompt)
                status.update(label="Done", state="complete")
            except httpx.HTTPError as e:
                answer = {"status": "rejected", "summary": f"The assistant is unavailable right now ({type(e).__name__})."}
                status.update(label="Failed", state="error")
        render_answer(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer_markdown(answer), "answer": answer})
