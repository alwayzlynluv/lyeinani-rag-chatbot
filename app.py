"""
LYEINANI chat app (Streamlit) on top of rag_bot.py.

Run:
    pip install streamlit
    streamlit run app.py
"""

import streamlit as st

from rag_bot import QA_THRESHOLD, LyeinaniBot

st.set_page_config(page_title="LYEINANI Assistant", layout="centered")


@st.cache_resource(show_spinner="Loading models and index...")
def get_bot():
    return LyeinaniBot()


bot = get_bot()

# ---- sidebar: controls that make the routing visible for a demo
with st.sidebar:
    st.header("Settings")
    bot.qa_threshold = st.slider(
        "Q&A match threshold", 0.50, 0.95, QA_THRESHOLD, 0.01,
        help="Similarity needed to return a stored Q&A answer instead of searching the site.",
    )
    show_details = st.toggle("Show routing details", value=True)
    st.caption(f"{len(bot.qa.pairs)} Q&A pairs · {len(bot.pdfs.chunks)} site chunks")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

st.title("LYEINANI Assistant")
st.caption("Ask about orders, shipping, returns, preorders, products or Brand Support.")

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_details(meta):
    if meta["route"] == "qa":
        st.caption(f"Answered from Q&A · score {meta['score']} · matched \"{meta['matched_question']}\"")
    elif meta["route"] == "rag":
        st.caption(f"Answered from site pages (RAG) · best Q&A score {meta['score']}")
    else:
        st.caption("No relevant content found · fallback answer")
    links = [s for s in meta["sources"] if s.get("url")]
    if links:
        with st.expander("Sources"):
            for s in links:
                prefix = f"[{s['n']}] " if "n" in s else ""
                st.markdown(f"- {prefix}[{s['title']}]({s['url']})")


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if show_details and m.get("meta"):
            render_details(m["meta"])

if question := st.chat_input("Type your question..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = bot.ask(question, history)
            except Exception as e:  # e.g. missing API key or network error
                result = {"answer": f"Sorry, something went wrong: {e}", "route": "fallback",
                          "score": 0, "matched_question": None, "sources": []}
        st.markdown(result["answer"])
        if show_details:
            render_details(result)

    st.session_state.messages.append({"role": "assistant", "content": result["answer"], "meta": result})
