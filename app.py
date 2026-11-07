"""
LYEINANI Assistant: Streamlit chat app on top of rag_bot.py.

Run:
    streamlit run app.py

Features:
- branded chat UI with suggested starter questions
- every answer labeled by where it came from (my Q&A list or my site pages) with source links
- thumbs up / down feedback on each answer, saved to feedback.csv
- download the conversation as a text file
- per-session question limit to protect a public demo
- developer panel: Q&A match threshold slider and ChromaDB stats
"""

import csv
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

from rag_bot import QA_THRESHOLD, LyeinaniBot

BRAND = "#F04A23"  # LYEINANI red-orange
MAX_QUESTIONS = 30  # per browser session
FEEDBACK_FILE = Path("feedback.csv")
SUGGESTIONS = [
    "Who pays for return shipping?",
    "When will my preorder ship?",
    "How much is the Crimson Bloom Blazer?",
    "What's in the Tokyo 1992 collection?",
    "Do you ship internationally?",
    "Do you offer brand support services?",
]

st.set_page_config(page_title="LYEINANI Assistant", page_icon=":material/auto_awesome:", layout="centered")

st.markdown(
    f"""
<style>
  .block-container {{ padding-top: 2rem; max-width: 760px; }}
  .ly-hero {{ text-align: center; margin-bottom: 1.2rem; }}
  .ly-hero h1 {{ font-size: 2.1rem; letter-spacing: 0.35em; margin: 0; font-weight: 800; }}
  .ly-hero h1 span {{ color: {BRAND}; }}
  .ly-hero p {{ opacity: 0.75; margin: 0.35rem 0 0; }}
  .ly-badge {{ display: inline-block; font-size: 0.72rem; font-weight: 600; letter-spacing: 0.04em;
              padding: 2px 10px; border-radius: 999px; margin: 4px 6px 2px 0; }}
  .ly-qa  {{ background: rgba(240, 74, 35, 0.15); color: {BRAND}; border: 1px solid rgba(240, 74, 35, 0.4); }}
  .ly-rag {{ background: rgba(120, 90, 255, 0.14); color: #8C7BFF; border: 1px solid rgba(120, 90, 255, 0.4); }}
  .ly-fb  {{ background: rgba(150, 150, 150, 0.15); color: inherit; border: 1px solid rgba(150, 150, 150, 0.4); }}
  .ly-src {{ font-size: 0.85rem; opacity: 0.85; }}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the models and the ChromaDB knowledge base...")
def get_bot():
    return LyeinaniBot()


bot = get_bot()

ss = st.session_state
ss.setdefault("messages", [])
ss.setdefault("asked", 0)
ss.setdefault("pending", None)


def save_feedback(idx):
    """Log a thumbs up/down for the answer at messages[idx]."""
    rating = ss.get(f"fb_{idx}")
    msg = ss.messages[idx]
    if rating is None or msg.get("rated"):
        return
    msg["rated"] = True
    question = ss.messages[idx - 1]["content"] if idx > 0 else ""
    new = not FEEDBACK_FILE.exists()
    with FEEDBACK_FILE.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time_utc", "rating", "route", "question", "answer"])
        w.writerow([datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "up" if rating == 1 else "down", msg["meta"]["route"], question, msg["content"]])


def transcript():
    lines = [f"LYEINANI Assistant chat, {datetime.now().strftime('%Y-%m-%d %H:%M')}", ""]
    for m in ss.messages:
        who = "You" if m["role"] == "user" else "Assistant"
        lines.append(f"{who}: {m['content']}")
        for s in (m.get("meta") or {}).get("sources", []):
            if s.get("url"):
                lines.append(f"   source: {s['url']}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown(f"### <span style='color:{BRAND}'>✦</span> About", unsafe_allow_html=True)
    st.write(
        "I answer questions about LYEINANI orders, shipping, returns, preorders, "
        "products and Brand Support, using the official site and the brand's own answers."
    )
    st.caption("Need a person? Email info@lyeinaniofficial.com")

    st.divider()
    left = MAX_QUESTIONS - ss.asked
    st.caption(f"Questions left this session: {left}/{MAX_QUESTIONS}")
    c1, c2 = st.columns(2)
    if c1.button("Clear chat", use_container_width=True):
        ss.messages, ss.pending = [], None
        st.rerun()
    c2.download_button("Save chat", transcript(), file_name="lyeinani_chat.txt",
                       use_container_width=True, disabled=not ss.messages)

    with st.expander("Developer settings"):
        bot.qa_threshold = st.slider(
            "Q&A match threshold", 0.50, 0.95, QA_THRESHOLD, 0.01,
            help="Similarity needed to return a stored Q&A answer instead of searching the site.",
        )
        show_details = st.toggle("Show match scores", value=False)
        st.caption(
            f"Vector database: ChromaDB  \n"
            f"Q&A phrasings: {bot.qa.col.count()} ({len(bot.qa.pairs)} pairs)  \n"
            f"Site chunks: {bot.pdfs.count}  \n"
            f"Answer model: {getattr(bot.llm, 'name', 'Claude')}"
        )
        if FEEDBACK_FILE.exists():
            rows = list(csv.DictReader(FEEDBACK_FILE.open(encoding="utf-8")))
            ups = sum(r["rating"] == "up" for r in rows)
            st.caption(f"Feedback: {ups} 👍 / {len(rows) - ups} 👎")

# ---------------------------------------------------------------- header
st.markdown(
    "<div class='ly-hero'><h1>LYEIN<span>A</span>NI</h1>"
    "<p>Customer assistant · anime-inspired urban fashion</p></div>",
    unsafe_allow_html=True,
)


def render_meta(meta, idx):
    route = meta["route"]
    if route == "qa":
        badge = "<span class='ly-badge ly-qa'>✦ FROM LYEINANI'S ANSWERS</span>"
    elif route == "rag":
        badge = "<span class='ly-badge ly-rag'>⌕ FROM THE WEBSITE</span>"
    else:
        badge = "<span class='ly-badge ly-fb'>NOT FOUND ON THE SITE</span>"
    if show_details:
        badge += f"<span class='ly-src'>match score {meta['score']}</span>"
    st.markdown(badge, unsafe_allow_html=True)

    links, seen = [], set()
    for s in meta["sources"]:
        url = s.get("url")
        if url and url not in seen:
            seen.add(url)
            if route == "qa":  # name the policy page, e.g. ".../refund-exchange-policy" -> "Refund Exchange Policy"
                title = url.rstrip("/").rsplit("/", 1)[-1].replace("-", " ").title()
            else:
                title = s["title"].split(" – ")[0] if s.get("title") else url
            links.append(f"[{title}]({url})")
    if links:
        st.markdown("<span class='ly-src'>Sources: " + " · ".join(links) + "</span>",
                    unsafe_allow_html=True)
    st.feedback("thumbs", key=f"fb_{idx}", on_change=save_feedback, args=(idx,),
                disabled=ss.messages[idx].get("rated", False))


# ---------------------------------------------------------------- conversation
if not ss.messages:
    st.markdown("**Try asking:**")
    cols = st.columns(2)
    for i, q in enumerate(SUGGESTIONS):
        if cols[i % 2].button(q, key=f"sugg_{i}", use_container_width=True):
            ss.pending = q
            st.rerun()

for i, m in enumerate(ss.messages):
    with st.chat_message(m["role"], avatar=":material/auto_awesome:" if m["role"] == "assistant" else None):
        st.markdown(m["content"])
        if m.get("meta"):
            render_meta(m["meta"], i)

typed = st.chat_input("Ask about orders, shipping, returns, products...",
                      disabled=ss.asked >= MAX_QUESTIONS)
question = typed or ss.pending
ss.pending = None

if ss.asked >= MAX_QUESTIONS:
    st.info("You've reached the question limit for this session. "
            "Email info@lyeinaniofficial.com and the team will help you.")
elif question:
    ss.asked += 1
    history = [{"role": m["role"], "content": m["content"]} for m in ss.messages]
    ss.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant", avatar=":material/auto_awesome:"):
        with st.spinner("Searching LYEINANI..."):
            try:
                result = bot.ask(question, history)
            except Exception as e:
                result = {"answer": f"Sorry, something went wrong on my side ({e}). "
                                    "Please try again or email info@lyeinaniofficial.com.",
                          "route": "fallback", "score": 0, "matched_question": None, "sources": []}
        ss.messages.append({"role": "assistant", "content": result["answer"], "meta": result})
    st.rerun()
