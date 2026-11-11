import streamlit as st
from rag_engine import OpenSourceRAGTutor, STYLE_PROMPTS

st.set_page_config(
    page_title="Smart Subject Tutor v1.0 (Open-Source RAG)",
    page_icon="🎓",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading Open-Source Embedding & LLM Models (Zero API Keys)...")
def load_tutor_engine() -> OpenSourceRAGTutor:
    return OpenSourceRAGTutor(notes_path="data/subject_notes.md")


tutor = load_tutor_engine()

# =====================================================================
# SIDEBAR: SUBJECT TELEMETRY & STYLE SELECTOR
# =====================================================================
with st.sidebar:
    st.header("⚙️ Tutor Controls (v1.0)")
    st.success("🔓 100% Open-Source Stack (No API Key)")
    st.markdown(f"**Active Subject:** `{tutor.subject_title}`")
    st.markdown(f"**Indexed Knowledge Chunks:** `{tutor.chunk_count}`")

    selected_style = st.radio(
        "🎨 Select Explanation Style:",
        options=list(STYLE_PROMPTS.keys()),
        index=0,
    )

    top_k = st.slider("🔍 Top-K Retrieved Chunks:", min_value=1, max_value=3, value=2)

    if st.button("🔄 Reload data/subject_notes.md"):
        st.cache_resource.clear()
        st.rerun()

# =====================================================================
# MAIN INTERFACE
# =====================================================================
st.title("🎓 Smart Subject Tutor (v1.0)")
st.caption(
    f"Grounded in `data/subject_notes.md` ({tutor.subject_title}) • "
    "Embeddings: `all-MiniLM-L6-v2` • LLM: `Qwen2.5-0.5B-Instruct` • Vector DB: `ChromaDB`"
)

st.info(
    f"**Current Persona Active:** `{selected_style}` — Ask the exact same question under "
    "different styles to see how the tutor adapts its teaching voice while staying grounded in your notes!"
)

default_question = "What is a Deadlock and what are the four conditions required for it?"
user_query = st.text_input(
    "Ask a question about your chosen subject:",
    value=default_question,
)

if st.button("🚀 Generate Style-Adapted Answer", type="primary"):
    if not user_query.strip():
        st.warning("Please enter a question first.")
    else:
        with st.spinner(f"Retrieving chunks & generating '{selected_style}' explanation..."):
            answer, retrieved_chunks = tutor.generate_styled_answer(
                query=user_query,
                style_name=selected_style,
                top_k=top_k,
            )

        col1, col2 = st.columns([3, 2])

        with col1:
            st.subheader(f"💡 Tutor Response ({selected_style})")
            st.markdown(answer)

        with col2:
            st.subheader("📚 Retrieved RAG Context & Cosine Similarity")
            for idx, chunk in enumerate(retrieved_chunks, start=1):
                with st.expander(
                    f"Rank #{idx}: {chunk['topic']} (cos θ = {chunk['similarity']})",
                    expanded=(idx == 1),
                ):
                    st.write(chunk["content"])
                