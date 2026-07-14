import streamlit as st
import pandas as pd

import config
from rag_chain import run_pipeline

st.set_page_config(page_title="PCB Design Research RAG", layout="wide")

st.title("🔧 PCB Design Research RAG")
st.caption(
    "Weighted-keyword web research + Pixel-RAG (screenshot tiling) + open-source-model "
    "synthesis via Groq + Hugging Face. Embeddings/search stay local & key-less."
)

with st.sidebar:
    st.header("Settings")
    use_pixel_rag = st.checkbox("Enable Pixel-RAG (screenshot tiling + vision model)", value=config.ENABLE_PIXEL_RAG_DEFAULT)
    max_sources = st.slider("Max sources to fetch", 4, 30, config.MAX_TOTAL_SOURCES)
    st.markdown("---")
    st.markdown(f"**LLM:** `{config.GROQ_MODEL}` (Groq API)")
    st.markdown(f"**Vision model:** `{config.HF_VISION_MODEL}` (Hugging Face Inference API)")
    st.markdown(f"**Embeddings:** `{config.EMBEDDING_MODEL}` (local)")
    st.markdown("---")
    groq_ok = "✅" if config.GROQ_API_KEY else "❌ missing GROQ_API_KEY"
    hf_ok = "✅" if config.HF_TOKEN else "❌ missing HF_TOKEN"
    st.markdown(f"Groq key: {groq_ok}")
    st.markdown(f"HF token: {hf_ok}")
    st.markdown(
        "Set `GROQ_API_KEY` and `HF_TOKEN` as environment variables or in a `.env` file. "
        "Also run `playwright install chromium` once."
    )

default_prompt = (
    "Design an ultra-low-power precision voltage reference module producing selectable "
    "outputs of 2.5 V, 5 V, and 10 V while consuming less than 500 µW. Use a buried-zener "
    "or low-power bandgap reference, zero-drift buffer amplifier, ultra-low-TCR resistor "
    "networks, and ultra-low-IQ LDO regulators. Include trimming capability, "
    "reverse-polarity protection, and estimate temperature drift, output noise, and "
    "long-term stability."
)

query = st.text_area("PCB design prompt", value=default_prompt, height=140)
run_btn = st.button("🚀 Run Research Pipeline", type="primary")

if run_btn and query.strip():
    progress_area = st.empty()
    log_lines = []

    def progress_cb(msg):
        log_lines.append(msg)
        progress_area.info("\n\n".join(log_lines[-6:]))

    with st.spinner("Running pipeline — this can take a while (screenshots + local model calls)..."):
        try:
            result = run_pipeline(
                query, use_pixel_rag=use_pixel_rag, max_sources=max_sources, progress_cb=progress_cb
            )
        except Exception as e:
            st.error(f"Pipeline failed: {e}")
            st.stop()

    progress_area.empty()
    st.success("Done.")

    st.subheader("📝 Research Brief")
    st.markdown(result["answer"])

    st.subheader("⚖️ Weighted Keywords")
    kw_df = pd.DataFrame(result["weighted_keywords"], columns=["Keyword", "Weight"])
    c1, c2 = st.columns([1, 2])
    with c1:
        st.dataframe(kw_df, use_container_width=True, hide_index=True)
    with c2:
        st.bar_chart(kw_df.set_index("Keyword"))

    usable = [s for s in result["sources"] if not s.get("blocked")]
    blocked = [s for s in result["sources"] if s.get("blocked")]

    st.subheader("🌐 Sources Fetched")
    st.caption(f"{len(usable)} usable · {len(blocked)} blocked/unavailable")
    for s in usable:
        with st.expander(f"[{s['score']:.2f}] {s['title'] or s['url']}"):
            st.write(s["url"])
            st.caption(f"Matched keyword: *{s['keyword']}* · text chunks: {s['text_chunks']} · pixel chunks: {s['pixel_chunks']}")
            if s["screenshot_path"]:
                try:
                    st.image(s["screenshot_path"], caption="Full-page screenshot (Pixel-RAG source)", use_container_width=True)
                except Exception:
                    pass

    if blocked:
        with st.expander(f"🚫 {len(blocked)} source(s) blocked or unavailable (skipped)"):
            for s in blocked:
                st.write(f"- {s['url']} — *{s['block_reason']}*")

    st.subheader("🔍 Top Retrieved Chunks Used for the Brief")
    for i, (doc, score) in enumerate(result["retrieved"], start=1):
        with st.expander(f"[S{i}] {doc.metadata.get('type')} · relevance {score:.2f} · {doc.metadata.get('source')}"):
            st.write(doc.page_content)

elif run_btn:
    st.warning("Please enter a design prompt first.")
