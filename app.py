# app.py
import streamlit as st
import os
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
from huggingface_hub import login
from sklearn.metrics.pairwise import cosine_similarity
from data_loader import load_and_combine_datasets

# 1. Hugging Face Token Login (optional if set via env)
HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN", "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")

# 2. Load Embedding Model
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# 3. Load LLM (Mistral)
@st.cache_resource
def load_llm():
    try:
        model_id = "mistralai/Mistral-7B-Instruct-v0.1"
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=HUGGINGFACE_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype="auto",
            use_auth_token=HUGGINGFACE_TOKEN
        )
        return pipeline("text-generation", model=model, tokenizer=tokenizer)
    except Exception as e:
        print(f"⚠️ Failed to load Mistral LLM: {e}")
        return None

# 4. Load and Embed Dataset
@st.cache_data
def load_dataset_and_embeddings():
    df = load_and_combine_datasets()

    if 'text' not in df.columns or 'label' not in df.columns:
        st.error("Dataset must contain 'text' and 'label' columns.")
        st.stop()

    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(str).tolist()

    embed_model = load_embedding_model()
    embeddings = embed_model.encode(texts, show_progress_bar=True)
    return texts, labels, embeddings, embed_model

# 5. Build FAISS Index
def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

# 6. RAG Answering
def generate_answer(query, texts, embeddings, model, index, labels, llm_pipe):
    query_vec = model.encode([query])
    D, I = index.search(query_vec, k=3)
    retrieved_texts = [texts[i] for i in I[0]]
    retrieved_labels = [labels[i] for i in I[0]]

    context = "\n".join(retrieved_texts)
    prompt = f"""Answer the following financial query using the given context:
Context:
{context}

Query:
{query}

Answer:"""

    result = llm_pipe(prompt, max_new_tokens=200, do_sample=True, temperature=0.7)[0]['generated_text']
    answer = result.split("Answer:")[-1].strip()
    return answer, retrieved_labels

# 7. Streamlit UI
def main():
    st.set_page_config(page_title="💸 GenAI Financial Chatbot", layout="centered")
    st.markdown("""
        <div style='text-align:center;'>
            <h1>💬 GenAI Financial Chatbot</h1>
            <p style='font-size:18px;'>Ask about finance, fraud detection, or company-related queries powered by LLM + RAG.</p>
        </div>
    """, unsafe_allow_html=True)

    with st.spinner("📦 Loading models and data..."):
        texts, labels, embeddings, embed_model = load_dataset_and_embeddings()
        index = build_faiss_index(np.array(embeddings))
        llm_pipe = load_llm()
        use_llm = llm_pipe is not None

    st.markdown("---")
    query = st.text_input("🔍 Enter your financial question:", placeholder="e.g., What is a suspicious transaction?")

    if query:
        with st.spinner("🤔 Generating answer..."):
            if use_llm:
                answer, related = generate_answer(query, texts, embeddings, embed_model, index, labels, llm_pipe)
                st.success("📌 Answer (via LLM):")
                st.write(answer)
            else:
                query_vec = embed_model.encode([query])
                similarities = cosine_similarity(query_vec, embeddings)[0]
                idx = np.argmax(similarities)
                st.success("📌 Most Similar Match:")
                st.write(texts[idx])
                st.info(f"🔖 Label: {labels[idx]}")
                related = [labels[idx]]

        if related:
            st.markdown("### 🧾 Related Labels:")
            st.markdown("\n".join([f"- {label}" for label in related]))

if __name__ == "__main__":
    main()
