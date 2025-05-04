import streamlit as st
from typing import List, Dict

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain.schema import Document


st.set_page_config(
    page_title="AWS Well-Architected Lens Identifier",
    layout="wide",
    initial_sidebar_state="auto",
)

st.title("🔍 AWS Well-Architected Lens Identifier")


@st.cache_resource(show_spinner=False)
def load_resources():
    hf_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    index = FAISS.load_local(
        "prueba/wa_lenses_index_hf",
        hf_embeddings,
        allow_dangerous_deserialization=True
    )

    prompt_template = PromptTemplate(
        template="""
You are a classifier of AWS Well-Architected Lenses.
Given this documentation fragment (context):
{context}

And the user’s input (question):
{question}

Respond ONLY with the key of the lens (e.g. generative-ai, analytics, serverless), nothing else.
Lens:
""".strip(),
        input_variables=["context", "question"]
    )

    llm = OllamaLLM(model="gemma3", temperature=0)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=index.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template},
    )

    return qa

qa = load_resources()


def classify_lens(text: str) -> Dict:
    """
    Ejecuta el chain RAG y devuelve:
      - result: la clave del lens
      - source_documents: lista de Document con metadata
    """
    return qa({"query": text})


st.markdown(
    """
    Ingresa un fragmento de texto o descripción y pulsa **Identificar Lens**.
    """
)

user_input = st.text_area(
    label="📋 Texto a clasificar",
    height=200,
    placeholder="Pega aquí tu texto…"
)

if st.button("🔍 Identificar Lens"):
    if not user_input.strip():
        st.warning("Por favor ingresa algún texto.")
    else:
        with st.spinner("Clasificando…"):
            result = classify_lens(user_input)

        st.markdown("### 🏷️ Lens identificado:")
        st.success(f"`{result['result'].strip()}`")

        st.markdown("### 📚 Fragmentos de documentación consultados:")
        for doc in result["source_documents"]:
            lens = doc.metadata.get("lens", "unknown")
            src  = doc.metadata.get("source", "")
            st.write(f"- `{lens}`  —  {src}")


st.markdown("---")
st.caption("Construido con Streamlit · LangChain · FAISS · Ollama· Equipo17")