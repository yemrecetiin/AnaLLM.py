import os
import warnings
import shutil
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.chains import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Uyarıları kapat
warnings.filterwarnings("ignore", category=DeprecationWarning)

# OpenAI API anahtarını ayarla
os.environ["OPENAI_API_KEY"] = "your openai api key"


# PDF'den Anayasa metnini yükle
loader = PyPDFLoader("anayasa_eng.pdf")
documents = loader.load()

# Metni küçük parçalara böl
text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=60)
splits = text_splitter.split_documents(documents)

# OpenAI Embedding oluşturma
embedding = OpenAIEmbeddings(model="text-embedding-ada-002")

# Chroma veritabanını sıfırla
if os.path.exists("./chroma_db"):
    shutil.rmtree("./chroma_db")

# Yeni Chroma vektör veritabanını oluştur
vectorstore = Chroma.from_documents(splits, embedding)
retriever = vectorstore.as_retriever()

# LLM başlat
llm = Ollama(model="mistral", temperature=0.2, top_p=0.9)

# Prompting şablonları
zero_shot_prompt = PromptTemplate(input_variables=["question"], template="Question: {question}\nAnswer:")

examples = [
    {"question": "What is the official language of the Republic of Turkey?", "answer": "Turkish."},
    {"question": "How many years does the President serve?", "answer": "5 years."},
    {"question": "What is the capital of the Republic of Turkey?", "answer": "Ankara."},
    {"question": "Who was the first President of the Republic of Turkey?", "answer": "Mustafa Kemal Atatürk."},
    {"question": "What is the highest judicial body in Turkey?", "answer": "The Constitutional Court."},
    {"question": "How are laws passed in Turkey?",
     "answer": "Laws are enacted by the Grand National Assembly of Turkey."},
    {"question": "What are the fundamental rights in Turkey?", "answer": "Right to life, freedom, and equality."},
    {"question": "How is constitutional amendment made in Turkey?",
     "answer": "Requires 3/5 majority in the Grand National Assembly of Turkey."},
    {"question": "Who can vote in Turkey?", "answer": "Turkish citizens aged 18 and above."},
    {"question": "How many members are in the Turkish Parliament?", "answer": "600."},
    {"question": "Which body exercises legislative power in Turkey?",
     "answer": "The Grand National Assembly of Turkey."},
    {"question": "What is the national flag of Turkey?", "answer": "A red background with a white star and crescent."}
]

example_template = """
Question: {question}
Answer: {answer}
"""

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=PromptTemplate(input_variables=["question", "answer"], template=example_template),
    suffix="Question: {question}\nAnswer:",
    input_variables=["question"]
)


def get_prompting_type(question, few_shot=True):
    return few_shot_prompt.format(question=question) if few_shot else zero_shot_prompt.format(question=question)


tqa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)


def evaluate_answer(user_input, few_shot=True):
    prompt = get_prompting_type(user_input, few_shot)
    response = tqa_chain.invoke({"query": user_input})

    answer = response.get("result", "No answer found.")
    sources = response.get("source_documents", [])

    source_texts = " ".join([getattr(doc, "page_content", "") for doc in sources if hasattr(doc, "page_content")])

    if source_texts.strip():
        relevance_score = sum(1 for word in answer.split() if word in source_texts) / (len(answer.split()) or 1)
    else:
        relevance_score = 0

    confidence = round(relevance_score * 100, 2)
    hallucination_flag = confidence < 50

    return answer, confidence, hallucination_flag, sources


st.title("Turkish Constitution Chatbot 🇹🇷📜")
few_shot_toggle = st.checkbox("Use Few-Shot Prompting", value=True)

for message in st.session_state.get("history", []):
    role = "You" if message["role"] == "user" else "Bot"
    st.text_area(role, value=message["content"], height=75, disabled=True)

user_input = st.text_input("You: ", key="input")
if st.button("Send"):
    if user_input.lower() == "exit":
        st.stop()
    else:
        answer, confidence, hallucination_flag, sources = evaluate_answer(user_input, few_shot=few_shot_toggle)

        st.session_state.setdefault("history", []).append({"role": "user", "content": user_input})
        st.session_state.setdefault("history", []).append({"role": "assistant", "content": answer})

        st.text_area("Bot", value=answer, height=100, disabled=True)
        st.write(f"Confidence Score: {confidence}%")
        if hallucination_flag:
            st.warning("⚠ The answer may not fully match the source documents! Possible hallucination.")

        if sources:
            with st.expander("Sources"):
                for i, doc in enumerate(sources):
                    st.write(f"Source {i + 1}: {doc.metadata}")
