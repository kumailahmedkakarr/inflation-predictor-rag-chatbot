import os
import gradio as gr
import pandas as pd
from prophet import Prophet
from dotenv import load_dotenv

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter

from groq import Groq

load_dotenv()

# ========== LOAD DATA ==========
df = pd.read_csv("inflation.csv")
df.columns = ["ds", "y"]
df["ds"] = pd.to_datetime(df["ds"], format="%Y")

# ========== TRAIN MODEL ==========
model = Prophet()
model.fit(df)

future = model.make_future_dataframe(periods=5, freq="Y")
forecast = model.predict(future)
future_inflation = forecast[["ds", "yhat"]].tail(5)

# ========== KNOWLEDGE BASE ==========
with open("knowledge_base/inflation_causes.txt", "r", encoding="utf-8") as f:
    text = f.read()

splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_text(text)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = FAISS.from_texts(docs, embeddings)

# ========== LLM ==========
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def chatbot(query):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents(query)
    context = "\n".join([d.page_content for d in docs])

    inflation_info = "\n".join(
        [f"{row.ds.year}: {row.yhat:.2f}%" for _, row in future_inflation.iterrows()]
    )

    prompt = f"""
You are an economic advisor AI.

Context:
{context}

Future Inflation Forecast:
{inflation_info}

User Question:
{query}

Answer in Roman Urdu + simple English:
- Root causes
- Government role
- Citizen role
- Practical advice
"""

    response = client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

# ========== UI ==========
gr.Interface(
    fn=chatbot,
    inputs=gr.Textbox(label="Ask about inflation"),
    outputs="text",
    title="Inflation Predictor & Advisor (RAG-Based AI)"
).launch()
