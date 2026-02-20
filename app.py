import os
import gradio as gr
import pandas as pd
import numpy as np
from prophet import Prophet
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
import faiss

from groq import Groq

# ================= ENV =================
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env")

# ================= LOAD DATA =================
df = pd.read_csv("inflation.csv")
df.columns = ["ds", "y"]
df["ds"] = pd.to_datetime(df["ds"], format="%Y")

# ================= PROPHET =================
model = Prophet()
model.fit(df)

future = model.make_future_dataframe(periods=5, freq="YE")
forecast = model.predict(future)
future_inflation = forecast[["ds", "yhat"]].tail(5)

# ================= KNOWLEDGE BASE =================
with open("knowledge_base/inflation_causes.txt", "r", encoding="utf-8") as f:
    documents = f.read().split("\n")

embedder = SentenceTransformer("all-MiniLM-L6-v2")

doc_embeddings = embedder.encode(documents, convert_to_numpy=True)

dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings)

# ================= LLM =================
client = Groq(api_key=API_KEY)

def retrieve_context(query, k=3):
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k)
    return "\n".join([documents[i] for i in indices[0]])

def chatbot(query):
    context = retrieve_context(query)

    forecast_text = "\n".join(
        [f"{row.ds.year}: {row.yhat:.2f}%" for _, row in future_inflation.iterrows()]
    )

    prompt = f"""
You are an economic advisor AI.

Context from knowledge base:
{context}

Future Inflation Forecast:
{forecast_text}

User Question:
{query}

Answer in Roman Urdu + simple English:
- Inflation root causes
- Government role
- Citizen role
- Practical advice
"""

    response = client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

# ================= UI =================
gr.Interface(
    fn=chatbot,
    inputs=gr.Textbox(label="Ask about inflation"),
    outputs="text",
    title="Inflation Predictor & Advisor (RAG – Python 3.14 Compatible)"
).launch()
