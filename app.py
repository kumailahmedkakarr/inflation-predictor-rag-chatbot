import gradio as gr
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

from utils.rag_engine import retrieve_context
from utils.forecast_engine import generate_forecast
from utils.prompt_template import build_prompt

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
def chatbot(query):
    context = retrieve_context(query)
    forecast_text = generate_forecast()

    prompt = build_prompt(query, context, forecast_text)

    response = client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )

    return response.choices[0].message.content


with gr.Blocks() as demo:

    gr.HTML("""
    <style>
    body {
        background-color: #111827 !important;
    }

    .gradio-container {
        background-color: #111827 !important;
        color: #ffffff !important;
    }

    textarea, input {
        background-color: #1f2937 !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        border: 1px solid #374151 !important;
    }

    button {
        background-color: #2563eb !important;
        color: white !important;
        border-radius: 8px !important;
        font-weight: bold !important;
    }

    button:hover {
        background-color: #1d4ed8 !important;
    }

    .output {
        background-color: #1f2937 !important;
        color: #ffffff !important;
        border-radius: 8px !important;
    }

    h1, h2, h3, label {
        color: #f9fafb !important;
    }
    </style>
    """)

    gr.Markdown("# 🌍 EcoMind AI")
    gr.Markdown("### Inflation Predictor & Economic Advisor (RAG Model)")
    gr.Markdown("Ask about inflation causes, policies, or future trends.")

    gr.Interface(
        fn=chatbot,
        inputs=gr.Textbox(label="Ask about Inflation"),
        outputs=gr.Textbox(label="AI Response")
    )

demo.launch()
