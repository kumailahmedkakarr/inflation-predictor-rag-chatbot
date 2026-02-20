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
        background-color: #f4f6f9 !important;
    }

    .gradio-container {
        background-color: #f4f6f9 !important;
        font-family: 'Segoe UI', sans-serif;
    }

    h1 {
        color: #1f2937 !important;
        font-weight: bold;
    }

    h2, h3, label {
        color: #374151 !important;
    }

    textarea, input {
        background-color: #ffffff !important;
        color: #111827 !important;
        border-radius: 8px !important;
        border: 1px solid #d1d5db !important;
        padding: 10px !important;
    }

    button {
        background-color: #2563eb !important;
        color: white !important;
        border-radius: 8px !important;
        font-weight: bold !important;
        padding: 8px 16px !important;
    }

    button:hover {
        background-color: #1d4ed8 !important;
    }

    .output {
        background-color: #ffffff !important;
        color: #111827 !important;
        border-radius: 8px !important;
        border: 1px solid #d1d5db !important;
    }
    </style>
    """)

    gr.Markdown("# 🌍 EcoMind AI")
    gr.Markdown("### Inflation Predictor & Economic Advisor (RAG Model)")
    gr.Markdown("Ask about inflation causes, policies, or future trends.")

    with gr.Row():
        user_input = gr.Textbox(label="Ask about Inflation")

    with gr.Row():
        output_box = gr.Textbox(label="AI Response")

    submit_btn = gr.Button("Submit")

    submit_btn.click(chatbot, inputs=user_input, outputs=output_box)

demo.launch()
