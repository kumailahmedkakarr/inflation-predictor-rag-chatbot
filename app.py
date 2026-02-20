import gradio as gr
from groq import Groq
import os

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


with gr.Blocks(
    theme=gr.themes.Base(),
    css="""
    body {
        background-color: #0f172a;
        color: white;
    }
    .gradio-container {
        background-color: #0f172a;
    }
    textarea, input {
        background-color: #1e293b !important;
        color: white !important;
    }
"""
) as demo:

    gr.Markdown("# 🌍 EcoMind AI")
    gr.Markdown("### Inflation Predictor & Economic Advisor (RAG Model)")
    gr.Markdown("Ask about inflation causes, policies, or future trends.")

    gr.Interface(
        fn=chatbot,
        inputs=gr.Textbox(label="Ask about Inflation"),
        outputs="text"
    )

demo.launch()
