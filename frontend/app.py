"""Minimal Streamlit client for the Tiny LLM API."""

import os

import requests
import streamlit as st

API_URL = os.getenv("TINY_LLM_API_URL", "http://localhost:8000/api/v1")

st.set_page_config(page_title="Quantum Q&A", page_icon="Q", layout="centered")
st.title("Quantum Computing Q&A")
st.caption("A deliberately small domain-specialized language model")

prompt = st.text_area("Question", height=140, max_chars=4000, placeholder="Ask about quantum computing...")
max_tokens = st.slider("Maximum answer tokens", min_value=16, max_value=256, value=100, step=8)
temperature = st.slider("Temperature", min_value=0.1, max_value=2.0, value=1.0, step=0.1)

if st.button("Generate answer", type="primary", disabled=not prompt.strip()):
    try:
        response = requests.post(
            f"{API_URL}/generate",
            json={"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature},
            timeout=10,
        )
        response.raise_for_status()
        result = response.json()
        st.write(result["output"])
        st.caption(
            f"{result['tokens_generated']} generated tokens | "
            f"{result['latency_ms']:.0f} ms"
        )
    except requests.RequestException as exc:
        st.error(f"The inference API is unavailable: {exc}")
