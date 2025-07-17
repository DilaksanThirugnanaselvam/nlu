import streamlit as st
import pandas as pd
import torch
from transformers import pipeline, BitsAndBytesConfig
from unsloth import FastLanguageModel
import io

# === Force Unsloth to be imported first ===
import unsloth

# === Suppress Torch Dynamo Warnings ===
torch._dynamo.config.suppress_errors = True

# === Load Fine-Tuned Model ===
@st.cache_resource
def load_model():
    model_path = "Dilaksan/NLA"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_path,
        max_seq_length=8192,
        dtype=None,
        quantization_config=bnb_config,
        device_map="auto",
    )
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

pipe = load_model()

# === LaTeX-Aware Prompt ===
SYSTEM_PROMPT = r"""
You are an expert in numerical linear algebra with deep knowledge.

Respond to each user question with a direct, well-structured answer using LaTeX for all mathematical notation.

Use LaTeX to represent:
- matrices (e.g., $A$),
- vectors (e.g., $\vec{x}$),
- norms (e.g., $\lVert x \rVert_2$),
- operators (e.g., $A^T A$, $\kappa(A)$),
- and complexity notations (e.g., $\mathcal{O}(n^3)$).

Do not include explanations about what you're doing.
Avoid meta-comments, apologies, or summaries.

Only provide the clean final answer using appropriate LaTeX syntax.
"""

# === Streamlit App UI ===
st.set_page_config(page_title="Numerical Linear Algebra Q&A", layout="wide")
st.title("📘 Numerical Linear Algebra Answer Generator")
st.write("Upload a CSV file with `id` and `questions` columns. The model will generate LaTeX-formatted answers.")

uploaded_file = st.file_uploader("📤 Upload your question CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if "id" not in df.columns or "questions" not in df.columns:
        st.error("❌ The CSV must have `id` and `questions` columns.")
    else:
        st.success(f"✅ {len(df)} questions loaded.")

        # Button to start processing
        if st.button("🚀 Generate Answers"):
            output_data = []
            progress_bar = st.progress(0, text="Starting...")

            for idx, row in df.iterrows():
                question_id = row["id"]
                question = row["questions"]
                prompt = SYSTEM_PROMPT.strip() + "\n\nUser Question: " + question.strip()

                try:
                    response = pipe(prompt, max_new_tokens=2048)
                    generated_text = response[0]["generated_text"]
                    ai_answer = generated_text.replace(prompt, "").strip()
                except Exception as e:
                    ai_answer = "Error"

                output_data.append({"id": question_id, "answer": ai_answer})

                # Update progress
                progress_percent = int((idx + 1) / len(df) * 100)
                progress_bar.progress(progress_percent, text=f"Processing {idx+1}/{len(df)}")

            progress_bar.empty()
            st.success("🎉 All answers generated!")

            result_df = pd.DataFrame(output_data)
            st.dataframe(result_df)

            # Download option
            csv = result_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Answer CSV", data=csv, file_name="answers.csv", mime="text/csv")
