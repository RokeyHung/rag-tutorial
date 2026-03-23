import torch
import os
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain_huggingface import HuggingFacePipeline


def get_hf_llm(
    model_name: str | None = None,
    temperature: float = 0.2,
    max_new_tokens: int = 300,
):
    model_name = model_name or os.getenv("LLM_MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct")
    hf_token = os.getenv("HF_TOKEN")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        token=hf_token,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=0.75,
    )

    return HuggingFacePipeline(pipeline=pipe)