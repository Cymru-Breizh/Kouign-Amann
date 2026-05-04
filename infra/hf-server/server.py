#!/usr/bin/env python3
"""Lightweight OpenAI-compatible chat completions server for HuggingFace models.

Loads a merged model with standard transformers and exposes a
/v1/chat/completions endpoint that LiteLLM can call directly.

For repos that contain a merged model in a subfolder (e.g. Unsloth repos
with a merged "sft/" directory), set MODEL_SUBFOLDER to point at it.

Environment variables:
    MODEL_ID        – HuggingFace model ID or local path (required)
    MODEL_SUBFOLDER – subfolder within the repo to load from (optional)
    DEVICE          – device for inference (default: "cuda")
    PORT            – port to listen on (default: 8000)
"""

import os
import time
import uuid

import torch
import uvicorn

from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = os.environ.get("MODEL_ID", "techiaith/llama-3.2-1b-welsh-sft")
MODEL_SUBFOLDER = os.environ.get("MODEL_SUBFOLDER", None)
DEVICE = os.environ.get("DEVICE", "cuda")
PORT = int(os.environ.get("PORT", "8000"))

app = FastAPI(title="HF Inference Server")

# Loaded at startup
model = None
tokenizer = None


@app.on_event("startup")
def load_model():
    global model, tokenizer
    extra = {"subfolder": MODEL_SUBFOLDER} if MODEL_SUBFOLDER else {}
    print(f"Loading model: {MODEL_ID}" + (f" (subfolder: {MODEL_SUBFOLDER})" if MODEL_SUBFOLDER else ""))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **extra)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        **extra,
        dtype=torch.float16,
        device_map=DEVICE,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Model loaded on {DEVICE}")


# ---------- OpenAI-compatible request/response schemas ----------

class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = ""
    messages: list[Message]
    temperature: float = 0.0
    max_tokens: int = 500
    top_p: float = 1.0


class Choice(BaseModel):
    index: int = 0
    message: Message
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: list[Choice]
    usage: Usage = Usage()


# ---------- Endpoints ----------

@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": MODEL_ID, "object": "model"}],
    }


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest):
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    else:
        # Llama 3 chat format fallback
        parts = []
        for m in messages:
            parts.append(
                f"<|start_header_id|>{m['role']}<|end_header_id|>\n\n"
                f"{m['content']}<|eot_id|>"
            )
        parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        prompt = "<|begin_of_text|>" + "".join(parts)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[-1]

    temperature = max(request.temperature, 1e-7)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            temperature=temperature,
            top_p=request.top_p,
            do_sample=request.temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = output_ids[0][prompt_len:]
    completion_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return ChatCompletionResponse(
        model=MODEL_ID,
        choices=[
            Choice(message=Message(role="assistant", content=completion_text))
        ],
        usage=Usage(
            prompt_tokens=prompt_len,
            completion_tokens=len(new_tokens),
            total_tokens=prompt_len + len(new_tokens),
        ),
    )


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
