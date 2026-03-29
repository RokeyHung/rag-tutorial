from __future__ import annotations

import functools
import importlib.util
import os
import sys
from typing import Literal

import torch
from langchain_huggingface import HuggingFacePipeline
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


DevicePref = Literal["auto", "cpu", "cuda"]
DTypePref = Literal["auto", "fp16", "bf16", "fp32"]


def _parse_device_pref(raw: str | None) -> DevicePref:
    v = (raw or "auto").strip().lower()
    if v in ("auto", "cpu", "cuda"):
        return v  # type: ignore[return-value]
    return "auto"


def _parse_dtype_pref(raw: str | None) -> DTypePref:
    v = (raw or "auto").strip().lower()
    if v in ("auto", "fp16", "bf16", "fp32"):
        return v  # type: ignore[return-value]
    return "auto"


def _resolve_dtype(*, device: str, dtype_pref: DTypePref) -> torch.dtype:
    if device != "cuda":
        return torch.float32

    if dtype_pref == "fp32":
        return torch.float32
    if dtype_pref == "bf16":
        return torch.bfloat16
    if dtype_pref == "fp16":
        return torch.float16

    # auto: ưu tiên bf16 nếu GPU hỗ trợ, fallback fp16
    try:
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
    except Exception:
        pass
    return torch.float16


def _version_tuple(v: str) -> tuple[int, ...]:
    out: list[int] = []
    for part in (v or "").split("."):
        digits = "".join(ch for ch in part if ch.isdigit())
        if digits == "":
            break
        out.append(int(digits))
    return tuple(out)


def _ensure_utf8_stdio() -> None:
    # Tránh crash khi console Windows không phải UTF-8 (emoji/tiếng Việt).
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


@functools.lru_cache(maxsize=4)
def _build_hf_pipeline_cached(
    *,
    model_name: str,
    hf_token: str | None,
    device_pref: DevicePref,
    dtype_pref: DTypePref,
    temperature: float,
    max_new_tokens: int,
    top_p: float,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
):
    cuda_available = torch.cuda.is_available()
    torch_cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
    cuda_built = torch_cuda_version is not None

    if device_pref == "cpu":
        device = "cpu"
    elif device_pref == "cuda":
        device = "cuda" if cuda_available else "cpu"
    else:
        device = "cuda" if cuda_available else "cpu"

    if device_pref == "cuda" and device != "cuda":
        print(
            "⚠️ LLM_DEVICE=cuda nhưng torch.cuda.is_available()=False — fallback CPU. "
            "Thường do đang chạy nhầm Python/venv hoặc torch CPU-only."
        )

    dtype = _resolve_dtype(device=device, dtype_pref=dtype_pref)

    # Tăng tốc matmul trên CUDA (nếu user bật).
    tf32 = (os.getenv("LLM_TF32", "") or "").strip().lower() in {"1", "true", "yes", "y", "on"}
    if device == "cuda" and tf32:
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

    # Chỉ dùng device_map khi accelerate có sẵn (tránh hard dependency).
    accelerate_ok = importlib.util.find_spec("accelerate") is not None
    using_device_map = False

    model_kwargs: dict = {"token": hf_token}
    # transformers>=5: ưu tiên "dtype"; transformers<5: dùng "torch_dtype"
    if _version_tuple(getattr(transformers, "__version__", "0")) >= (5, 0, 0):
        model_kwargs["dtype"] = dtype
    else:
        model_kwargs["torch_dtype"] = dtype

    if accelerate_ok and device == "cuda" and device_pref in ("auto", "cuda"):
        model_kwargs["device_map"] = "auto"
        using_device_map = True

    # Ưu tiên load từ local cache trước; nếu không có thì tự tải.
    model = None
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            local_files_only=True,
            **model_kwargs,
        )
    except Exception:
        model = None
    if model is None:
        print(f"⬇️ Không tìm thấy model local, đang tải: {model_name}")
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if not using_device_map:
        model = model.to(device)
    model.eval()

    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=hf_token,
            use_fast=True,
            local_files_only=True,
        )
    except Exception:
        tokenizer = None
    if tokenizer is None:
        print(f"⬇️ Không tìm thấy tokenizer local, đang tải: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, use_fast=True)
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # Đặt generation config trực tiếp trên model để:
    # - áp dụng mặc định cho mọi lần gọi (LangChain wrapper không truyền pipeline_kwargs khi invoke)
    # - tăng chất lượng (giảm lặp) và tránh bị cắt do max_new_tokens mặc định thấp.
    try:
        gen = model.generation_config
        gen.do_sample = True
        gen.temperature = float(temperature)
        gen.top_p = float(top_p)
        gen.max_new_tokens = int(max_new_tokens)
        if hasattr(gen, "repetition_penalty"):
            gen.repetition_penalty = float(repetition_penalty)
        if hasattr(gen, "no_repeat_ngram_size"):
            gen.no_repeat_ngram_size = int(no_repeat_ngram_size)
        if getattr(gen, "pad_token_id", None) is None and getattr(tokenizer, "pad_token_id", None) is not None:
            gen.pad_token_id = tokenizer.pad_token_id
        if getattr(gen, "eos_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
            gen.eos_token_id = tokenizer.eos_token_id
    except Exception:
        pass

    _ensure_utf8_stdio()
    print(
        "🧠 LLM runtime:",
        f"device_pref={device_pref}",
        f"cuda_available={cuda_available}",
        f"cuda_built={cuda_built}",
        f"torch_cuda={torch_cuda_version}",
        f"torch={getattr(torch, '__version__', 'unknown')}",
        f"python={sys.executable}",
        f"using={device}",
        f"accelerate={accelerate_ok}",
        f"device_map={'auto' if using_device_map else 'none'}",
        f"dtype={str(dtype).replace('torch.', '')}",
        f"max_new_tokens={int(max_new_tokens)}",
        f"temperature={float(temperature)}",
        f"top_p={float(top_p)}",
    )

    pipe_kwargs = dict(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        # Không trả lại phần prompt, giúp parser sạch hơn
        return_full_text=False,
    )

    # Nếu model đã được shard/placed qua device_map, không set device cho pipeline (tránh xung đột).
    if not using_device_map:
        pipe_kwargs["device"] = 0 if device == "cuda" else -1

    return pipeline(**pipe_kwargs)


def get_hf_llm(
    model_name: str | None = None,
    temperature: float = 0.2,
    max_new_tokens: int = 512,
):
    model_name = model_name or os.getenv("LLM_MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct")
    hf_token = os.getenv("HF_TOKEN")

    device_pref = _parse_device_pref(os.getenv("LLM_DEVICE", "auto"))
    dtype_pref = _parse_dtype_pref(os.getenv("LLM_DTYPE", "auto"))

    # Cho phép tinh chỉnh generation qua env (không bắt buộc).
    temp_raw = os.getenv("LLM_TEMPERATURE", "")
    if temp_raw.strip() != "":
        try:
            temperature = float(temp_raw)
        except Exception:
            pass

    mnt_raw = os.getenv("LLM_MAX_NEW_TOKENS", "")
    if mnt_raw.strip() != "":
        try:
            max_new_tokens = int(float(mnt_raw))
        except Exception:
            pass

    # Cho phép override top_p qua env, nhưng vẫn giữ API hiện tại.
    top_p_raw = os.getenv("LLM_TOP_P", "")
    try:
        top_p = float(top_p_raw) if top_p_raw.strip() != "" else 0.75
    except Exception:
        top_p = 0.75

    rep_raw = os.getenv("LLM_REPETITION_PENALTY", "")
    try:
        repetition_penalty = float(rep_raw) if rep_raw.strip() != "" else 1.12
    except Exception:
        repetition_penalty = 1.12

    ngram_raw = os.getenv("LLM_NO_REPEAT_NGRAM", "")
    try:
        no_repeat_ngram_size = int(float(ngram_raw)) if ngram_raw.strip() != "" else 4
    except Exception:
        no_repeat_ngram_size = 4

    pipe = _build_hf_pipeline_cached(
        model_name=model_name,
        hf_token=hf_token,
        device_pref=device_pref,
        dtype_pref=dtype_pref,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
    )

    return HuggingFacePipeline(pipeline=pipe)