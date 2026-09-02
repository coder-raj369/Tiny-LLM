"""Text generation service used by the API and command-line clients."""

import time
from dataclasses import dataclass
from typing import Optional

import torch

from tiny_llm.inference.model_loader import ModelLoader


class GenerationError(ValueError):
    """Raised when a prompt cannot be generated safely."""


@dataclass(frozen=True)
class GenerationResult:
    """Generated text and request-level inference metadata."""

    output: str
    prompt_tokens: int
    tokens_generated: int
    latency_ms: float


class Generator:
    """Validate prompts and generate text from a loaded model."""

    def __init__(self, loader: ModelLoader) -> None:
        self.loader = loader

    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 40,
    ) -> GenerationResult:
        if not prompt or not prompt.strip():
            raise GenerationError("prompt must not be empty")
        if max_tokens < 1:
            raise GenerationError("max_tokens must be at least 1")
        if temperature <= 0:
            raise GenerationError("temperature must be greater than 0")
        model, tokenizer = self.loader.load()
        input_ids = tokenizer.encode(prompt, add_special_tokens=True)
        if len(input_ids) > model.context_length:
            raise GenerationError(
                f"prompt is too long: {len(input_ids)} tokens, maximum is {model.context_length}"
            )

        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.loader.device)
        start = time.perf_counter()
        with torch.inference_mode():
            generated_ids = model.generate(
                input_tensor,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                eos_token_id=tokenizer.special_tokens["<eos>"],
            )
        latency_ms = (time.perf_counter() - start) * 1000
        output = tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=True)
        return GenerationResult(
            output=output,
            prompt_tokens=len(input_ids),
            tokens_generated=max(0, generated_ids.shape[1] - len(input_ids)),
            latency_ms=latency_ms,
        )
