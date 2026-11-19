"""Qwen chat generation helpers with explicit turn-boundary stopping."""

from __future__ import annotations

import torch
from transformers import StoppingCriteria, StoppingCriteriaList


class StopAtAssistantTurnStart(StoppingCriteria):
    """Guard against continuing into a new serialized chat turn."""

    def __init__(self, token_id: int) -> None:
        self.token_id = token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return bool((input_ids[:, -1] == self.token_id).all().item())


def generate_assistant_response(
    model,
    processor,
    inputs,
    max_new_tokens: int,
    *,
    do_sample: bool = False,
    temperature: float = 0.72,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
) -> str:
    tokenizer = processor.tokenizer
    eos_token_id = tokenizer.eos_token_id
    im_start_token_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    if eos_token_id is None or im_start_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id and <|im_start|> for safe turn stopping")

    generation_kwargs = {"repetition_penalty": repetition_penalty} if repetition_penalty != 1.0 else {}
    if do_sample:
        generation_kwargs.update(temperature=temperature, top_p=top_p)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            use_cache=True,
            eos_token_id=eos_token_id,
            stopping_criteria=StoppingCriteriaList([StopAtAssistantTurnStart(im_start_token_id)]),
            **generation_kwargs,
        )

    prompt_length = inputs["input_ids"].shape[-1]
    generated = output[0, prompt_length:]
    # Do not expose even a role token if the model starts another serialized turn.
    boundary_positions = [
        index
        for index, token_id in enumerate(generated.tolist())
        if token_id in {eos_token_id, im_start_token_id}
    ]
    if boundary_positions:
        generated = generated[: boundary_positions[0]]
    return processor.batch_decode(generated.unsqueeze(0), skip_special_tokens=True)[0].strip()
