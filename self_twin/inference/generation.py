"""Chat generation helpers with explicit turn-boundary stopping.

Work with a multimodal processor (Qwen3.5) or a plain tokenizer (LFM2.5); both
use <|im_start|>/<|im_end|> turns.
"""

from __future__ import annotations

import torch
from transformers import LogitsProcessor, LogitsProcessorList, StoppingCriteria, StoppingCriteriaList


class GeneratedTokenRepetitionPenalty(LogitsProcessor):
    """Repetition penalty over generated tokens only, as WebLLM applies it.

    Transformers' built-in ``repetition_penalty`` also penalizes every prompt
    token (system prompt, memory candidates, the user's words). WebLLM counts
    only the tokens it has generated in the current reply, so a reply that
    repeats a word from a memory candidate is not penalized on the site.
    """

    def __init__(self, penalty: float, prompt_length: int) -> None:
        self.penalty = penalty
        self.prompt_length = prompt_length

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        generated = input_ids[:, self.prompt_length:]
        if generated.shape[-1] == 0:
            return scores
        score = torch.gather(scores, 1, generated)
        score = torch.where(score < 0, score * self.penalty, score / self.penalty)
        return scores.scatter(1, generated, score)


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
    tokenizer = getattr(processor, "tokenizer", processor)
    eos_token_id = tokenizer.eos_token_id
    im_start_token_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    if eos_token_id is None or im_start_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id and <|im_start|> for safe turn stopping")

    prompt_length = inputs["input_ids"].shape[-1]
    generation_kwargs = {}
    if repetition_penalty != 1.0:
        generation_kwargs["logits_processor"] = LogitsProcessorList(
            [GeneratedTokenRepetitionPenalty(repetition_penalty, prompt_length)]
        )
    if do_sample:
        # Transformers fills an unset top_k with 50; WebLLM has no top-k cutoff.
        generation_kwargs.update(temperature=temperature, top_p=top_p, top_k=0)

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

    generated = output[0, prompt_length:]
    # Do not expose even a role token if the model starts another serialized turn.
    boundary_positions = [
        index
        for index, token_id in enumerate(generated.tolist())
        if token_id in {eos_token_id, im_start_token_id}
    ]
    if boundary_positions:
        generated = generated[: boundary_positions[0]]
    return tokenizer.batch_decode(generated.unsqueeze(0), skip_special_tokens=True)[0].strip()
