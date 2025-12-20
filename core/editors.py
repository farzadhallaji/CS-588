from __future__ import annotations

from typing import Dict, List

try:
    import requests
except ImportError:  # pragma: no cover - optional
    requests = None

try:
    from transformers import pipeline
except Exception:  # pragma: no cover - optional
    pipeline = None  # type: ignore

from .utils import sentence_split


class BaseEditor:
    def propose(
        self,
        current_review: str,
        uncovered: List[str],
        offending: List[str],
        evidence: Dict[str, List[str]],
        prompt: str,
        num_samples: int,
    ) -> List[str]:
        raise NotImplementedError


class TemplateEditor(BaseEditor):
    """
    Deterministic, offline editor that minimally edits by
    dropping noisy sentences and appending focused notes per uncovered ref.
    """

    def propose(
        self,
        current_review: str,
        uncovered: List[str],
        offending: List[str],
        evidence: Dict[str, List[str]],
        prompt: str,
        num_samples: int,
    ) -> List[str]:
        cleaned = [s for s in sentence_split(current_review) if s.strip() not in offending]
        additions = []
        for ref in uncovered:
            ev = evidence.get(ref, [])
            ev_str = f" (Evidence: {ev[0]})" if ev else ""
            additions.append(f"{ref}{ev_str}".strip())
        base = " ".join(cleaned).strip()
        combined = (base + " " + " ".join(additions)).strip()
        if not combined:
            combined = current_review.strip()
        return [combined for _ in range(num_samples)]


class OllamaEditor(BaseEditor):
    def __init__(self, model: str = "llama3:8b-instruct-q4_0", temperature: float = 0.2):
        if requests is None:
            raise ImportError("requests is required for OllamaEditor")
        self.model = model
        self.temperature = temperature
        self.system = (
            "You are a senior code reviewer. Improve the review so it is concise, specific, and supported by the diff/evidence. "
            "Make minimal edits unless rewrite is requested. Output only the revised review text."
        )

    def propose(
        self,
        current_review: str,
        uncovered: List[str],
        offending: List[str],
        evidence: Dict[str, List[str]],
        prompt: str,
        num_samples: int,
    ) -> List[str]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system},
                {"role": "user", "content": prompt},
            ],
            "options": {"temperature": self.temperature, "num_predict": 512, "stop": ["</s>"]},
            "stream": False,
        }
        outputs = []
        for _ in range(num_samples):
            resp = requests.post("http://127.0.0.1:11434/api/chat", json=payload, timeout=120)
            resp.raise_for_status()
            outputs.append(resp.json()["message"]["content"].strip())
        return outputs


class EchoEditor(BaseEditor):
    """No-op editor for ablations."""

    def propose(
        self,
        current_review: str,
        uncovered: List[str],
        offending: List[str],
        evidence: Dict[str, List[str]],
        prompt: str,
        num_samples: int,
    ) -> List[str]:
        return [current_review for _ in range(num_samples)]


class HFLocalEditor(BaseEditor):
    """
    HuggingFace text-generation pipeline editor for local LLMs.
    """

    def __init__(
        self,
        model_path: str,
        max_new_tokens: int = 128,
        temperature: float = 0.6,
        top_p: float = 0.9,
        device: str = "cpu",
    ):
        if pipeline is None:
            raise ImportError("transformers is required for HFLocalEditor")
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.device = device
        self.generator = pipeline(
            "text-generation",
            model=model_path,
            tokenizer=model_path,
            device_map=device if device != "cpu" else None,
            return_full_text=False,
        )

    def propose(
        self,
        current_review: str,
        uncovered: List[str],
        offending: List[str],
        evidence: Dict[str, List[str]],
        prompt: str,
        num_samples: int,
    ) -> List[str]:
        gen_prompt = (
            "You are refining a code review. Improve it using the uncovered items and evidence. "
            "Output only the revised review.\n\n"
            f"{prompt}"
        )
        outputs = self.generator(
            gen_prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            num_return_sequences=num_samples,
            pad_token_id=self.generator.tokenizer.eos_token_id,
        )
        cleaned = []
        for out in outputs:
            text = out.get("generated_text", "").strip()
            if "\n\n" in text:
                text = text.split("\n\n")[-1].strip()
            cleaned.append(text or current_review)
        return cleaned
