from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_HF_EMBEDDING_MODEL = "jhgan/ko-sroberta-multitask"
DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"


def _use_local_packages() -> None:
    package_dir = Path(__file__).resolve().parent.parent / ".python_packages"
    if package_dir.exists():
        package_path = str(package_dir)
        if package_path not in sys.path:
            sys.path.insert(0, package_path)


@dataclass
class Embedder:
    provider: str
    model: str
    dimensions: int | None = None
    device: str | None = None
    _client: Any = None

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        texts = [str(text or "") for text in texts]
        if self.provider == "huggingface":
            vectors = self._client.encode(
                texts,
                batch_size=min(64, max(1, len(texts))),
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            return vectors.tolist()

        kwargs: dict[str, Any] = {"model": self.model, "input": texts}
        if self.dimensions is not None:
            kwargs["dimensions"] = self.dimensions
        response = self._client.embeddings.create(**kwargs)
        return [item.embedding for item in response.data]

    def embed_query(self, query: str) -> list[float]:
        return self.embed_texts([query])[0]


class TransformersEmbeddingClient:
    def __init__(self, model: str, device: str | None = None, token: str | None = None) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except Exception as exc:
            raise RuntimeError("transformers and torch are required for local Hugging Face embeddings.") from exc

        kwargs: dict[str, Any] = {}
        if token:
            kwargs["token"] = token
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model, **kwargs)
        self.model = AutoModel.from_pretrained(model, **kwargs).to(self.device)
        self.model.eval()

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True,
    ) -> Any:
        vectors = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with self.torch.inference_mode():
                output = self.model(**encoded)
            token_embeddings = output.last_hidden_state
            attention_mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
            pooled = (token_embeddings * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1e-9)
            if normalize_embeddings:
                pooled = self.torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.append(pooled.cpu())
        return self.torch.cat(vectors, dim=0)


def _resolve_device(device: str) -> str | None:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return None


def make_embedder(
    provider: str = "huggingface",
    model: str = DEFAULT_HF_EMBEDDING_MODEL,
    dimensions: int | None = None,
    device: str = "auto",
    openai_api_key_env: str = "OPENAI_API_KEY",
    hf_token_env: str = "HF_TOKEN",
    embedding_backend: str | None = None,
) -> Embedder:
    if provider == "huggingface":
        _use_local_packages()
        resolved_device = _resolve_device(device)
        token = os.getenv(hf_token_env)
        backend = (embedding_backend or os.getenv("RAG_HF_EMBEDDING_BACKEND") or "sentence-transformers").strip().lower()
        if backend in {"transformers", "hf-transformers", "auto-model"}:
            client = TransformersEmbeddingClient(model, device=resolved_device, token=token)
            sample = client.encode(["dimension probe"], batch_size=1)
            dims = int(sample.shape[-1])
            return Embedder(provider=provider, model=model, dimensions=dims, device=client.device, _client=client)

        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            raise RuntimeError(
                "sentence-transformers is required for Hugging Face embeddings. "
                "Install it with: python -m pip install sentence-transformers"
            ) from exc

        kwargs: dict[str, Any] = {}
        if resolved_device:
            kwargs["device"] = resolved_device
        if token:
            kwargs["token"] = token
        client = SentenceTransformer(model, **kwargs)
        dims = int(client.get_sentence_embedding_dimension() or 0) or None
        return Embedder(provider=provider, model=model, dimensions=dims, device=resolved_device, _client=client)

    if provider == "openai":
        try:
            from openai import OpenAI
        except Exception as exc:
            raise RuntimeError("openai package is required for OpenAI embeddings.") from exc
        api_key = os.getenv(openai_api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing {openai_api_key_env}. Put it in .env or export it.")
        client = OpenAI(api_key=api_key)
        return Embedder(provider=provider, model=model, dimensions=dimensions, device=None, _client=client)

    raise ValueError(f"Unsupported embedding provider: {provider}")
