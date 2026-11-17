import re
import threading
import time
from functools import lru_cache
from types import SimpleNamespace
from ..config import settings
from .chunker import token_count


def numbers(text):
    # Extractive token joining may insert spaces before percent signs. Treat
    # those typographic variants equally, without normalizing numeric values.
    text = re.sub(r'(?<=\d)\s+%', '%', text)
    return set(re.findall(r'(?<!\w)[+-]?\d+(?:[.,]\d+)*(?:%|e[+-]?\d+)?', text, re.I))


class CompressionService:
    def __init__(self):
        import torch
        from llmlingua import PromptCompressor
        device = settings.compression_device
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = PromptCompressor(model_name=settings.compression_model, device_map=device,
            use_llmlingua2=True, model_config={'trust_remote_code': False}, llmlingua2_config={'max_batch_size': 1})
        self.tokenizer = self.model.tokenizer
        self.limit = min(512, self.model.model.config.max_position_embeddings)
        self.lock = threading.Lock()

    def compress(self, text, rate, protected=()):
        start = time.perf_counter()
        original = token_count(self.tokenizer, text)
        force = ['\n', '.', '%', '±', 'not', 'no', 'without', 'limited'] + [p for p in protected if p and p in text]
        fallback, output = None, text
        with self.lock:
            for attempt, retention in enumerate((rate, min(1., max(.8, rate)))):
                result = self.model.compress_prompt(text, rate=retention, force_tokens=force[:90], force_reserve_digit=True)
                candidate = result['compressed_prompt']
                count = token_count(self.tokenizer, candidate)
                missing = numbers(text) - numbers(candidate)
                lost_terms = any(term in text and term not in candidate for term in protected)
                if candidate.strip() and not missing and not lost_terms and count >= original * max(.1, retention * .45):
                    output = candidate
                    if attempt:
                        fallback = 'Retried at higher retention after preservation check failed.'
                    break
            else:
                fallback = 'Original source used: compression failed numeric, protected-term, or token-loss safeguards.'
        compressed = token_count(self.tokenizer, output)
        return dict(compressed_text=output, original_tokens=original, compressed_tokens=compressed,
                    ratio=original / max(1, compressed), reduction_percent=(original-compressed)/max(1,original)*100,
                    seconds=time.perf_counter()-start, tokenizer=settings.compression_model,
                    requested_retention=rate, fallback=fallback)


@lru_cache(maxsize=1)
def get_compressor():
    return CompressionService()


@lru_cache(maxsize=1)
def get_tokenizer_only():
    """Baseline chunking needs the same tokenizer, not classifier weights/inference."""
    from transformers import AutoTokenizer, AutoConfig
    tokenizer = AutoTokenizer.from_pretrained(settings.compression_model, trust_remote_code=False)
    tokenizer.add_special_tokens({'additional_special_tokens':[f'[NEW{i}]' for i in range(100)]})
    config = AutoConfig.from_pretrained(settings.compression_model, trust_remote_code=False)
    return SimpleNamespace(tokenizer=tokenizer, limit=min(512, config.max_position_embeddings))
