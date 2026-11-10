"""Token-aware overlapping evidence windows."""
try:
    from .config import MODEL_NAME, WINDOW_SIZE, WINDOW_OVERLAP
except ImportError:
    from config import MODEL_NAME, WINDOW_SIZE, WINDOW_OVERLAP


def load_tokenizer(model_name=MODEL_NAME):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_name)


def split_into_windows(text, tokenizer, window_size=WINDOW_SIZE, overlap=WINDOW_OVERLAP):
    text = (text or "").strip()
    if not text:
        return []
    if window_size <= 0 or overlap < 0 or overlap >= window_size:
        raise ValueError("Require window_size > overlap >= 0")

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= window_size:
        return [{
            "text": text,
            "window_index": 0,
            "start_token": 0,
            "end_token": len(token_ids),
        }]

    step = window_size - overlap
    windows = []
    for start in range(0, len(token_ids), step):
        end = min(start + window_size, len(token_ids))
        ids = token_ids[start:end]
        window_text = tokenizer.decode(ids, skip_special_tokens=True).strip()
        if window_text:
            windows.append({
                "text": window_text,
                "window_index": len(windows),
                "start_token": start,
                "end_token": end,
            })
        if end >= len(token_ids):
            break
    return windows
