import re


def token_count(tokenizer, text):
    return len(tokenizer.encode(text, add_special_tokens=False))


def chunk_pages(pages, tokenizer, target=400, encoder_limit=512):
    limit = min(target, encoder_limit - tokenizer.num_special_tokens_to_add(pair=False))
    if limit < 16:
        raise ValueError('Compression encoder context is too small.')
    chunks, current, section, start, end = [], [], None, 1, 1

    def flush():
        nonlocal current
        if current:
            text = '\n'.join(current)
            chunks.append(dict(ordinal=len(chunks), section=section, page_start=start, page_end=end,
                               original_text=text, original_tokens=token_count(tokenizer, text)))
            current = []

    for page in pages:
        for block in page['blocks']:
            if block['section'] != section:
                flush()
                section = block['section']
            for sentence in re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', block['text']):
                # Oversized sentences must split; use tokenizer offsets to retain exact source text.
                pending = [sentence]
                while pending:
                    part = pending.pop(0)
                    if token_count(tokenizer, part) > limit:
                        encoded = tokenizer(part, add_special_tokens=False, return_offsets_mapping=True)
                        cut = encoded['offset_mapping'][limit-1][1]
                        if cut <= 0 or cut >= len(part):
                            raise ValueError('Tokenizer cannot split an oversized sentence safely.')
                        pending.insert(0, part[cut:])
                        part = part[:cut]
                    if current and token_count(tokenizer, '\n'.join(current + [part])) > limit:
                        flush()
                    if not current:
                        start = page['number']
                    end = page['number']
                    current.append(part)
    flush()
    return chunks
