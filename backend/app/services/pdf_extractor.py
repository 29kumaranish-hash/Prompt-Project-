import re
from pathlib import Path
from collections import Counter
import pymupdf
from ..config import settings


SECTIONS = ('Abstract', 'Introduction', 'Related Work', 'Related Works', 'Background', 'Methodology', 'Methods', 'Materials and Methods', 'Experimental Setup', 'Experiment', 'Experiments', 'Results', 'Discussion', 'Limitations', 'Conclusion', 'Conclusions', 'Future Work', 'References', 'Bibliography')


def section_heading(text):
    stripped = text.strip()
    clean = re.sub(r'^\s*(?:\d+(?:\.\d+)*\.?|[IVX]+\.?|[A-Z]\.?)\s+', '', stripped)
    exact = next((s for s in SECTIONS if clean.casefold().rstrip(':') == s.casefold()), None)
    if exact:
        return exact
    for heading in SECTIONS:
        if re.match(re.escape(heading) + r'\s+(?:and|&|of|for|to)\s+', clean, re.I) and len(clean) < 100:
            return heading
    # Preserve nonstandard numbered scientific headings rather than treating all
    # following pages as the abstract. Avoid prose/sentences and numeric table rows.
    if clean != stripped and 1 <= len(clean.split()) <= 12 and len(clean) < 100:
        words = clean.split()
        if len(clean) >= 4 and clean[0].isupper() and not re.search(r'[,\u0040]|[.!?=:]$', clean) and sum(w[0].isupper() for w in words) >= len(words) / 2:
            return clean
    return None


def extract_pdf(path):
    try:
        # Open bounded upload bytes so a failed native PDF parse cannot retain a
        # Windows file handle and prevent cleanup of a corrupt upload.
        doc = pymupdf.open(stream=Path(path).read_bytes(), filetype='pdf')
    except Exception as exc:
        raise ValueError('Corrupted or unsupported PDF.') from exc
    with doc:
        if doc.needs_pass:
            raise ValueError('Password-protected PDFs are not supported. Upload an unlocked copy.')
        if not doc.page_count or doc.page_count > settings.max_pages:
            raise ValueError(f'PDF must contain 1–{settings.max_pages} pages.')
        raw, margins = [], Counter()
        for page in doc:
            blocks = [b for b in page.get_text('blocks') if b[6] == 0 and b[4].strip()]
            # Detect separated columns, retaining full-width headings at their vertical positions.
            mid = page.rect.width / 2
            left = [b for b in blocks if b[2] < mid + 15]
            right = [b for b in blocks if b[0] > mid - 15]
            if len(left) >= 3 and len(right) >= 3:
                wide = sorted([b for b in blocks if b not in left and b not in right], key=lambda b: b[1])
                ordered, remaining = [], left + right
                for band in wide:
                    before = [b for b in remaining if b[1] < band[1]]
                    ordered += sorted(before, key=lambda b: (b[0] > mid - 15, b[1])) + [band]
                    remaining = [b for b in remaining if b not in before]
                blocks = ordered + sorted(remaining, key=lambda b: (b[0] > mid - 15, b[1]))
            else:
                blocks.sort(key=lambda b: (b[1], b[0]))
            raw.append((page.rect.height, blocks))
            margins.update(set(re.sub(r'\d+', '#', b[4].strip()) for b in blocks if b[1] < 45 or b[3] > page.rect.height - 45))
        pages, warnings, section = [], [], 'Unspecified'
        headings, abstract = [], []
        for n, (height, blocks) in enumerate(raw, 1):
            paragraphs = []
            for b in blocks:
                text = b[4].strip()
                key = re.sub(r'\d+', '#', text)
                if len(raw) >= 3 and (b[1] < 45 or b[3] > height - 45) and margins[key] >= max(3, len(raw) * .5):
                    continue
                for part in re.split(r'\n\s*\n', text):
                    lines = part.splitlines()
                    # Many PDF engines put the section number on its own line.
                    if 2 <= len(lines) <= 4 and re.fullmatch(r'\s*(?:\d+(?:\.\d+)*\.?|[A-Z])\s*', lines[0]):
                        joined = ' '.join(line.strip() for line in lines)
                        if section_heading(joined):
                            lines = [joined]
                    for i, line in enumerate(lines):
                        detected = section_heading(line)
                        if n == 1 and section == 'Unspecified' and detected not in SECTIONS:
                            detected = None  # Preserve title/author/affiliation front matter.
                        if detected:
                            section = detected
                            headings.append({'heading': section, 'page': n})
                        elif line.strip():
                            # Join contiguous body lines, flushing at headings.
                            if paragraphs and paragraphs[-1]['section'] == section and i > 0 and not section_heading(lines[i-1]):
                                paragraphs[-1]['text'] += ' ' + line.strip()
                            else:
                                paragraphs.append({'section': section, 'text': line.strip()})
            text = '\n\n'.join(p['text'] for p in paragraphs)
            if len(text.strip()) < 60:
                warnings.append(f'Page {n} has little extractable text; scans, figures or equations may require OCR/manual review.')
            abstract.extend(p['text'] for p in paragraphs if p['section'] == 'Abstract')
            pages.append({'number': n, 'text': text, 'blocks': paragraphs})
        full = '\n'.join(p['text'] for p in pages)
        if len(full.strip()) < 80:
            raise ValueError('No meaningful text extracted. This PDF may be scanned; run local OCR first.')
        dois = re.findall(r'\b10\.\d{4,9}/[^\s<>]+', full)
        # Prefer the complete duplicate when a DOI was wrapped across lines.
        doi = max((value for value in dois if value.startswith(dois[0].rstrip('.,;)'))), key=len).rstrip('.,;)') if dois else None
        # Publication year is intentionally unknown rather than inferred from bibliography dates.
        meta = {'title': doc.metadata.get('title') or None, 'authors': doc.metadata.get('author') or None,
                'year': None, 'doi': doi,
                'page_count': len(pages), 'abstract': ' '.join(abstract) or None, 'headings': headings}
        return meta, pages, warnings
