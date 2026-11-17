from .compressor import numbers


class UnsupportedComparison(ValueError):
    """A purported cross-paper claim cites fewer than two source papers."""


def validate_evidence(summary, sources, require_cross_paper=False):
    """Reject unknown citations and numerical values absent from cited ORIGINAL passages.

    This is a structural/numeric check, not a proof of semantic entailment.
    """
    for section in summary.sections:
        for claim in section.claims:
            if any(ref not in sources for ref in claim.evidence_ids):
                raise ValueError('Model returned an unknown evidence reference. Copy evidence_ids exactly from input notes.')
            if require_cross_paper and section.heading.lower() in ('agreements', 'contradictions'):
                if len({sources[ref]['paper_id'] for ref in claim.evidence_ids}) < 2:
                    raise UnsupportedComparison(f'{section.heading}: requires evidence from multiple papers. Use claims=[] unless two papers explicitly support this comparison; do not add unrelated citations.')
            passages = '\n'.join(sources[ref]['text'] for ref in claim.evidence_ids)
            missing = numbers(claim.text) - numbers(passages)
            if missing:
                raise ValueError(f'Model introduced a numerical claim absent from its cited source: {sorted(missing)}. Remove unsupported claims, not the citation check.')
    return summary


def source_record(chunk):
    return {'chunk_id': chunk.id, 'paper_id': chunk.paper_id, 'section': chunk.section,
            'page_start': chunk.page_start, 'page_end': chunk.page_end, 'text': chunk.original_text}
