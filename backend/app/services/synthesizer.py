from .summarizer import hierarchical_summary, flatten_reports
from .evidence import UnsupportedComparison

SYNTHESIS_INSTRUCTION = ('Integrate the papers, do not concatenate them. Sections: Research overview, Major findings, '
                         'Agreements, Contradictions, Methodology comparison, Research gaps, Limitations, '
                         'Future research directions, Overall conclusion. One concise claim per section. '
                         'Agreements and Contradictions need evidence from at least two different papers. '
                         'Check citation_sources for paper identity. Set Contradictions claims=[] when no directly conflicting findings exist. '
                         'Different datasets or conditions are not automatically contradictions. Label novel '
                         'suggestions analysis. Empty claims if unsupported. Preserve original evidence_ids.')


def synthesize(client, reports, sources, compress=None):
    notes = sorted(flatten_reports(reports), key=lambda n: n['theme'])
    if compress:
        # Only prose is lossy. Citation IDs, themes, and immutable sources remain external.
        notes = [dict(n, text=compress(n['text'])) for n in notes]
    # Focus each request on a report section. A single small-model response for
    # all nine sections tended to put unrelated findings under comparisons.
    # Select themes, not a prefix of the notes, retaining every paper's matches.
    themes = {
        'Research overview': ('overview', 'objective'),
        'Major findings': ('finding', 'takeaway'),
        'Agreements': ('finding', 'contribution'),
        'Contradictions': ('finding', 'limitation'),
        'Methodology comparison': ('method',),
        'Research gaps': ('limitation', 'future'),
        'Limitations': ('limitation',),
        'Future research directions': ('future', 'limitation'),
        'Overall conclusion': ('takeaway', 'contribution'),
    }
    sections = []
    for heading, keywords in themes.items():
        selected = [n for n in notes if any(k in n['theme'].lower() for k in keywords)]
        instruction = (f'Write only the "{heading}" section of an integrated multi-paper report. '
                       'One concise claim; use evidence from the supplied papers and retain evidence_ids. '
                       'Use citation_sources to distinguish papers. Do not treat different methods as conflicting results. '
                       'For Agreements or Contradictions, cite at least two papers only if they explicitly support the comparison; '
                       'otherwise return claims=[]. Do not add unrelated citations. Label novel suggestions analysis.')
        if not selected:
            sections.append({'heading': heading, 'claims': []})
            continue
        try:
            report = hierarchical_summary(client, selected, instruction, sources, 'synthesis:' + heading, cross=True)
        except (UnsupportedComparison, ValueError) as exc:
            if heading not in ('Agreements', 'Contradictions') or not (isinstance(exc, UnsupportedComparison) or 'requires evidence from multiple papers' in str(exc)):
                raise
            sections.append({'heading': heading, 'claims': [], 'notice':
                'No validated comparison produced. Three model attempts lacked evidence from multiple papers and were withheld. Review the source papers; this does not establish that no comparison exists.'})
            continue
        sections.extend(report['sections'])
    return {'sections': sections}
