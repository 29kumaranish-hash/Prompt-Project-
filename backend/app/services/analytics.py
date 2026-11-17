from sqlalchemy import select
from ..db import Paper, Chunk, Compression, Usage, Job, Report


def analytics(db, project_id, job):
    if not job:
        return {'job_id': None, 'papers': [], 'stages': [], 'baseline': None}
    papers = db.scalars(select(Paper).where(Paper.project_id == project_id)).all()
    rows = db.execute(select(Chunk, Compression).join(Compression, Compression.chunk_id == Chunk.id).where(Compression.job_id == job.id)).all()
    per_paper = []
    for p in papers:
        data = [c.data for ch,c in rows if ch.paper_id == p.id]
        original, compressed = sum(d['original_tokens'] for d in data), sum(d['compressed_tokens'] for d in data)
        per_paper.append(dict(id=p.id, title=p.title, original=original, compressed=compressed,
                              chunks=len(data), ratio=original/max(1,compressed),
                              compression_seconds=sum(d['seconds'] for d in data), extraction_seconds=p.extraction_seconds,
                              fallbacks=sum(bool(d['fallback']) for d in data)))
    usage = db.scalars(select(Usage).where(Usage.job_id == job.id)).all()
    stages = [dict(stage=stage, input_tokens=sum(u.input_tokens for u in usage if u.stage == stage),
                   output_tokens=sum(u.output_tokens for u in usage if u.stage == stage),
                   seconds=sum(u.seconds for u in usage if u.stage == stage), calls=sum(u.stage == stage for u in usage))
              for stage in sorted({u.stage for u in usage})]
    secondary = db.scalars(select(Report).where(Report.job_id == job.id, Report.kind == 'synthesis_compression')).all()
    baseline = None
    other = db.scalars(select(Job).where(Job.project_id == project_id, Job.status == 'completed').order_by(Job.created_at.desc())).all()
    for candidate in other:
        a, b = dict(candidate.config), dict(job.config)
        opposite = a.pop('baseline', False) != b.pop('baseline', False)
        if opposite and a == b:
            bu = db.scalars(select(Usage).where(Usage.job_id == candidate.id)).all()
            baseline = dict(job_id=candidate.id, is_baseline=candidate.config['baseline'], seconds=candidate.seconds,
                            input_tokens=sum(u.input_tokens for u in bu), output_tokens=sum(u.output_tokens for u in bu),
                            quality='Not automatically scored. Compare cited claims against original passages.')
            break
    return dict(job_id=job.id, baseline_mode=job.config['baseline'], papers=per_paper, stages=stages,
                tokenizer=job.config.get('compression_model'), generation_model=job.config['model'],
                seconds=job.seconds, llm_input_tokens=sum(u.input_tokens for u in usage),
                llm_output_tokens=sum(u.output_tokens for u in usage), llm_seconds=sum(u.seconds for u in usage),
                synthesis_compression=[r.data for r in secondary], baseline=baseline)
