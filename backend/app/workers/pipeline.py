import logging
import time
from concurrent.futures import ThreadPoolExecutor
from sqlalchemy import select
from ..db import Session, Paper, Page, Chunk, Job, Compression, Report, Usage
from ..config import settings
from ..services.compressor import get_compressor, get_tokenizer_only
from ..services.chunker import chunk_pages, token_count
from ..services.evidence import source_record
from ..services.ollama_client import OllamaClient
from ..services.summarizer import CHUNK_INSTRUCTION, PAPER_INSTRUCTION, flatten_reports, hierarchical_summary
from ..services.synthesizer import synthesize

executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='research-worker')
logger = logging.getLogger('researchcompress')


def run_job(job_id):
    start = time.perf_counter()
    with Session() as db:
        job = db.get(Job, job_id)
        if not job:
            return
        previous_seconds = job.seconds

        def progress(stage, value):
            job.stage, job.progress, job.status = stage, value, 'running'
            job.seconds = previous_seconds + time.perf_counter() - start
            db.commit()
            logger.info('pipeline_progress', extra={'job_id': job.id, 'stage': stage, 'progress': value})

        def usage(**kwargs):
            diagnostic = kwargs.pop('diagnostic', None)
            if diagnostic:
                db.add(Report(job_id=job.id, kind='rejected_response', data={'stage':kwargs['stage'], **diagnostic}))
            db.add(Usage(job_id=job.id, **kwargs))
            db.commit()

        def existing(kind, paper_id=None, chunk_id=None):
            return db.scalar(select(Report).where(Report.job_id == job.id, Report.kind == kind,
                                                 Report.paper_id == paper_id, Report.chunk_id == chunk_id))

        def save(kind, data, paper_id=None, chunk_id=None):
            db.add(Report(job_id=job.id, kind=kind, data=data, paper_id=paper_id, chunk_id=chunk_id))
            db.commit()

        try:
            progress('Checking local models', .02)
            status = OllamaClient.status()
            if not status['available'] or job.config['model'] not in status['models']:
                raise RuntimeError(f"Start Ollama and run: ollama pull {job.config['model']}")
            progress('Loading compression tokenizer for baseline' if job.config['baseline'] else 'Loading LLMLingua-2 (first run may download weights)', .04)
            compressor = get_tokenizer_only() if job.config['baseline'] else get_compressor()
            papers = db.scalars(select(Paper).where(Paper.project_id == job.project_id).order_by(Paper.id)).all()
            for paper in papers:
                if not db.scalar(select(Chunk).where(Chunk.paper_id == paper.id)):
                    progress(f'Chunking: {paper.title}', .08)
                    pages = db.scalars(select(Page).where(Page.paper_id == paper.id).order_by(Page.number)).all()
                    chunks = chunk_pages([{'number': p.number, 'blocks': p.blocks} for p in pages], compressor.tokenizer,
                                         settings.chunk_target_tokens, compressor.limit)
                    for data in chunks:
                        db.add(Chunk(paper_id=paper.id, **data))
                    db.commit()
            chunks = db.scalars(select(Chunk).join(Paper).where(Paper.project_id == job.project_id).order_by(Chunk.paper_id, Chunk.ordinal)).all()
            sources = {c.id: source_record(c) for c in chunks}
            client = OllamaClient(job.config['model'], usage)
            for i, chunk in enumerate(chunks):
                progress(f'Compressing chunk {i+1} of {len(chunks)}', .1 + .65*i/max(1,len(chunks)))
                compression = db.scalar(select(Compression).where(Compression.job_id == job.id, Compression.chunk_id == chunk.id))
                if not compression:
                    if job.config['baseline']:
                        n = token_count(compressor.tokenizer, chunk.original_text)
                        data = dict(compressed_text=chunk.original_text, original_tokens=n, compressed_tokens=n,
                                    ratio=1, reduction_percent=0, seconds=0, tokenizer=settings.compression_model,
                                    requested_retention=1, fallback=None)
                    else:
                        data = compressor.compress(chunk.original_text, job.config['retention'], job.config['protected_terms'])
                    compression = Compression(job_id=job.id, chunk_id=chunk.id, data=data)
                    db.add(compression)
                    db.commit()
                if not existing('chunk', chunk.paper_id, chunk.id):
                    progress(f'Summarizing chunk {i+1} of {len(chunks)}', .1 + .65*i/max(1,len(chunks)))
                    data = client.generate({'chunk_id': chunk.id, 'paper_id': chunk.paper_id,
                                            'section': chunk.section, 'text': compression.data['compressed_text']},
                                           CHUNK_INSTRUCTION, {chunk.id: sources[chunk.id]}, 'chunk')
                    save('chunk', data, chunk.paper_id, chunk.id)
            reports = []
            for i, paper in enumerate(papers):
                progress(f'Generating paper summary: {paper.title}', .76 + .12*i/max(1,len(papers)))
                report = existing('paper', paper.id)
                if not report:
                    chunk_reports = db.scalars(select(Report).where(Report.job_id == job.id, Report.paper_id == paper.id, Report.kind == 'chunk')).all()
                    paper_sources = {k:v for k,v in sources.items() if v['paper_id'] == paper.id}
                    data = hierarchical_summary(client, flatten_reports([r.data for r in chunk_reports]), PAPER_INSTRUCTION,
                                                paper_sources, 'paper')
                    save('paper', data, paper.id)
                    report = existing('paper', paper.id)
                reports.append(report.data)
            if len(papers) > 1 and not existing('synthesis'):
                progress('Synthesizing across papers', .9)
                def compress_note(text):
                    cached = db.scalars(select(Report).where(Report.job_id == job.id, Report.kind == 'synthesis_compression')).all()
                    for note in cached:
                        if note.data.get('original_text') == text:
                            return note.data['compressed_text']
                    if token_count(compressor.tokenizer, text) > settings.chunk_target_tokens:
                        return text  # Never exceed encoder limit; keep oversized notes unchanged.
                    data = compressor.compress(text, .75, job.config['protected_terms'])
                    save('synthesis_compression', dict(original_text=text, **data))
                    return data['compressed_text']
                callback = compress_note if job.config['synthesis_compression'] and not job.config['baseline'] else None
                save('synthesis', synthesize(client, reports, sources, callback))
            job.status, job.stage, job.progress = 'completed', 'Completed', 1
            job.error = None
        except Exception as exc:
            db.rollback()
            job = db.get(Job, job_id)
            job.status, job.error = 'failed', f'{type(exc).__name__}: {exc}'[:3000]
            logger.exception('pipeline_failed', extra={'job_id': job_id})
        finally:
            job.seconds = previous_seconds + time.perf_counter()-start
            db.commit()
