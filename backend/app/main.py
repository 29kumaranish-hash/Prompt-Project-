import csv
import hashlib
import io
import json
import logging
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, FileResponse
from sqlalchemy import select, delete, func
from .config import settings
from .db import Base, engine, Session, Project, Paper, Page, Chunk, Job, Report, Compression, serialize
from .schemas import (ProjectCreate, Analyze, MetadataEdit, ProjectResponse, ProjectListItem,
                      ProjectDetail, PaperResponse, JobResponse, UploadResponse, ReportResponse)
from .services.pdf_extractor import extract_pdf
from .services.ollama_client import OllamaClient
from .services.compressor import get_compressor
from .services.evidence import source_record
from .services.analytics import analytics
from .workers.pipeline import executor, run_job


class JsonFormatter(logging.Formatter):
    def format(self, record):
        data = {'level': record.levelname, 'message': record.getMessage()}
        for key in ('job_id', 'stage', 'progress'):
            if hasattr(record, key):
                data[key] = getattr(record, key)
        if record.exc_info:
            data['exception'] = self.formatException(record.exc_info)
        return json.dumps(data)


handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logging.getLogger('researchcompress').addHandler(handler)
logging.getLogger('researchcompress').setLevel(logging.INFO)
mutation_lock = threading.RLock()


@asynccontextmanager
async def lifespan(app):
    Base.metadata.create_all(engine)
    with Session() as db:
        for job in db.scalars(select(Job).where(Job.status.in_(['queued', 'running']))):
            job.status, job.error = 'failed', 'Server interrupted. Retry to resume saved checkpoints.'
        db.commit()
    yield


app = FastAPI(title='ResearchCompress AI', version='1.0.0', lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=['GET', 'POST', 'PATCH', 'DELETE'], allow_headers=['Content-Type'])


def require(db, cls, key):
    obj = db.get(cls, key)
    if not obj:
        raise HTTPException(404, f'{cls.__name__} not found')
    return obj


def ensure_idle(db, project_id):
    if db.scalar(select(Job).where(Job.project_id == project_id, Job.status.in_(['queued', 'running']))):
        raise HTTPException(409, 'Wait for the active analysis to finish before changing this project.')


def latest_job(db, project_id, job_id=None, completed=False):
    query = select(Job).where(Job.project_id == project_id)
    if job_id:
        query = query.where(Job.id == job_id)
    if completed:
        query = query.where(Job.status == 'completed')
    job = db.scalar(query.order_by(Job.created_at.desc()))
    if job_id and not job:
        raise HTTPException(404, 'Job not found in this project')
    return job


def invalidate(db, project_id):
    db.execute(delete(Job).where(Job.project_id == project_id))


@app.get('/api/health')
def health():
    with Session() as db:
        db.execute(select(1))
    return {'status': 'ok', 'local_only': True, 'max_upload_size_mb': settings.max_upload_size_mb,
            'baseline_enabled': settings.enable_baseline_comparison}


@app.get('/api/models/status')
def models():
    return dict(ollama=OllamaClient.status(), default_model=settings.ollama_model,
                compression_model=settings.compression_model, compressor_loaded=get_compressor.cache_info().currsize > 0)


@app.post('/api/projects', status_code=201, response_model=ProjectResponse)
def create_project(data: ProjectCreate):
    with Session() as db:
        p = Project(name=data.name.strip())
        db.add(p)
        db.commit()
        return serialize(p)


@app.get('/api/projects', response_model=list[ProjectListItem])
def list_projects():
    with Session() as db:
        return [dict(**serialize(p), paper_count=db.scalar(select(func.count()).select_from(Paper).where(Paper.project_id == p.id)),
                     job=serialize(j) if (j := latest_job(db, p.id)) else None)
                for p in db.scalars(select(Project).order_by(Project.created_at.desc()))]


@app.get('/api/projects/{project_id}', response_model=ProjectDetail)
def get_project(project_id: str):
    with Session() as db:
        project = require(db, Project, project_id)
        return dict(**serialize(project), papers=[serialize(p) for p in db.scalars(select(Paper).where(Paper.project_id == project_id))],
                    jobs=[serialize(j) for j in db.scalars(select(Job).where(Job.project_id == project_id).order_by(Job.created_at.desc()))])


@app.delete('/api/projects/{project_id}', status_code=204)
def delete_project(project_id: str):
    with mutation_lock, Session() as db:
        project = require(db, Project, project_id)
        ensure_idle(db, project_id)
        ids = list(db.scalars(select(Paper.id).where(Paper.project_id == project_id)))
        db.delete(project)
        db.commit()
        for paper_id in ids:
            (settings.storage_path / f'{paper_id}.pdf').unlink(missing_ok=True)


@app.post('/api/projects/{project_id}/papers', status_code=201, response_model=list[UploadResponse], response_model_exclude_none=True)
def upload(project_id: str, files: list[UploadFile] = File(...)):
    if not 1 <= len(files) <= 20:
        raise HTTPException(422, 'Upload 1–20 files per request.')
    outcomes = []
    with mutation_lock, Session() as db:
        require(db, Project, project_id)
        ensure_idle(db, project_id)
        for upload_file in files:
            paper_id = uuid4().hex
            path = settings.storage_path / f'{paper_id}.pdf'
            filename = Path((upload_file.filename or 'paper.pdf').replace('\\', '/')).name[:200]
            try:
                if not filename.lower().endswith('.pdf'):
                    raise ValueError('Only PDF files are accepted.')
                digest, size = hashlib.sha256(), 0
                with path.open('wb') as out:
                    while data := upload_file.file.read(1024 * 1024):
                        if size == 0 and not data.startswith(b'%PDF-'):
                            raise ValueError('File is not a PDF (invalid signature).')
                        size += len(data)
                        if size > settings.max_upload_size_mb * 1024 * 1024:
                            raise ValueError(f'File exceeds {settings.max_upload_size_mb} MB limit.')
                        digest.update(data)
                        out.write(data)
                duplicate = db.scalar(select(Paper).where(Paper.project_id == project_id, Paper.sha256 == digest.hexdigest()))
                if duplicate:
                    path.unlink(missing_ok=True)
                    outcomes.append({'filename': filename, 'duplicate': True, 'paper': serialize(duplicate)})
                    continue
                start = time.perf_counter()
                metadata, pages, warnings = extract_pdf(path)
                paper = Paper(id=paper_id, project_id=project_id, filename=filename, sha256=digest.hexdigest(),
                              title=metadata['title'] or filename, metadata_json=metadata, warnings=warnings,
                              extraction_seconds=time.perf_counter()-start)
                db.add(paper)
                db.flush()
                for page in pages:
                    db.add(Page(paper_id=paper_id, **page))
                invalidate(db, project_id)
                db.commit()
                outcomes.append({'filename': filename, 'duplicate': False, 'paper': serialize(paper)})
            except (ValueError, RuntimeError) as exc:
                db.rollback()
                path.unlink(missing_ok=True)
                outcomes.append({'filename': filename, 'error': str(exc)})
            finally:
                upload_file.file.close()
    return outcomes


@app.get('/api/projects/{project_id}/papers', response_model=list[PaperResponse])
def project_papers(project_id: str):
    return get_project(project_id)['papers']


@app.get('/api/papers/{paper_id}', response_model=PaperResponse)
def get_paper(paper_id: str):
    with Session() as db:
        return serialize(require(db, Paper, paper_id))


@app.patch('/api/papers/{paper_id}', response_model=PaperResponse)
def edit_metadata(paper_id: str, data: MetadataEdit):
    with mutation_lock, Session() as db:
        paper = require(db, Paper, paper_id)
        ensure_idle(db, paper.project_id)
        paper.title = data.title
        paper.metadata_json = {**paper.metadata_json, **data.model_dump()}
        db.commit()
        return serialize(paper)


@app.delete('/api/papers/{paper_id}', status_code=204)
def delete_paper(paper_id: str):
    with mutation_lock, Session() as db:
        paper = require(db, Paper, paper_id)
        ensure_idle(db, paper.project_id)
        invalidate(db, paper.project_id)
        db.delete(paper)
        db.commit()
        (settings.storage_path / f'{paper.id}.pdf').unlink(missing_ok=True)


@app.get('/api/papers/{paper_id}/source')
def source_pdf(paper_id: str):
    with Session() as db:
        paper = require(db, Paper, paper_id)
        return FileResponse(settings.storage_path / f'{paper.id}.pdf', media_type='application/pdf', filename=paper.filename)


@app.get('/api/papers/{paper_id}/chunks')
def get_chunks(paper_id: str, job_id: str | None = None):
    with Session() as db:
        paper = require(db, Paper, paper_id)
        job = latest_job(db, paper.project_id, job_id)
        result = []
        for chunk in db.scalars(select(Chunk).where(Chunk.paper_id == paper_id).order_by(Chunk.ordinal)):
            compression = db.scalar(select(Compression).where(Compression.chunk_id == chunk.id, Compression.job_id == job.id)) if job else None
            result.append(dict(**serialize(chunk), compression=compression.data if compression else None))
        return result


@app.get('/api/evidence/{chunk_id}')
def evidence(chunk_id: str):
    with Session() as db:
        return source_record(require(db, Chunk, chunk_id))


@app.post('/api/projects/{project_id}/analyze', status_code=202, response_model=JobResponse)
def analyze(project_id: str, config: Analyze):
    if any(len(term) > 80 for term in config.protected_terms):
        raise HTTPException(422, 'Protected terms must be at most 80 characters.')
    if config.baseline and not settings.enable_baseline_comparison:
        raise HTTPException(422, 'Set ENABLE_BASELINE_COMPARISON=true to enable the extra evaluation run.')
    with mutation_lock, Session() as db:
        require(db, Project, project_id)
        ensure_idle(db, project_id)
        papers = db.scalars(select(Paper).where(Paper.project_id == project_id)).all()
        if not papers:
            raise HTTPException(422, 'Upload at least one valid PDF first.')
        # Single local inference worker; bound queue and memory consumption.
        if db.scalar(select(func.count()).select_from(Job).where(Job.status.in_(['queued', 'running']))) >= 8:
            raise HTTPException(429, 'Worker queue is full. Try again after an analysis finishes.')
        snapshot = dict(**config.model_dump(), compression_model=settings.compression_model,
                        chunk_target=settings.chunk_target_tokens, context=settings.ollama_context,
                        output_tokens=settings.ollama_output_tokens, source_hashes=sorted(p.sha256 for p in papers))
        job = Job(project_id=project_id, config=snapshot)
        db.add(job)
        db.commit()
        executor.submit(run_job, job.id)
        return serialize(job)


@app.get('/api/jobs/{job_id}', response_model=JobResponse)
def get_job(job_id: str):
    with Session() as db:
        return serialize(require(db, Job, job_id))


@app.post('/api/jobs/{job_id}/retry', status_code=202, response_model=JobResponse)
def retry(job_id: str):
    with mutation_lock, Session() as db:
        job = require(db, Job, job_id)
        ensure_idle(db, job.project_id)
        if job.status != 'failed':
            raise HTTPException(409, 'Only failed jobs can be resumed.')
        if any(job.config.get(key) != value for key,value in {
            'compression_model':settings.compression_model, 'context':settings.ollama_context,
            'output_tokens':settings.ollama_output_tokens, 'chunk_target':settings.chunk_target_tokens}.items()):
            raise HTTPException(409, 'Model configuration changed; start a new analysis instead.')
        if db.scalar(select(func.count()).select_from(Job).where(Job.status.in_(['queued', 'running']))) >= 8:
            raise HTTPException(429, 'Worker queue is full.')
        job.status, job.error = 'queued', None
        db.commit()
        executor.submit(run_job, job.id)
        return serialize(job)


def report_response(db, project_id, kind, paper_id=None, job_id=None):
    job = latest_job(db, project_id, job_id, completed=not bool(job_id))
    report = db.scalar(select(Report).where(Report.job_id == job.id, Report.kind == kind, Report.paper_id == paper_id)) if job else None
    if not report:
        raise HTTPException(404, 'This report is not available yet.')
    return dict(**serialize(report), validation='Citation IDs and numerical presence checked; semantic accuracy requires source review.')


@app.get('/api/papers/{paper_id}/summary', response_model=ReportResponse)
def paper_summary(paper_id: str, job_id: str | None = None):
    with Session() as db:
        paper = require(db, Paper, paper_id)
        return report_response(db, paper.project_id, 'paper', paper_id, job_id)


@app.get('/api/projects/{project_id}/synthesis', response_model=ReportResponse)
def synthesis(project_id: str, job_id: str | None = None):
    with Session() as db:
        require(db, Project, project_id)
        return report_response(db, project_id, 'synthesis', job_id=job_id)


@app.get('/api/projects/{project_id}/analytics')
def project_analytics(project_id: str, job_id: str | None = None):
    with Session() as db:
        require(db, Project, project_id)
        return analytics(db, project_id, latest_job(db, project_id, job_id))


def export_data(db, project_id, job_id=None):
    project = require(db, Project, project_id)
    job = latest_job(db, project_id, job_id, completed=not bool(job_id))
    if not job:
        raise HTTPException(404, 'No completed analysis to export.')
    papers = db.scalars(select(Paper).where(Paper.project_id == project_id)).all()
    reports = db.scalars(select(Report).where(Report.job_id == job.id, Report.kind.in_(['paper', 'synthesis']))).all()
    sources = [source_record(c) for c in db.scalars(select(Chunk).join(Paper).where(Paper.project_id == project_id))]
    return {'project': serialize(project), 'job': serialize(job), 'papers': [serialize(p) for p in papers],
            'reports': [serialize(r) for r in reports], 'evidence': sources, 'analytics': analytics(db, project_id, job)}


def markdown(data, paper_id=None):
    lines = [f'# {data["project"]["name"]}', '', 'Generated analysis. Verify claims against original evidence.', '']
    titles = {p['id']: p['title'] for p in data['papers']}
    for report in data['reports']:
        if paper_id and report['paper_id'] != paper_id:
            continue
        lines += [f'## {titles.get(report["paper_id"], "Cross-paper synthesis")}', '']
        for section in report['data']['sections']:
            lines += [f'### {section["heading"]}', '']
            if section.get('notice'):
                lines += [section['notice'], '']
            lines += [f'- {c["text"]} ({c["kind"]}) ' + ' '.join(f'[^{ref}]' for ref in c['evidence_ids']) for c in section['claims']] or ([] if section.get('notice') else ['Information not provided.'])
            lines.append('')
    lines += ['## Evidence references', '']
    for e in data['evidence']:
        if paper_id and e['paper_id'] != paper_id:
            continue
        lines += [f'[^{e["chunk_id"]}]: {titles[e["paper_id"]]} | paper {e["paper_id"]} | {e["section"]} | pages {e["page_start"]}–{e["page_end"]}', '', '    > ' + e['text'].replace('\n', '\n    > '), '']
    return '\n'.join(lines)


@app.get('/api/projects/{project_id}/export')
def export(project_id: str, format: str = Query('json', pattern='^(json|md|csv)$'), job_id: str | None = None, paper_id: str | None = None):
    with Session() as db:
        data = export_data(db, project_id, job_id)
        if paper_id and paper_id not in {p['id'] for p in data['papers']}:
            raise HTTPException(404, 'Paper not in project')
        if format == 'json':
            content, mime = json.dumps(data, indent=2, ensure_ascii=False), 'application/json'
        elif format == 'md':
            content, mime = markdown(data, paper_id), 'text/markdown'
        else:
            out = io.StringIO()
            writer = csv.writer(out)
            writer.writerow(['paper_id', 'title', 'research_problem', 'method', 'main_finding', 'limitations', 'evidence_ids'])
            def safe(value):
                value = str(value)
                return "'" + value if value.lstrip().startswith(('=', '+', '-', '@', '\t', '\r')) else value
            for p in data['papers']:
                report = next((r for r in data['reports'] if r['paper_id'] == p['id']), None)
                sections = report['data']['sections'] if report else []
                def field(heading):
                    return ' '.join(c['text'] for s in sections if s['heading'].lower() == heading for c in s['claims'])
                refs = sorted({ref for s in sections for c in s['claims'] for ref in c['evidence_ids']})
                writer.writerow([safe(v) for v in [p['id'], p['title'], field('research objective'), field('methodology'), field('key findings'), field('limitations'), ';'.join(refs)]])
            content, mime = out.getvalue(), 'text/csv'
        return Response(content, media_type=mime, headers={'Content-Disposition': f'attachment; filename="researchcompress.{format}"'})
