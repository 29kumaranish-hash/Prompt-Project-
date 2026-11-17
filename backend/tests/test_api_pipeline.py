from types import SimpleNamespace
import pytest
from sqlalchemy import select
from app.db import Session, Job, Chunk, Report, Usage
from app.services.ollama_client import OllamaClient
from app.workers import pipeline
from app import main
from app.config import settings
from conftest import TestTokenizer


def setup_project(client, pdf_bytes):
    project=client.post('/api/projects',json={'name':'Research'}).json()
    result=client.post(f'/api/projects/{project["id"]}/papers',files={'files':('test.pdf',pdf_bytes,'application/pdf')})
    assert result.status_code==201
    return project['id'],result.json()[0]['paper']['id']


def test_upload_duplicates_invalid_and_metadata(client,pdf_bytes):
    project,paper=setup_project(client,pdf_bytes)
    path=f'/api/projects/{project}/papers'
    assert client.post(path,files={'files':('duplicate.pdf',pdf_bytes)}).json()[0]['duplicate']
    assert 'error' in client.post(path,files={'files':('../bad.pdf',b'not pdf')}).json()[0]
    assert 'error' in client.post(path,files={'files':('bad.txt',pdf_bytes)}).json()[0]
    assert client.patch(f'/api/papers/{paper}',json={'title':'Corrected','year':2024}).status_code==200
    assert client.get(f'/api/projects/{project}').json()['papers'][0]['title']=='Corrected'
    assert client.get('/api/papers/../../etc/passwd').status_code==404
    assert len(client.get(path).json())==1


def test_invalid_api_requests(client):
    assert client.post('/api/projects',json={'name':'   '}).status_code==422
    project=client.post('/api/projects',json={'name':'Valid'}).json()['id']
    assert client.post(f'/api/projects/{project}/analyze',json={'retention':0}).status_code==422
    assert client.post(f'/api/projects/{project}/analyze',json={}).status_code==422
    assert client.get('/api/projects/missing').status_code==404


def test_password_and_scan_rejected(client):
    import pymupdf
    project=client.post('/api/projects',json={'name':'Test'}).json()['id']
    doc=pymupdf.open();doc.new_page()
    scanned=doc.tobytes()
    encrypted=doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256,owner_pw='owner',user_pw='secret')
    for content in (scanned,encrypted,b'%PDF-1.7\ncorrupt'):
        result=client.post(f'/api/projects/{project}/papers',files={'files':('test.pdf',content)}).json()
        assert 'error' in result[0]


def test_upload_limit(client,pdf_bytes,monkeypatch):
    monkeypatch.setattr(settings,'max_upload_size_mb',1)
    project=client.post('/api/projects',json={'name':'Test'}).json()['id']
    result=client.post(f'/api/projects/{project}/papers',files={'files':('big.pdf',b'%PDF-'+b'x'*1024*1024)}).json()
    assert 'exceeds' in result[0]['error']


def test_background_failure_and_retry(client,pdf_bytes,monkeypatch):
    project,paper=setup_project(client,pdf_bytes)
    monkeypatch.setattr(main.executor,'submit',lambda *args:None)
    monkeypatch.setattr(OllamaClient,'status',lambda:{'available':False,'models':[]})
    job=client.post(f'/api/projects/{project}/analyze',json={}).json()
    assert client.delete(f'/api/papers/{paper}').status_code==409
    pipeline.run_job(job['id'])
    failed=client.get(f'/api/jobs/{job["id"]}').json()
    assert failed['status']=='failed' and 'ollama pull' in failed['error']
    assert client.post(f'/api/jobs/{job["id"]}/retry').status_code==202


class FakeCompressor:
    tokenizer=TestTokenizer()
    limit=512
    def compress(self,text,rate,protected=()):
        return dict(compressed_text=text,original_tokens=len(text),compressed_tokens=len(text),ratio=1,
                    reduction_percent=0,seconds=.01,tokenizer='test-only',fallback=None)


def test_two_paper_pipeline_persistence_exports_and_resume(client,pdf_bytes,monkeypatch):
    import pymupdf
    project,paper=setup_project(client,pdf_bytes)
    doc=pymupdf.open(stream=pdf_bytes,filetype='pdf');doc.set_metadata({'title':'A second study'})
    other=client.post(f'/api/projects/{project}/papers',files={'files':('second.pdf',doc.tobytes())}).json()[0]['paper']['id']
    monkeypatch.setattr(main.executor,'submit',lambda *args:None)
    monkeypatch.setattr(pipeline,'get_compressor',lambda:FakeCompressor())
    monkeypatch.setattr(OllamaClient,'status',lambda:{'available':True,'models':[settings.ollama_model]})
    calls=[]
    def generate(self,data,instruction,sources,stage,cross=False):
        calls.append(stage)
        self.record_usage(stage=stage,model=self.model,input_tokens=100,output_tokens=30,seconds=.1,accepted=True)
        return {'sections':[{'heading':stage.split(':',1)[1] if stage.startswith('synthesis:') else 'Key findings','claims':[{'text':'A controlled study is reported.',
                'evidence_ids':[next(iter(sources))],'kind':'reported'}]}]}
    monkeypatch.setattr(OllamaClient,'generate',generate)
    job=client.post(f'/api/projects/{project}/analyze',json={}).json()
    pipeline.run_job(job['id'])
    assert client.get(f'/api/jobs/{job["id"]}').json()['status']=='completed'
    assert calls.count('paper')==2
    assert {stage for stage in calls if stage.startswith('synthesis:')} == {
        'synthesis:Major findings','synthesis:Agreements','synthesis:Contradictions'}
    report=client.get(f'/api/papers/{paper}/summary').json()
    ref=report['data']['sections'][0]['claims'][0]['evidence_ids'][0]
    evidence=client.get(f'/api/evidence/{ref}').json()
    assert evidence['paper_id']==paper and evidence['page_start']==1
    synthesis=client.get(f'/api/projects/{project}/synthesis')
    assert synthesis.status_code==200
    assert len(synthesis.json()['data']['sections'])==9
    stats=client.get(f'/api/projects/{project}/analytics').json()
    assert stats['llm_input_tokens']==len(calls)*100
    assert stats['llm_output_tokens']==len(calls)*30
    before=len(calls)
    pipeline.run_job(job['id'])
    assert len(calls)==before  # persisted completed stages are reused
    for format in ('md','json','csv'):
        response=client.get(f'/api/projects/{project}/export?format={format}')
        assert response.status_code==200 and paper in response.text
    other_project=client.post('/api/projects',json={'name':'Other'}).json()['id']
    assert client.get(f'/api/projects/{other_project}/analytics?job_id={job["id"]}').status_code==404
    assert client.delete(f'/api/papers/{other}').status_code==204
    assert client.get(f'/api/jobs/{job["id"]}').status_code==404


def test_invalid_structured_responses_count_retries(monkeypatch):
    usage=[]
    class Client:
        def __init__(self,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def post(self,*args,**kwargs):return SimpleNamespace(raise_for_status=lambda:None,json=lambda:{'message':{'content':'not json'},'prompt_eval_count':10,'eval_count':2})
    monkeypatch.setattr('app.services.ollama_client.httpx.Client',Client)
    client=OllamaClient('test',lambda **kw:usage.append(kw))
    with pytest.raises(ValueError,match='after 3 attempts'):client.generate({},'Summarize',{},'chunk')
    assert len(usage)==3 and sum(u['input_tokens'] for u in usage)==30
    assert not any(u['accepted'] for u in usage)


def test_context_guard():
    client=OllamaClient('test')
    assert not client.fits({'text':'x'*settings.ollama_context},'Summarize')


def test_baseline_does_not_load_or_execute_classifier(client,pdf_bytes,monkeypatch):
    project,paper=setup_project(client,pdf_bytes)
    monkeypatch.setattr(main.executor,'submit',lambda *args:None)
    monkeypatch.setattr(pipeline,'get_tokenizer_only',lambda:FakeCompressor())
    def forbidden(): raise AssertionError('Baseline must not initialize classifier weights')
    monkeypatch.setattr(pipeline,'get_compressor',forbidden)
    monkeypatch.setattr(OllamaClient,'status',lambda:{'available':True,'models':[settings.ollama_model]})
    monkeypatch.setattr(OllamaClient,'generate',lambda *args,**kwargs:{'sections':[{'heading':'No findings','claims':[]}]})
    job=client.post(f'/api/projects/{project}/analyze',json={'baseline':True}).json()
    pipeline.run_job(job['id'])
    assert client.get(f'/api/jobs/{job["id"]}').json()['status']=='completed'
    chunks=client.get(f'/api/papers/{paper}/chunks').json()
    assert all(c['original_text']==c['compression']['compressed_text'] and c['compression']['seconds']==0 for c in chunks)
