import json
from types import SimpleNamespace
from app.services.ollama_client import OllamaClient, citation_aliases, wire_data


def test_compact_citation_labels_preserve_paper_identity():
    data=[{'text':'Methods','evidence_ids':['long-chunk-a','long-chunk-b']}]
    sources={'long-chunk-a':{'paper_id':'paper-a'},'long-chunk-b':{'paper_id':'paper-b'}}
    aliases=citation_aliases(data)
    wire=wire_data(data,aliases,sources)
    assert wire['notes'][0]['evidence_ids']==['E1','E2']
    assert wire['citation_sources']==[{'evidence_id':'E1','paper_id':'P1'},{'evidence_id':'E2','paper_id':'P2'}]
    schema=OllamaClient('test').payload(data,'Summarize','synthesis',sources)['format']
    assert schema['$defs']['Claim']['properties']['evidence_ids']['items']['enum']==['E1','E2']


def test_generation_restores_original_citation_ids(monkeypatch):
    response={'sections':[{'heading':'Results','claims':[{'text':'Accuracy was 92.5%.','evidence_ids':['E1'],'kind':'reported'}]}]}
    class Client:
        def __init__(self,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def post(self,*args,**kwargs):return SimpleNamespace(raise_for_status=lambda:None,json=lambda:{'message':{'content':json.dumps(response)},'prompt_eval_count':100,'eval_count':50})
    monkeypatch.setattr('app.services.ollama_client.httpx.Client',Client)
    original='ae309778dabc4989880b548abe578673'
    report=OllamaClient('test').generate({'chunk_id':original,'text':'Accuracy was 92.5%.'},'Summarize',
            {original:{'paper_id':'paper','text':'Accuracy was 92.5%.'}},'chunk')
    assert report['sections'][0]['claims'][0]['evidence_ids']==[original]


def test_cross_paper_retry_keeps_targeted_correction(monkeypatch):
    requests=[]
    class Client:
        def __init__(self,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def post(self,*args,**kwargs):
            requests.append(kwargs['json'])
            claims=[{'text':'Methods conflict.','evidence_ids':['E1']} ] if len(requests)==1 else []
            result={'sections':[{'heading':'Contradictions','claims':claims}]}
            return SimpleNamespace(raise_for_status=lambda:None,json=lambda:{'message':{'content':json.dumps(result)}})
    monkeypatch.setattr('app.services.ollama_client.httpx.Client',Client)
    result=OllamaClient('test').generate([{'text':'A method','evidence_ids':['c1']}], 'Compare evidence',
        {'c1':{'paper_id':'p1','text':'A method'}},'synthesis',cross=True)
    assert len(requests)==2
    correction=requests[1]['messages'][1]['content']
    assert 'Contradictions: requires evidence from multiple papers' in correction
    assert 'claims=[]' in correction
    assert result['sections'][0]['claims']==[]
