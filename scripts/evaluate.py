"""Upload REAL local PDFs, run the real pipeline, and save reproducible outputs.

Usage: python scripts/evaluate.py path/to/paper1.pdf path/to/paper2.pdf [--baseline]
No synthetic summaries or mock calls are used by this script.
"""
import argparse
import json
import time
from pathlib import Path
import httpx

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('pdfs', nargs='+', type=Path)
parser.add_argument('--url', default='http://127.0.0.1:8000')
parser.add_argument('--model', default='qwen2.5:3b')
parser.add_argument('--baseline', action='store_true')
parser.add_argument('--retention', type=float, default=.5)
parser.add_argument('--output', type=Path, default=Path('storage/evaluation'))
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
with httpx.Client(base_url=args.url, timeout=180) as client:
    def request(method, path, **kwargs):
        response = client.request(method, '/api' + path, **kwargs)
        response.raise_for_status()
        return response
    project = request('POST', '/projects', json={'name':'Evaluation ' + time.strftime('%Y-%m-%d %H:%M')}).json()
    project_id = project['id']
    print('Project:', project_id, flush=True)
    for path in args.pdfs:
        with path.open('rb') as file:
            uploaded = request('POST', f'/projects/{project_id}/papers', files={'files':(path.name,file,'application/pdf')}).json()
        if uploaded[0].get('error'):
            raise RuntimeError(uploaded[0]['error'])
        print('Uploaded:', path.name, flush=True)
    for baseline in ([False, True] if args.baseline else [False]):
        job=request('POST',f'/projects/{project_id}/analyze',json={'model':args.model,'retention':args.retention,'baseline':baseline}).json()
        last=None
        while job['status'] not in ('completed','failed'):
            time.sleep(3)
            job=request('GET',f'/jobs/{job["id"]}').json()
            if job['stage']!=last:
                print(job['status'], job['stage'], flush=True)
                last=job['stage']
        if job['status']=='failed':
            raise RuntimeError(f"Job {job['id']} failed; resume in UI: {job['error']}")
        label='baseline' if baseline else 'compressed'
        for extension in ('json','md','csv'):
            result=request('GET',f'/projects/{project_id}/export?format={extension}&job_id={job["id"]}')
            (args.output/f'{project_id}-{label}.{extension}').write_bytes(result.content)
        stats=request('GET',f'/projects/{project_id}/analytics?job_id={job["id"]}').json()
        print(json.dumps(stats,indent=2),flush=True)
print('Review results and cited original passages in', args.output)
