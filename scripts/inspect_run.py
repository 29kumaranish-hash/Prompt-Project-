"""Read persisted local run progress without requiring a running HTTP service."""
import argparse
import json
import sqlite3
from pathlib import Path

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--database',type=Path,default=Path(__file__).resolve().parents[1]/'storage/researchcompress.db')
parser.add_argument('--diagnostics',action='store_true')
args=parser.parse_args()
with sqlite3.connect(args.database.resolve().as_uri()+'?mode=ro',uri=True) as db:
    db.row_factory=sqlite3.Row
    result={'jobs':[dict(r) for r in db.execute('select id,status,stage,progress,seconds,error from jobs order by created_at desc')],
            'reports':[dict(r) for r in db.execute('select kind,count(*) as count from reports group by kind')],
            'usage':dict(db.execute('select count(*) as calls,sum(accepted) as accepted,sum(input_tokens) as input_tokens,sum(output_tokens) as output_tokens from usage').fetchone())}
    if args.diagnostics:
        result['diagnostics']=[json.loads(r[0]) for r in db.execute('select data from reports where kind=? order by rowid desc limit 2',('rejected_response',))]
    print(json.dumps(result,indent=2,ensure_ascii=True))
