'use client';
import { Report } from '@/lib/api';
export function ReportView({ report, onEvidence }: { report: Report; onEvidence: (id:string)=>void }) {
  return <div className="report"><div className="report-note">{report.validation}</div>{report.data.sections.map((section,i)=><section key={i}><div className="report-number">{String(i+1).padStart(2,'0')}</div><div><h3>{section.heading}</h3>{!section.claims.length && <p className="muted">{section.notice || "Information not provided in the available evidence."}</p>}{section.claims.map((claim,j)=><div className="claim" key={j}><p>{claim.kind==='analysis' && <span className="tag">Analysis</span>}{claim.text}</p><div className="citations">{claim.evidence_ids.map(id=><button key={id} onClick={()=>onEvidence(id)} title={`Inspect source ${id}`}>↗ {id.slice(0,8)}</button>)}</div></div>)}</div></section>)}</div>;
}

