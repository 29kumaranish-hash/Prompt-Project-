from typing import Literal
from pydantic import BaseModel, Field, ConfigDict
from .config import settings


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120, pattern=r'\S')


class Analyze(BaseModel):
    retention: float = Field(settings.default_compression_rate, ge=0.1, le=1)
    model: str = Field(settings.ollama_model, min_length=1, max_length=120, pattern=r'^[\w./:-]+$')
    baseline: bool = False
    synthesis_compression: bool = True
    protected_terms: list[str] = Field(default_factory=list, max_length=40)


class MetadataEdit(BaseModel):
    title: str = Field(min_length=1, max_length=500)
    authors: str | None = Field(None, max_length=2000)
    year: int | None = Field(None, ge=1500, le=2200)
    doi: str | None = Field(None, max_length=200)


class Claim(BaseModel):
    model_config = ConfigDict(extra='forbid')
    text: str = Field(min_length=1, max_length=1000)
    evidence_ids: list[str] = Field(min_length=1, max_length=12)
    kind: Literal['reported', 'analysis'] = 'reported'


class Section(BaseModel):
    model_config = ConfigDict(extra='forbid')
    heading: str = Field(max_length=100)
    claims: list[Claim] = Field(max_length=8)
    notice: str | None = Field(None, max_length=500)


class Summary(BaseModel):
    model_config = ConfigDict(extra='forbid')
    sections: list[Section] = Field(min_length=1, max_length=12)


class JobResponse(BaseModel):
    id: str
    project_id: str
    status: Literal['queued', 'running', 'completed', 'failed']
    stage: str
    progress: float = Field(ge=0, le=1)
    config: dict
    error: str | None
    created_at: str
    seconds: float = Field(ge=0)


class PaperResponse(BaseModel):
    id: str
    project_id: str
    filename: str
    sha256: str
    title: str
    metadata_json: dict
    warnings: list[str]
    extraction_seconds: float = Field(ge=0)


class ProjectResponse(BaseModel):
    id: str
    name: str
    created_at: str


class ProjectListItem(ProjectResponse):
    paper_count: int = Field(ge=0)
    job: JobResponse | None


class ProjectDetail(ProjectResponse):
    papers: list[PaperResponse]
    jobs: list[JobResponse]


class UploadResponse(BaseModel):
    filename: str
    duplicate: bool | None = None
    paper: PaperResponse | None = None
    error: str | None = None


class ReportResponse(BaseModel):
    id: str
    job_id: str
    paper_id: str | None
    chunk_id: str | None
    kind: Literal['paper', 'synthesis']
    data: Summary
    validation: str
