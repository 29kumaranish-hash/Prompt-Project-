from datetime import datetime, timezone
from uuid import uuid4
from sqlalchemy import create_engine, event, String, ForeignKey, JSON, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from .config import settings


def uid():
    return uuid4().hex


def now():
    return datetime.now(timezone.utc).isoformat()


class Base(DeclarativeBase):
    pass


class Project(Base):
    __tablename__ = 'projects'
    id: Mapped[str] = mapped_column(primary_key=True, default=uid)
    name: Mapped[str]
    created_at: Mapped[str] = mapped_column(default=now)


class Paper(Base):
    __tablename__ = 'papers'
    __table_args__ = (UniqueConstraint('project_id', 'sha256'),)
    id: Mapped[str] = mapped_column(primary_key=True, default=uid)
    project_id: Mapped[str] = mapped_column(ForeignKey('projects.id', ondelete='CASCADE'), index=True)
    filename: Mapped[str]
    sha256: Mapped[str]
    title: Mapped[str]
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    warnings: Mapped[list] = mapped_column(JSON, default=list)
    extraction_seconds: Mapped[float] = mapped_column(default=0)


class Page(Base):
    __tablename__ = 'pages'
    id: Mapped[str] = mapped_column(primary_key=True, default=uid)
    paper_id: Mapped[str] = mapped_column(ForeignKey('papers.id', ondelete='CASCADE'), index=True)
    number: Mapped[int]
    text: Mapped[str]
    blocks: Mapped[list] = mapped_column(JSON)


class Chunk(Base):
    __tablename__ = 'chunks'
    id: Mapped[str] = mapped_column(primary_key=True, default=uid)
    paper_id: Mapped[str] = mapped_column(ForeignKey('papers.id', ondelete='CASCADE'), index=True)
    ordinal: Mapped[int]
    section: Mapped[str]
    page_start: Mapped[int]
    page_end: Mapped[int]
    original_text: Mapped[str]
    original_tokens: Mapped[int]


class Job(Base):
    __tablename__ = 'jobs'
    id: Mapped[str] = mapped_column(primary_key=True, default=uid)
    project_id: Mapped[str] = mapped_column(ForeignKey('projects.id', ondelete='CASCADE'), index=True)
    status: Mapped[str] = mapped_column(default='queued')
    stage: Mapped[str] = mapped_column(default='Queued')
    progress: Mapped[float] = mapped_column(default=0)
    config: Mapped[dict] = mapped_column(JSON)
    error: Mapped[str | None]
    created_at: Mapped[str] = mapped_column(default=now)
    seconds: Mapped[float] = mapped_column(default=0)


class Compression(Base):
    __tablename__ = 'compressions'
    __table_args__ = (UniqueConstraint('job_id', 'chunk_id'),)
    id: Mapped[str] = mapped_column(primary_key=True, default=uid)
    job_id: Mapped[str] = mapped_column(ForeignKey('jobs.id', ondelete='CASCADE'), index=True)
    chunk_id: Mapped[str] = mapped_column(ForeignKey('chunks.id', ondelete='CASCADE'))
    data: Mapped[dict] = mapped_column(JSON)


class Report(Base):
    __tablename__ = 'reports'
    id: Mapped[str] = mapped_column(primary_key=True, default=uid)
    job_id: Mapped[str] = mapped_column(ForeignKey('jobs.id', ondelete='CASCADE'), index=True)
    paper_id: Mapped[str | None] = mapped_column(ForeignKey('papers.id', ondelete='CASCADE'))
    chunk_id: Mapped[str | None] = mapped_column(ForeignKey('chunks.id', ondelete='CASCADE'))
    kind: Mapped[str]  # chunk, paper, synthesis, intermediate
    data: Mapped[dict] = mapped_column(JSON)


class Usage(Base):
    __tablename__ = 'usage'
    id: Mapped[str] = mapped_column(primary_key=True, default=uid)
    job_id: Mapped[str] = mapped_column(ForeignKey('jobs.id', ondelete='CASCADE'), index=True)
    stage: Mapped[str]
    model: Mapped[str]
    input_tokens: Mapped[int]
    output_tokens: Mapped[int]
    seconds: Mapped[float]
    accepted: Mapped[bool] = mapped_column(default=False)


engine = create_engine(settings.database_url, connect_args={'check_same_thread': False} if settings.database_url.startswith('sqlite') else {})
if settings.database_url.startswith('sqlite'):
    @event.listens_for(engine, 'connect')
    def configure_sqlite(conn, _):
        conn.execute('PRAGMA foreign_keys=ON')
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('PRAGMA busy_timeout=30000')
Session = sessionmaker(engine, expire_on_commit=False)


def serialize(obj):
    return {col.name: getattr(obj, col.name) for col in obj.__table__.columns}
