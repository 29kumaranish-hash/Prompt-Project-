import threading
from types import SimpleNamespace
import pytest
from app.services.pdf_extractor import extract_pdf, section_heading
from app.services.chunker import chunk_pages
from app.services.compressor import CompressionService, numbers
from app.services.evidence import validate_evidence
from app.schemas import Summary


def test_extraction_sections_and_unknown_year(tmp_path, pdf_bytes):
    path = tmp_path / 'study.pdf'
    path.write_bytes(pdf_bytes)
    metadata, pages, warnings = extract_pdf(path)
    assert metadata['title'] == 'A Controlled Scientific Study'
    assert metadata['authors'] == 'Test Author'
    assert metadata['year'] is None
    assert '92.5%' in pages[0]['text']
    assert {'Methods', 'Results', 'Limitations'} <= {h['heading'] for h in metadata['headings']}
    assert pages[0]['number'] == 1


@pytest.mark.parametrize('text,expected', [('2.1 Methods','Methods'),('III. RESULTS','Results'),('Our results are promising',None),('References','References'),('1 Introduction and Related Work','Introduction'),('2 Research Questions','Research Questions'),('A Implementation Details','Implementation Details'),('3 models were evaluated.',None)])
def test_sections(text, expected):
    assert section_heading(text) == expected


def test_chunk_boundaries_provenance_and_long_sentences(tokenizer):
    pages = [{'number': 3, 'blocks': [{'section': 'Methods', 'text': 'First statement. Second statement.'}]},
             {'number': 4, 'blocks': [{'section': 'Results', 'text': 'x'*1100}]}]
    chunks = chunk_pages(pages, tokenizer, target=400, encoder_limit=512)
    assert all(c['original_tokens'] + 2 <= 512 for c in chunks)
    assert chunks[0]['section'] == 'Methods' and chunks[0]['page_start'] == 3
    assert all(c['page_start'] == c['page_end'] == 4 for c in chunks[1:])
    assert ''.join(c['original_text'] for c in chunks[1:]) == 'x'*1100


def test_no_chunk_mix_and_budget(tokenizer):
    pages=[{'number':n,'blocks':[{'section':'Results','text':'A result. '*40}]} for n in (1,2)]
    chunks=chunk_pages(pages,tokenizer,target=80,encoder_limit=100)
    assert all(c['original_tokens']<=80 for c in chunks)
    assert chunks[0]['page_start']==1 and chunks[-1]['page_end']==2


def compressor_double(tokenizer, output):
    service = CompressionService.__new__(CompressionService)
    service.tokenizer, service.lock = tokenizer, threading.Lock()
    service.model = SimpleNamespace(compress_prompt=lambda *args, **kwargs: {'compressed_prompt': output})
    return service


def test_compression_statistics(tokenizer):
    result = compressor_double(tokenizer, 'accuracy 92.5%').compress('The accuracy was 92.5% in this study.', .3)
    assert result['compressed_tokens'] == len('accuracy 92.5%')
    assert result['ratio'] == pytest.approx(result['original_tokens']/result['compressed_tokens'])
    assert result['reduction_percent'] > 0


@pytest.mark.parametrize('output', ['', 'accuracy 12%', 'lost numeric measurement'])
def test_numeric_loss_falls_back(tokenizer, output):
    text='The accuracy was 92.5% in this study.'
    result=compressor_double(tokenizer,output).compress(text,.5)
    assert result['compressed_text']==text
    assert result['fallback']
    assert result['reduction_percent']==0


def test_protected_term_loss_falls_back(tokenizer):
    text='Dataset Alpha has many samples.'
    assert compressor_double(tokenizer,'many samples').compress(text,.3,['Alpha'])['compressed_text']==text


def test_citation_and_numeric_validation():
    sources={'c1':{'text':'We reached 92.5% accuracy.', 'paper_id':'p1'}}
    def summary(text, ref='c1'):
        return Summary.model_validate({'sections':[{'heading':'Results','claims':[{'text':text,'evidence_ids':[ref]}]}]})
    validate_evidence(summary('Accuracy reached 92.5%.'),sources)
    with pytest.raises(ValueError,match='unknown evidence'): validate_evidence(summary('A finding','invented'),sources)
    with pytest.raises(ValueError,match='numerical'): validate_evidence(summary('Accuracy reached 99%.'),sources)
    assert numbers('p = 0.05, n = 40 and 92.5%') == {'0.05','40','92.5%'}


def test_cross_paper_requires_multiple_sources():
    s=Summary.model_validate({'sections':[{'heading':'Agreements','claims':[{'text':'Methods agree.','evidence_ids':['c1']}]}]})
    with pytest.raises(ValueError,match='multiple papers'):
        validate_evidence(s,{'c1':{'text':'Methods agree.', 'paper_id':'p1'}},True)


@pytest.mark.parametrize('text',['H. Vicky Zhao1, Lili Qiu2, Dongmei Zhang2','1 Tsinghua University, 2 Microsoft Corporation','N X'])
def test_front_matter_and_math_are_not_section_headings(text):
    assert section_heading(text) is None
