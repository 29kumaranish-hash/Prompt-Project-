import os
import pytest
from app.config import settings
from app.services.compressor import get_compressor, numbers
from app.services.chunker import chunk_pages
from app.services.ollama_client import OllamaClient


@pytest.mark.integration
@pytest.mark.skipif(os.getenv('RUN_LLMLINGUA_INTEGRATION') != '1', reason='Set RUN_LLMLINGUA_INTEGRATION=1 to download and execute real weights')
def test_actual_llmlingua2():
    compressor=get_compressor()
    text=('The controlled experiment evaluated the classifier on a held-out test set. '
          'The model achieved 92.5% accuracy across 40 samples. '
          'The results suggest that the method is useful within this experimental setting. '
          'However, evaluation used only one dataset and generalization remains uncertain.')
    chunks=chunk_pages([{'number':1,'blocks':[{'section':'Results','text':text}]}],compressor.tokenizer,400,compressor.limit)
    assert all(c['original_tokens']+compressor.tokenizer.num_special_tokens_to_add(False)<=compressor.limit for c in chunks)
    result=compressor.compress(text,.5)
    assert result['compressed_text'].strip()
    assert numbers(text)<=numbers(result['compressed_text'])
    assert 0<result['compressed_tokens']<=result['original_tokens']
    # Prove actual classifier output, rather than accepting original-text safeguard fallback as integration success.
    raw=compressor.model.compress_prompt(text,rate=.5,force_tokens=['\n'],force_reserve_digit=True)
    assert raw['compressed_prompt']!=text
    print({'tokenizer':settings.compression_model,**result})


@pytest.mark.integration
@pytest.mark.skipif(os.getenv('RUN_OLLAMA_INTEGRATION') != '1', reason='Set RUN_OLLAMA_INTEGRATION=1 with a running model')
def test_actual_ollama():
    usage=[]
    client=OllamaClient(settings.ollama_model,lambda **kw:usage.append(kw))
    source={'c1':{'paper_id':'p1','text':'The model achieved 92.5% accuracy on 40 samples.'}}
    result=client.generate({'chunk_id':'c1','text':source['c1']['text']},'Return one result claim citing the supplied chunk_id.',source,'chunk')
    assert result['sections'] and usage[-1]['input_tokens']>0 and usage[-1]['output_tokens']>0
