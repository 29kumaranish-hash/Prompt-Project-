from app.services.summarizer import hierarchical_summary, flatten_reports
from app.services.synthesizer import synthesize
import pytest


class BoundedClient:
    def __init__(self): self.calls=[]
    def fits(self,data,instruction):return len(data)<=2
    def generate(self,data,instruction,sources,stage,cross=False):
        self.calls.append((stage,len(data),cross,data))
        return {'sections':[{'heading':'Key findings','claims':[{'text':'Merged evidence.',
                'evidence_ids':list(dict.fromkeys(ref for n in data for ref in n['evidence_ids'])), 'kind':'reported'}]}]}


def test_hierarchy_never_overfills_and_preserves_original_ids():
    client=BoundedClient()
    notes=[{'text':f'Finding {i}','evidence_ids':[f'c{i}'],'kind':'reported'} for i in range(7)]
    result=hierarchical_summary(client,notes,'Paper summary',{},'paper')
    assert all(count<=2 for _,count,_,_ in client.calls)
    assert any(stage=='paper_intermediate' for stage,_,_,_ in client.calls)
    assert set(flatten_reports([result])[0]['evidence_ids'])=={f'c{i}' for i in range(7)}


def test_synthesis_compresses_only_prose():
    client=BoundedClient();seen=[]
    def compress(text): seen.append(text);return 'Compressed prose'
    reports=[{'sections':[{'heading':'Methodology','claims':[{'text':'Original method', 'evidence_ids':['c1'],'kind':'reported'}]}]}]
    synthesize(client,reports,{},compress)
    assert seen==['Original method']
    assert client.calls[-1][2] is True
    assert client.calls[-1][3][0]['evidence_ids']==['c1']


def test_oversized_note_fails_instead_of_truncation():
    client=BoundedClient();client.fits=lambda *args:False
    with pytest.raises(ValueError,match='exceeds context'):
        hierarchical_summary(client,[{'text':'Large'}],'Paper',{},'paper')


def test_unsupported_comparisons_abstain_visibly_without_accepting_claims():
    from app.services.evidence import UnsupportedComparison
    class Client(BoundedClient):
        def generate(self,data,instruction,sources,stage,cross=False):
            if stage in ('synthesis:Agreements','synthesis:Contradictions'):
                raise UnsupportedComparison('Three unsupported attempts')
            return super().generate(data,instruction,sources,stage,cross)
    reports=[{'sections':[{'heading':'Key findings','claims':[{'text':'A finding','evidence_ids':['c1'],'kind':'reported'}]}]}]
    result=synthesize(Client(),reports,{})
    for section in result['sections']:
        if section['heading'] in ('Agreements','Contradictions'):
            assert section['claims']==[] and 'withheld' in section['notice']


def test_unsupported_comparison_with_value_error_abstains_visibly():
    class Client(BoundedClient):
        def generate(self,data,instruction,sources,stage,cross=False):
            if stage in ('synthesis:Agreements','synthesis:Contradictions'):
                raise ValueError('Contradictions: requires evidence from multiple papers. Use claims=[]')
            return super().generate(data,instruction,sources,stage,cross)
    reports=[{'sections':[{'heading':'Key findings','claims':[{'text':'A finding','evidence_ids':['c1'],'kind':'reported'}]}]}]
    result=synthesize(Client(),reports,{})
    for section in result['sections']:
        if section['heading'] in ('Agreements','Contradictions'):
            assert section['claims']==[] and 'withheld' in section['notice']
