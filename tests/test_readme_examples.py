"""Execute README Python blocks using a deterministic local encoder double."""
from pathlib import Path
import hashlib
import re
import numpy as np
from xpsych.text import EmbeddingScorer


def test_readme_workflow_and_custom_inventory(monkeypatch):
    class Encoder:
        def encode(self, texts):
            return np.array([list(hashlib.sha256(t.encode()).digest())[:8] for t in texts], dtype=float)
    monkeypatch.setattr(EmbeddingScorer, 'from_pretrained',
                        staticmethod(lambda *args, **kwargs: EmbeddingScorer(Encoder())))
    root = Path(__file__).resolve().parents[1]
    blocks = re.findall(r'```python\n(.*?)```', (root / 'README.md').read_text(), re.S)
    namespace = {}
    for block in blocks:
        if 'observation_embeddings' in block:
            namespace['observation_embeddings'] = np.ones((3, 8))
            namespace['item_embeddings'] = np.ones((len(namespace['bank'].items), 8))
        exec(compile(block, 'README.md', 'exec'), namespace)
    bank = namespace['bank']
    assert len(bank.items) == 10
    assert sum(bank.keys['extraversion'].values()) == 0
    assert namespace['activities'].shape == (3, 10)
    assert namespace['scores']['extraversion'].shape == (3,)
