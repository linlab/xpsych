"""Instrument items, their order, and construct scoring keys."""
from dataclasses import dataclass
from types import MappingProxyType
import numpy as np
from ._validation import matrix


@dataclass(frozen=True)
class Item:
    """Item text and optional stance hypothesis, identified by a stable item ID."""
    id: str
    text: str
    hypothesis: str | None = None

    def __post_init__(self):
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("item id must be a nonempty string")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("item text must be nonempty")
        if self.hypothesis is not None and (not isinstance(self.hypothesis, str) or not self.hypothesis.strip()):
            raise ValueError("hypothesis must be nonempty when supplied")


class ItemBank:
    """Ordered items and construct keys mapping item IDs to numeric weights.

    Weights are signed linear coefficients, not Likert reverse-scoring rules.
    Text similarity and NLI activity are not calibrated questionnaire responses.
    """
    def __init__(self, items, keys):
        self.items = tuple(items)
        if not self.items or not all(isinstance(i, Item) for i in self.items):
            raise ValueError("items must contain Item objects")
        self.ids = tuple(i.id for i in self.items)
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("item IDs must be unique")
        if not keys:
            raise ValueError("provide at least one construct key")
        checked = {}
        for name, weights in keys.items():
            if not isinstance(name, str) or not name or not weights:
                raise ValueError("each construct needs a name and weights")
            if set(weights) - set(self.ids):
                raise ValueError(f"unknown item ID in key {name}")
            values = np.asarray(list(weights.values()), dtype=float)
            if not np.isfinite(values).all() or not np.any(values):
                raise ValueError("weights must be finite and not all zero")
            checked[name] = MappingProxyType(dict(zip(weights, values)))
        self.keys = MappingProxyType(checked)
        self._weights = np.array([[weights.get(i, 0) for weights in self.keys.values()] for i in self.ids])
        self._weights.setflags(write=False)

    def score(self, activities):
        """Return one array of signed item sums per construct, preserving row order."""
        x = matrix(activities, "activities")
        if x.shape[1] != len(self.items):
            raise ValueError("activity columns must match bank.items in order")
        values = x @ self._weights
        return {name: values[:, j] for j, name in enumerate(self.keys)}



# The 50-item set (five 10-item scales), ordered by the official scoring key.
# Source: https://ipip.ori.org/newBigFive5broadKey.htm
# Items and scales are public domain: https://ipip.ori.org/
_IPIP50 = {
    "extraversion": (
        ("Am the life of the party.", 1),
        ("Feel comfortable around people.", 1),
        ("Start conversations.", 1),
        ("Talk to a lot of different people at parties.", 1),
        ("Don't mind being the center of attention.", 1),
        ("Don't talk a lot.", -1),
        ("Keep in the background.", -1),
        ("Have little to say.", -1),
        ("Don't like to draw attention to myself.", -1),
        ("Am quiet around strangers.", -1),
    ),
    "agreeableness": (
        ("Am interested in people.", 1),
        ("Sympathize with others' feelings.", 1),
        ("Have a soft heart.", 1),
        ("Take time out for others.", 1),
        ("Feel others' emotions.", 1),
        ("Make people feel at ease.", 1),
        ("Am not really interested in others.", -1),
        ("Insult people.", -1),
        ("Am not interested in other people's problems.", -1),
        ("Feel little concern for others.", -1),
    ),
    "conscientiousness": (
        ("Am always prepared.", 1),
        ("Pay attention to details.", 1),
        ("Get chores done right away.", 1),
        ("Like order.", 1),
        ("Follow a schedule.", 1),
        ("Am exacting in my work.", 1),
        ("Leave my belongings around.", -1),
        ("Make a mess of things.", -1),
        ("Often forget to put things back in their proper place.", -1),
        ("Shirk my duties.", -1),
    ),
    "emotional_stability": (
        ("Am relaxed most of the time.", 1),
        ("Seldom feel blue.", 1),
        ("Get stressed out easily.", -1),
        ("Worry about things.", -1),
        ("Am easily disturbed.", -1),
        ("Get upset easily.", -1),
        ("Change my mood a lot.", -1),
        ("Have frequent mood swings.", -1),
        ("Get irritated easily.", -1),
        ("Often feel blue.", -1),
    ),
    "intellect_imagination": (
        ("Have a rich vocabulary.", 1),
        ("Have a vivid imagination.", 1),
        ("Have excellent ideas.", 1),
        ("Am quick to understand things.", 1),
        ("Use difficult words.", 1),
        ("Spend time reflecting on things.", 1),
        ("Am full of ideas.", 1),
        ("Have difficulty understanding abstract ideas.", -1),
        ("Am not interested in abstract ideas.", -1),
        ("Do not have a good imagination.", -1),
    ),
}


def ipip50():
    """Return the public-domain 50-item IPIP Big-Five Factor Markers.

    Item wording and signs follow the official five 10-item scoring keys.
    IDs are construct_1 ... construct_10, grouped by construct, not the item
    numbers of an interleaved questionnaire. Use bank.ids to align your data.
    COMPASS uses signs as linear activity weights; this is not the standard
    1–5 questionnaire scoring rule (negative responses there become 6 - x).
    NLI hypotheses prepend 'I ' to the lowercased initial item word.
    """
    items, keys = [], {}
    for construct, entries in _IPIP50.items():
        keys[construct] = {}
        for number, (text, sign) in enumerate(entries, 1):
            item_id = f"{construct}_{number}"
            items.append(Item(item_id, text, hypothesis="I " + text[0].lower() + text[1:]))
            keys[construct][item_id] = sign
    return ItemBank(items, keys)
