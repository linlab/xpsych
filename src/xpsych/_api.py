"""Short entry points for the two principal analyses."""
from . import geometry
from .scoring import Compass


def compass(data, bank, *, scorer=None):
    """Score ordered item activities, or text when a scorer is supplied.

    With no scorer, data is observations × items. Returns a mapping from each
    construct name to its signed weighted sum for every observation. With a
    scorer, data is a sequence of texts; the scorer produces item activities.
    Scores are not calibrated questionnaire responses or clinical thresholds.
    """
    model = Compass(bank, scorer)
    return model.score(data) if scorer is None else model.score_texts(data)


def rsa(first, second=None, *, input_type="profiles", control=None,
        method="spearman", n_permutations=999, seed=None):
    """Construct an item RDM, or compare two sources with item-label inference.

    With profiles (default), each input has observations in rows and the same
    ordered items in columns. One input returns its correlation-distance RDM;
    two inputs return an RSAResult comparing their RDMs. Sources may have
    different observation counts. No participant matching is inferred.

    Use input_type="rdm" for two already constructed item matrices. A control,
    if supplied, is always an aligned item × item matrix in the same distance
    convention. A single input cannot be used with a control. Use
    validation.paired_correlation separately for person-level agreement.
    """
    if input_type not in {"profiles", "rdm"}:
        raise ValueError("input_type must be profiles or rdm")
    if second is None:
        if input_type != "profiles" or control is not None:
            raise ValueError("provide two inputs for RDM comparison or controlled RSA")
        return geometry.rdm(first)
    if input_type == "profiles":
        first, second = geometry.rdm(first), geometry.rdm(second)
    return geometry.compare(first, second, control=control, method=method,
                            n_permutations=n_permutations, seed=seed)
