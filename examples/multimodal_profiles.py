"""Two simulated modalities in real IPIP-50 item coordinates.

This exercises the profile interface. It is not a trained acoustic model or
validation of personality inference. Each modality needs its own validated
mapping to the instrument's items before these comparisons are interpretable.
"""
import numpy as np
import xpsych as xp

rng = np.random.default_rng(11)
bank = xp.instruments.ipip50()
n_items = len(bank.items)
text_profiles = xp.cosine_activities(rng.normal(size=(80, 64)), rng.normal(size=(n_items, 64)))
audio_profiles = xp.cosine_activities(rng.normal(size=(80, 96)), rng.normal(size=(n_items, 96)))
# Both matrices use bank.ids, despite different raw embedding dimensions.
print("Text scores:", xp.compass(text_profiles, bank)["extraversion"].shape)
print("Audio scores:", xp.compass(audio_profiles, bank)["extraversion"].shape)
print("Simulated cross-modal RSA:", xp.rsa(text_profiles, audio_profiles, seed=11))
