"""Real IPIP-50 inventory and keys; simulated activities, no model download."""
import numpy as np
import xpsych as xp

bank = xp.instruments.ipip50()
# Simulated observations keep this example offline. These are not measured people.
rng = np.random.default_rng(7)
activities = rng.normal(size=(60, len(bank.items)))
scores = xp.compass(activities, bank)
rdm = xp.rsa(activities)
# A second simulated source illustrates comparison, not empirical validation.
second_source = activities + rng.normal(size=activities.shape)
comparison = xp.rsa(activities, second_source, n_permutations=99, seed=7)
print("First person's COMPASS scores:", {name: float(values[0]) for name, values in scores.items()})
print("Item RDM shape:", rdm.shape)
print("RSA between simulated sources:", comparison)
