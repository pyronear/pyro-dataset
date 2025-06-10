# Platfrom Train Loop

We deploy new stations or we want to reduce the amount of false positives from
the existing ones. This document explains a proven workflow designed to augment
the dataset with current bad predictions from the running model. The goal being
that retraining the model on a dataset that includes its current mistakes will
make it a more performant model that generates fewer false positives.
