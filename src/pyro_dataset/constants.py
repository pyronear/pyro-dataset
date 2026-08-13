DATE_FORMAT_OUTPUT = "%Y-%m-%dT%H-%M-%S"
CLASS_ID_SMOKE = 0
CLASS_SMOKE_LABEL = "smoke"

# Boxes on a false-positive sequence mark where the detector fired, not smoke.
# A sentinel id keeps them honest and collides with none of the ids present in
# the historical pools (0, 1, 3, 4, 8, 9, 10, 12, 15, 18, 19 — see issue #22).
CLASS_ID_FALSE_POSITIVE_PROPOSAL = 99
CLASS_FALSE_POSITIVE_PROPOSAL_LABEL = "false_positive_proposal"
