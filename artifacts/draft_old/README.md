# DRAFT_OLD — archived, for comparison only

These are the draft predictors trained on the **deleted** v1 corpus
(`experiments/m2_hidden.npz`, 173,991 samples, 667 MB).

They are here for one reason: the three-way comparison
(`DRAFT_OLD` / `FRESH_SUPERVISED` / `FRESH_RL`) needs the draft arm measured
under the same conditions as the others, and a number measured weeks apart on a
different harness is not a fair comparison.

**Nothing may initialise from these files.** They are frozen ridge weight
matrices and carry no optimizer state, so they cannot continue a training run,
but the rule is about intent rather than mechanism: the fresh model is trained
from scratch on a fresh corpus, and any warm start would make the comparison
meaningless.

They are deliberately not in `models/`, which is where the runtime looks.
