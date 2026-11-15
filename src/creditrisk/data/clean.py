"""Fixed recoding of undocumented category codes (ADR-008).

The mapping is written down here and does not depend on the data, so it can be applied to the
whole file before the train/test split without leaking anything.

Only ``EDUCATION`` and ``MARRIAGE`` are recoded. Everything else is kept as it comes in the raw
file, including the ``PAY_*`` codes -2 and 0, negative bills, duplicated rows and extreme amounts.
"""

from __future__ import annotations

import pandas as pd

from creditrisk.data.schema import DOCUMENTED_CODES

# Documented categories: EDUCATION 1 = graduate school, 2 = university, 3 = high school,
# 4 = others; MARRIAGE 1 = married, 2 = single, 3 = others.
EDUCATION_OTHERS = 4
MARRIAGE_OTHERS = 3

# Undocumented codes found in the raw file and the documented category they are moved to.
RECODES: dict[str, dict[int, int]] = {
    "EDUCATION": {0: EDUCATION_OTHERS, 5: EDUCATION_OTHERS, 6: EDUCATION_OTHERS},
    "MARRIAGE": {0: MARRIAGE_OTHERS},
}


def recode_undocumented_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy where undocumented ``EDUCATION`` and ``MARRIAGE`` codes become "others".

    Raises ValueError if a column holds a code that is neither documented nor in ``RECODES``:
    a new code needs its own decision, not a silent default.
    """
    out = df.copy()
    for column, mapping in RECODES.items():
        out[column] = out[column].replace(mapping)
        unexpected = sorted(set(out[column].unique()) - DOCUMENTED_CODES[column])
        if unexpected:
            raise ValueError(f"{column} has codes with no recorded decision: {unexpected}")
    return out
