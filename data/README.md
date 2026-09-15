# data/raw/

Put the Dubai Land Department transactions file here as `Transactions.csv`.
This directory is gitignored — the file is never committed.

**Source:** DLD's open `Transactions.csv` (Dubai Pulse), downloaded from the
Kaggle mirror `alexefimik/dubai-real-estate-transactions-dataset`
(https://www.kaggle.com/datasets/alexefimik/dubai-real-estate-transactions-dataset).
Real registered transactions — no synthetic data.

**Expected format:** UTF-8 CSV, 46 columns (see `ingestion/schema.py`),
one row per registered transaction (sales, mortgages, gifts). Missing numbers
are the literal `null`; dates are `DD-MM-YYYY`; sizes are square metres.
Ingestion stops with a schema-drift error if the columns or the transaction /
registration / property type categories change.

**Coverage of the current file:** 1,047,965 rows, 1995-03-07 to 2023-03-17.
