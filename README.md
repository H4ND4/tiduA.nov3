# Audit Workbench — Assignment Version

## Run

Open PowerShell in this folder and run:

    pip install -r requirements.txt
    python -m streamlit run app.py

## Supported files

- CSV
- XLSX

## Included methods

- Odd-numbered rows
- Top N rows
- Random rows with reproducible seed
- Systematic random sampling
- Benford first-digit audit test
- Public holiday transactions
- Weekend transactions
- 000'000 ending transactions

The app uses vectorized pandas/NumPy operations and does not iterate row-by-row for the main tests.
