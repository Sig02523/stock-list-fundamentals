# Stock List Fundamentals

A shareable, interactive Streamlit app that shows a daily fundamentals /
valuation table for a configurable list of tickers. Fundamentals come from
[Financial Modeling Prep](https://financialmodelingprep.com) (`FMP_API_KEY`
required — .env locally, Actions secret in CI, `st.secrets` on Streamlit
Cloud); price history and next-earnings dates come from free Yahoo data via
[`yfinance`](https://github.com/ranaroussi/yfinance). ~7 FMP calls per equity
per pull — the free FMP tier (250/day) covers one full refresh of ~35 tickers
per day; upgrade to Starter for more.

- **`metrics.py`** — pure data layer. `build_report(tickers_df)` returns one
  row per ticker with ~40 metrics. No Streamlit imports — safe to call from a
  notebook, CLI, or CI job.
- **`app.py`** — Streamlit UI. Editable ticker grid, live refresh, sortable
  colored table, Excel download, missing-fields diagnostics.
- **`snapshot.py`** — daily cache builder. Writes `snapshot.parquet` (fast
  reload by `app.py`) and `report.xlsx` (formatted spreadsheet).
- **`tickers.csv`** — default ticker list. Columns: `ticker`, `benchmark`
  (`SPX` or `NDX`), optional `note`.
- **`.github/workflows/snapshot.yml`** — daily cron (~07:00 ET) that rebuilds
  the snapshot and commits it, so the app always opens to a fresh view.

## Metrics

For each ticker, the report includes:

| Source | Metrics |
| --- | --- |
| `yfinance.info` | Last price, 52w high/low, market cap, P/E (trailing & forward), P/S, P/B, revenue growth (TTM), ROA, ROE, free cash flow, current & quick ratios, debt/equity, gross & operating margin, sector, industry, dividend yield, next earnings date, forward EPS & revenue growth (best-effort) |
| Price history (~2y daily) | YTD %, % from 52w high, 2-month annualized realized vol (~42 trading days), **beta** vs the row's assigned benchmark (^GSPC for SPX, ^NDX for NDX) via OLS on ~1y daily returns. Yahoo's default beta field is **not** used. |
| Derived | Debt/Assets, EV/EBITDA, EV/Sales, FCF yield (FCF ÷ market cap), Net Debt / EBITDA, Interest coverage (EBIT ÷ interest expense), PEG, Avg daily $ volume (30d) |

Defensive throughout: missing inputs return `null`, every network call has
retry/backoff, and one bad ticker never crashes the run. Each row carries a
`missing_fields` diagnostic.

## Local dev

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Build a snapshot (writes snapshot.parquet + report.xlsx)
python snapshot.py

# Run the app
streamlit run app.py
```

## Deploying to Streamlit Community Cloud

1. Push this repo to GitHub (public or private).
2. Go to [share.streamlit.io](https://share.streamlit.io), click **New app**,
   and connect your GitHub account.
3. Select the repo, branch (`main`), and entrypoint **`app.py`**.
4. (Optional) Under **Advanced settings → Secrets**, add a password gate:
   ```toml
   APP_PASSWORD = "your-shared-password"
   ```
   Leave it blank to run the app open.
5. Deploy. You'll get a public URL like `https://<app-name>.streamlit.app`.

**Cold-start note.** Streamlit Community Cloud sleeps free-tier apps after
~7 days of inactivity (and after several minutes of no traffic on some plans),
so the first hit may take 30–60s while the container wakes and re-imports.
The cached `snapshot.parquet` is what makes the *second* render instant — the
GitHub Action keeps it fresh.

## Editing the default ticker list

Open `tickers.csv` and add/remove rows. Columns:

| Column | Required | Notes |
| --- | --- | --- |
| `ticker` | yes | Yahoo symbol (e.g. `AAPL`, `BRK-B`, `TSM`) |
| `benchmark` | yes | `SPX` or `NDX` — determines the beta regression's market index |
| `note` | no | Free-text comment shown in the UI |

Commit the file. The next daily action run will pick up the new list. Users
of the deployed app can also edit the grid in the sidebar at runtime — that
edit is per-session and doesn't change `tickers.csv`.

## GitHub Action: daily snapshot

`.github/workflows/snapshot.yml` runs at 12:00 UTC every day (≈ 07:00 ET in
winter, 08:00 ET in summer) and on manual `workflow_dispatch`. It:

1. Installs the same `requirements.txt`.
2. Runs `python snapshot.py`.
3. Commits the refreshed `snapshot.parquet` and `report.xlsx` back to `main`
   (only if they changed).

Make sure **Settings → Actions → General → Workflow permissions** is set to
**Read and write permissions** so the action can push the commit.

## License

MIT (do whatever you like; no warranty).
