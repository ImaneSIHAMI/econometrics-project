# Econometrics Project 2027
* Project led by Imane SIHAMI, Francesco LAURIOLA, Jahouer ESSID and Théophile BERTHELOT*

## Download the data

The raw fuel price data used in this project comes from the official French government open-data platform.

The dataset is not stored in this Git repository because the raw files are relatively large and can be downloaded automatically. The `data/` folder is therefore excluded from Git using `.gitignore`.

### 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows without WSL:

```bash
.venv\Scripts\activate
```

### 2. Install the required dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the raw data

Run:

```bash
python src/download_data.py
```

The script downloads the yearly fuel price archives and stores them in:

```text
data/raw/
```

After running the script, the folder should contain files such as:

```text
data/
└── raw/
    ├── fuel_prices_2020.zip
    ├── fuel_prices_2021.zip
    ├── fuel_prices_2022.zip
    ├── fuel_prices_2023.zip
    ├── fuel_prices_2024.zip
    └── fuel_prices_2025.zip
```

The project mainly studies the 2021–2025 period. The 2020 archive is also downloaded so that prices already in effect at the beginning of 2021 can be reconstructed when necessary.

The download script automatically skips files that have already been downloaded and verifies that each downloaded file is a valid ZIP archive before saving it.

### Data source

French Ministry of Ecological Transition — *Prix des carburants en France*:

https://www.prix-carburants.gouv.fr/rubrique/opendata/

The original files are kept unchanged in `data/raw/`. They are later parsed with `src/parse_data.py` and transformed into analysis-ready tables by the data-processing scripts in `src/`.

## Checking data

After downloading the archives, run the parser from the project root:

```bash
python src/parse_data.py
```

This creates the yearly station and price tables in `data/interim/`.

Open [`notebook/01_data_check.ipynb`](notebook/01_data_check.ipynb) in VS Code or Jupyter, select the project's `.venv` Python kernel, and run all cells in order. The notebook currently checks `stations_2025.parquet` and `prices_2025.parquet`.

If prompted to install a notebook kernel, run `python -m pip install ipykernel` in the activated environment.

The checks cover:

- Dataset size, missing values and fuel types.
- Unique, nonmissing station identifiers.
- Timestamps outside 2025, with missing dates counted separately.
- Duplicate `(station_id, fuel, timestamp)` keys, distinguishing identical prices from conflicting reports.
- Minimum and maximum prices by fuel, with the station and timestamp of each extreme.
- The number of stations with at least one positive, finite Gazole price in 2025, forming the candidate population for the main panel.

The notebook reports issues without changing the parquet files. Continuous coverage must still be checked before constructing the panel.
