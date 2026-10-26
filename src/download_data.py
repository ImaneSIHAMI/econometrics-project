from pathlib import Path
import requests
import zipfile


START_YEAR = 2020
END_YEAR = 2025

BASE_URL = "https://donnees.roulez-eco.fr/opendata/annee"

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)


def download_year(year: int) -> None:
    url = f"{BASE_URL}/{year}"
    output_path = RAW_DIR / f"fuel_prices_{year}.zip"
    temp_path = RAW_DIR / f"fuel_prices_{year}.tmp"

    # Do not download again if the file already exists and is valid.
    if output_path.exists() and zipfile.is_zipfile(output_path):
        print(f"{year}: already downloaded")
        return

    print(f"{year}: downloading...")

    with requests.get(
        url,
        stream=True,
        timeout=(10, 180),
        headers={"User-Agent": "fuel-price-econometrics-project"},
    ) as response:

        response.raise_for_status()

        with open(temp_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)

    if not zipfile.is_zipfile(temp_path):
        temp_path.unlink(missing_ok=True)
        raise RuntimeError(f"{year}: downloaded file is not a valid ZIP")

    temp_path.rename(output_path)

    print(f"{year}: saved to {output_path}")


def main():
    for year in range(START_YEAR, END_YEAR + 1):
        download_year(year)


if __name__ == "__main__":
    main()