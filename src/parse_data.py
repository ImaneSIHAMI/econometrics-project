from pathlib import Path
from datetime import datetime
import zipfile
import xml.etree.ElementTree as ET

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


RAW_DIR = Path("data/raw")
INTERIM_DIR = Path("data/interim")

INTERIM_DIR.mkdir(parents=True, exist_ok=True)

YEARS = range(2020, 2026)

PRICE_BATCH_SIZE = 100_000


PRICE_SCHEMA = pa.schema([
    ("station_id", pa.string()),
    ("fuel_id", pa.string()),
    ("fuel", pa.string()),
    ("timestamp", pa.timestamp("us")),
    ("price", pa.float64()),
    ("source_year", pa.int16()),
])


def local_name(tag):
    """Remove an XML namespace if one exists."""
    return tag.split("}")[-1]


def parse_coordinate(value):
    if value is None or value == "":
        return None

    try:
        return float(value) / 100000
    except ValueError:
        return None


def parse_price(value):
    if value is None or value == "":
        return None

    try:
        return float(value.replace(",", "."))
    except ValueError:
        return None


def parse_datetime(value):
    if value is None or value == "":
        return None

    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def child_text(element, wanted_tag):
    for child in element:
        if local_name(child.tag) == wanted_tag:
            return child.text

    return None


def parse_year(year):
    zip_path = RAW_DIR / f"fuel_prices_{year}.zip"

    prices_output = INTERIM_DIR / f"prices_{year}.parquet"
    stations_output = INTERIM_DIR / f"stations_{year}.parquet"

    print(f"Parsing {year}...")

    station_rows = []
    price_buffer = []

    parquet_writer = None

    def flush_prices():
        nonlocal parquet_writer

        if not price_buffer:
            return

        table = pa.Table.from_pylist(
            price_buffer,
            schema=PRICE_SCHEMA,
        )

        if parquet_writer is None:
            parquet_writer = pq.ParquetWriter(
                prices_output,
                PRICE_SCHEMA,
                compression="snappy",
            )

        parquet_writer.write_table(table)
        price_buffer.clear()

    with zipfile.ZipFile(zip_path) as archive:

        xml_files = [
            filename
            for filename in archive.namelist()
            if filename.lower().endswith(".xml")
        ]

        if not xml_files:
            raise RuntimeError(f"No XML file found for {year}")

        for xml_filename in xml_files:

            print(f"  Reading {xml_filename}")

            with archive.open(xml_filename) as xml_file:

                context = ET.iterparse(
                    xml_file,
                    events=("end",)
                )

                for _, element in context:

                    if local_name(element.tag) != "pdv":
                        continue

                    station_id = element.attrib.get("id")

                    latitude = parse_coordinate(
                        element.attrib.get("latitude")
                    )

                    longitude = parse_coordinate(
                        element.attrib.get("longitude")
                    )

                    postal_code = element.attrib.get("cp")
                    road_type = element.attrib.get("pop")

                    address = child_text(element, "adresse")
                    city = child_text(element, "ville")

                    automate_24_24 = False

                    services = []

                    for descendant in element.iter():

                        tag = local_name(descendant.tag)

                        if tag == "horaires":
                            automate = descendant.attrib.get(
                                "automate-24-24"
                            )

                            automate_24_24 = automate == "1"

                        elif tag == "service":

                            if descendant.text:
                                services.append(
                                    descendant.text.strip()
                                )

                    services = sorted(set(services))

                    station_rows.append({
                        "station_id": station_id,
                        "source_year": year,
                        "latitude": latitude,
                        "longitude": longitude,
                        "postal_code": postal_code,
                        "road_type": road_type,
                        "address": address,
                        "city": city,
                        "automate_24_24": automate_24_24,
                        "n_services": len(services),
                        "services": " | ".join(services),
                    })

                    # Prices belonging to this station
                    for child in element:

                        if local_name(child.tag) != "prix":
                            continue

                        price_buffer.append({
                            "station_id": station_id,
                            "fuel_id": child.attrib.get("id"),
                            "fuel": child.attrib.get("nom"),
                            "timestamp": parse_datetime(
                                child.attrib.get("maj")
                            ),
                            "price": parse_price(
                                child.attrib.get("valeur")
                            ),
                            "source_year": year,
                        })

                        if len(price_buffer) >= PRICE_BATCH_SIZE:
                            flush_prices()

                    # Frees memory progressively
                    element.clear()

    flush_prices()

    if parquet_writer is not None:
        parquet_writer.close()

    stations_df = pd.DataFrame(station_rows)

    stations_df = stations_df.drop_duplicates()

    stations_df.to_parquet(
        stations_output,
        index=False
    )

    print(
        f"{year}: "
        f"{len(stations_df):,} station rows"
    )


def main():

    for year in YEARS:
        parse_year(year)


if __name__ == "__main__":
    main()