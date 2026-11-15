"""Create the Phase 4 sample datasets in data/structured/samples/.

* ``segment_kpis.csv`` holds **real** figures from the ArcelorMittal Annual
  Report 2025: segment sales and operating income (page 95, $ millions) and
  FTE employees by segment at December 31 (page 130), for 2024 and 2025.
* ``steel_orders_sample.xlsx`` holds **synthetic** data (fixed random seed,
  fictional customers) used to test joins, date filters and multi-sheet
  ingestion. Sheets: ``customers``, ``orders`` and an empty ``notes`` sheet
  (which ingestion must skip). It is not related to ArcelorMittal.

Re-running the script produces byte-identical CSV content and the same
workbook values.
"""

from __future__ import annotations

import csv
import random
from datetime import date, timedelta
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data" / "structured" / "samples"

# Annual Report 2025: page 95 (sales, operating income, $ millions) and page 130 (FTE employees).
SEGMENT_KPIS = [
    # segment, year, sales, operating income, employees
    ("North America", 2025, 12335, 2205, 16083), ("North America", 2024, 11896, 1310, 13861),
    ("Brazil", 2025, 11172, 608, 25014), ("Brazil", 2024, 12401, 1399, 22624),
    ("Europe", 2025, 28793, 522, 43878), ("Europe", 2024, 29952, 386, 48544),
    ("Sustainable Solutions", 2025, 10501, 142, 13149), ("Sustainable Solutions", 2024, 10722, 57, 12843),
    ("Mining", 2025, 3232, 789, 5918), ("Mining", 2024, 2663, 770, 4758),
]

CUSTOMERS = [
    ("C001", "Northwind Fabrication", "Germany", "Europe", "Automotive"),
    ("C002", "Bluepeak Construction", "Spain", "Europe", "Construction"),
    ("C003", "Ironleaf Appliances", "Poland", "Europe", "Appliances"),
    ("C004", "Redcliff Motors", "United States", "North America", "Automotive"),
    ("C005", "Granite Bay Builders", "Canada", "North America", "Construction"),
    ("C006", "Sertao Machinery", "Brazil", "South America", "Machinery"),
    ("C007", "Costa Verde Energia", "Brazil", "South America", "Energy"),
    ("C008", "Lakeshore Packaging", "United States", "North America", "Packaging"),
]
PRODUCTS = {"Hot-rolled coil": 640, "Cold-rolled coil": 760, "Galvanized sheet": 880, "Rebar": 590}


def write_segment_kpis() -> Path:
    path = OUT / "segment_kpis.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Segment", "Year", "Sales (USD m)", "Operating income (USD m)", "Employees (FTE)"])
        writer.writerows(SEGMENT_KPIS)
    return path


def write_orders_workbook() -> Path:
    import openpyxl

    rng = random.Random(42)
    workbook = openpyxl.Workbook()
    customers = workbook.active
    customers.title = "customers"
    customers.append(["Customer ID", "Customer name", "Country", "Region", "Industry"])
    for row in CUSTOMERS:
        customers.append(list(row))

    orders = workbook.create_sheet("orders")
    orders.append(["Order ID", "Customer ID", "Order date", "Product", "Quantity (tonnes)", "Revenue (USD)"])
    start = date(2024, 1, 1)
    for number in range(1, 121):
        customer = rng.choice(CUSTOMERS)[0]
        product = rng.choice(list(PRODUCTS))
        quantity = rng.randrange(50, 1500, 10)
        price = PRODUCTS[product] + rng.randrange(-40, 41, 5)
        day = start + timedelta(days=rng.randrange(0, 731))
        orders.append([f"O{number:04d}", customer, day, product, quantity, quantity * price])

    workbook.create_sheet("notes")  # intentionally empty
    path = OUT / "steel_orders_sample.xlsx"
    workbook.save(path)
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for path in (write_segment_kpis(), write_orders_workbook()):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
