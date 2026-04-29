# CSV File Generation

A Streamlit web application for generating structured CSV files for a high-throughput thin-film sample database. The app covers the full sample lifecycle: substrate registration, fabrication process logging, post-deposition treatment, and spectral data export.

## Requirements

- Python 3.9 or later
- Dependencies listed in `requirements.txt`:

```
streamlit
pandas
numpy
plotly
openpyxl
xlrd>=2.0.1
```

Install with:

```bash
pip install -r requirements.txt
```

## Running the application

```bash
streamlit run csv_file_generation.py
```

## Tabs

### 1. Substrate Generation

Registers one or more physical substrates and produces a CSV file per substrate.

**Substrate Number field** accepts two formats:

| Format | Example | Result |
|--------|---------|--------|
| Single substrate | `3716-15` | one file: `3716-15` |
| Range | `3716-1 to 30` | thirty files: `3716-01` ... `3716-30` |

Range suffixes are zero-padded to the digit width of the end number (end = 30 gives two-digit padding).

- Single substrate: generates one CSV with an inline download button and a content preview.
- Multiple substrates: bundles all CSV files into a ZIP archive (`substrates.zip`) for a single download.

Fields captured: substrate type, production batch, vendor, manufacturer, material properties (thickness, size, materials, softening point, expansion coefficient), cleaning details (method, description, duration, temperature, pressure, date, time), institution, and operator.

### 2. Fabrication Generation

Logs a deposition or growth step on a substrate. Select the fabrication method from the dropdown; the form adapts to show only the relevant parameters.

Supported methods:

| Method | Key parameters |
|--------|---------------|
| PVD-J | process number, recipe, box type, duration, substrate temperature, cooling temperature, holding time, deposition rate, power, tooling factor, crystal (xtal), sample orientation, mass before/after |
| Sputtering | program, duration, power, current, voltage, gas mix, process pressure, pre-fab pressure, notes |
| Tube Furnace | temperature, ramp rate, selenium/sulfur amounts, pressure, humidity, duration, cooling time, storage days, sample orientation, weight before/after |
| RTP | pressure, box type, selenium/sulfur amounts, steps, recipe, ramp rate, holding time, orientation, weight before/after |
| PLD | pre-ablation parameters (shots, frequency, fluence, gas pressure, gas type, duration) and deposition parameters (temperature, shots, frequency, fluence, gas pressure, gas type, duration) |
| PVD-P | upload an existing PVD-P CSV; the app extracts metadata automatically and re-formats the filename and operator field |

Common fields for all non-PVD-P methods: substrate number, institution, operator, fabrication sequence, date, time.

### 3. Treatment Generation

Records post-deposition treatment. Select the treatment method from the dropdown.

Supported methods:

| Method | Sequence | Notes |
|--------|----------|-------|
| As-deposited | fixed at 0 | default values set to room conditions |
| Annealing | user-defined | temperature, duration, gas, humidity, pressure |
| Storing-in-Glovebox | user-defined | environmental parameters |
| Storing-out-Glovebox | user-defined | environmental parameters |

Common fields: substrate number, institution, operator, sequence, date, time, treatment place, temperature, duration, humidity, oxygen level, gas type, pressure.

### 4. XRF & SPX Generation

Processes spectral measurement data from two uploaded sources:

- XRF results in XLS/XLSX format
- SPX spectrum files packaged in a ZIP archive

The tab reads both files, matches SPX spectra to XRF entries, and generates the corresponding CSV output for database import.

## File naming convention

Generated filenames follow a consistent pattern per record type:

```
<substrate_number>_<institution>_<operator>_substrate_<type>_<datetime>.csv
<substrate_number>_<institution>_<operator>_fab<sequence>_<method>_<datetime>.csv
<substrate_number>_<institution>_<operator>_treat<sequence>_<method>_<datetime>.csv
```

Datetime stamps use `YYYYMMDD_HHMMSS` format. PVD-P files use the date and time extracted from the uploaded source file instead of the current time.

## Operator name validation

All operator fields require both a first name and a last name (e.g., `Steinkopf Lars`). The form will not generate a file if this constraint is not met.

## Time format

All time inputs use 24-hour format (`HH:MM:SS`). Values are stored in 12-hour AM/PM format inside the generated CSV.
