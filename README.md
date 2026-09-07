Vial Temperature Viewer

A simplified Streamlit interface for reading vial temperature data from the
RheaVita SQL Server databases.

Install

pip install -r requirements.txt

You also need a Microsoft SQL Server ODBC driver installed on Windows.

Recommended:

ODBC Driver 18 for SQL Server

ODBC Driver 17 for SQL Server

Run

Open Anaconda Prompt / Command Prompt in this folder:

streamlit run app.py

Main functions

SQL Server connection

Database selection

Range or individual vial selection

DeviceData include/exclude filtering

Downsampling

Rolling mean

Rolling median

Robust rolling mean with MAD filtering

Savitzky-Golay smoothing

Align by minimum temperature

Align by temperature threshold

Manual offset alignment

Mean profile and ±1 SD

Adjustable axes and setpoints

PNG export

CSV export
