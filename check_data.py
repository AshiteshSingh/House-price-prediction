import pandas as pd
import glob
for f in glob.glob(r'c:\Users\ashitesh\Desktop\house price\PanTrainModel_data1\*.csv'):
    try:
        df = pd.read_csv(f, nrows=1, on_bad_lines='skip')
        print(f"--- {f.split(chr(92))[-1]} ---")
        print(f"Columns ({len(df.columns)}): {df.columns.tolist()}")
    except Exception as e:
        print(f"Error reading {f}: {e}")
