import pandas as pd
import glob
with open('data_columns_utf8.txt', 'w', encoding='utf-8') as f:
    for file in glob.glob(r'c:\Users\ashitesh\Desktop\house price\PanTrainModel_data1\*.csv'):
        try:
            df = pd.read_csv(file, nrows=1, on_bad_lines='skip')
            f.write(f"--- {file.split(chr(92))[-1]} ---\n")
            f.write(f"Columns ({len(df.columns)}): {df.columns.tolist()[:30]} ... and more if any\n")
            if 'Price_INR' in df.columns:
                 f.write("Target: Price_INR found.\n")
            elif 'price' in df.columns:
                 f.write("Target: price found.\n")
        except Exception as e:
            f.write(f"Error {file}: {e}\n")
