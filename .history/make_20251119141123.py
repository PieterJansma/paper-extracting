import pandas as pd

df = pd.read_excel('/Users/p.jansma/Documents/GitHub/paper-extracting/UMCG_Resources1763456291774.xlsx')

# Maak dataframe met kolomnamen + lege 'gedaan' kolom
labels = pd.DataFrame({
    "column_name": df.columns,
    "gedaan": ""
})

# Schrijf naar Excel
labels.to_excel('column_labels.xlsx', index=False)

print("Kolomnamen + lege 'gedaan' kolom opgeslagen in column_labels.xlsx")
