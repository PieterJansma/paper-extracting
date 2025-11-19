import pandas as pd

df = pd.read_excel('/Users/p.jansma/Documents/GitHub/paper-extracting/UMCG_Resources1763456291774.xlsx')

# Haal alleen de kolomnamen
labels = pd.DataFrame(df.columns, columns=["column_name"])

# Schrijf naar Excel
labels.to_excel('column_labels.xlsx', index=False)

print("Kolomnamen opgeslagen in column_labels.xlsx")
