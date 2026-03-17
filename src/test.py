import os
import pandas as pd


def main():
    profile = 'UMCGCohortsStaging'
    path = '/Users/p.jansma/Downloads/molgenis-emx2/data/_models/shared'

    data_model = get_data_model(path, profile)

    print(data_model.head(20).to_string())

    output_csv = f'molgenis_{profile}.csv'
    output_xlsx = f'molgenis_{profile}.xlsx'

    data_model.to_csv(output_csv, index=False)
    data_model.to_excel(output_xlsx, index=False)

    print("CSV saved to:", os.path.abspath(output_csv))
    print("Excel saved to:", os.path.abspath(output_xlsx))

    os.system(f'open "{output_xlsx}"')


def get_data_model(path, profile):
    data_model = pd.DataFrame()

    for file_name in os.listdir(path):
        if not file_name.endswith('.csv'):
            continue

        file_path = os.path.join(path, file_name)
        df = pd.read_csv(file_path, keep_default_na=False, dtype='object')

        if 'profiles' not in df.columns:
            continue

        df = df.loc[df['profiles'].apply(lambda p: profile in str(p).split(','))]
        data_model = pd.concat([data_model, df], ignore_index=True)

    return data_model


if __name__ == '__main__':
    main()