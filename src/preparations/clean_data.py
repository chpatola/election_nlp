"""Step 1: load and clean data"""
from os import path
import pandas as pd
from src import Generic_func as gf

def clean_data(base_path):
    # 1 LOAD, INSPECT AND CLEAN DATA
    yle_data = pd.read_csv(
    path.join(base_path,'data/raw/vastauksetavoimenadatana1.csv'),
    sep=';',
    encoding="ISO-8859-1"
    )

    yle_data = yle_data.filter(
        ['puolue',
        'MitÃ¤ asioita haluat edistÃ¤Ã¤ tai ajaa tulevalla vaalikaudella']
        )

    gf.nyName(
        yle_data,
        'MitÃ¤ asioita haluat edistÃ¤Ã¤ tai ajaa tulevalla vaalikaudella',
        'work_for'
        )

    gf.nyName(
        yle_data,
        'puolue',
        'party'
        )

    yle_data = gf.scandinavian_letters(yle_data)

    # 2 DROP NA rows, party with n < 100 and two swedish answers

    yle_data.dropna(axis=0, inplace=True)

    filtered = yle_data.groupby('party').filter(lambda x: len(x) >= 150)

    yle_data_cut = yle_data[yle_data['party'].isin(filtered.party)]

    yle_data_cut = yle_data_cut[~yle_data_cut["work_for"].str.contains(
        "löneklyftor|Kommunikationer")]

    # 3 WRITE CLEANED DATA TO FOLDER

    yle_data_cut.to_csv(
        path.join(base_path,'data/interim/cleaned_data.csv'),
        index=False, encoding="ISO-8859-1"
        )
