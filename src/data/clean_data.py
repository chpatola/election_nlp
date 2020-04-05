import pandas as pd
import numpy as np
from src import Generic_func as gf

# 1 LOAD, INSPECT AND CLEAN DATA
yle_data = pd.read_csv('/home/chpatola/Desktop/Skola/Python/cookie_nlp/data/raw/vastauksetavoimenadatana1.csv', sep=';', encoding="ISO-8859-1") 
yle_data = yle_data.filter(['puolue','MitÃ¤ asioita haluat edistÃ¤Ã¤ tai ajaa tulevalla vaalikaudella'])

gf.nyName(yle_data,'MitÃ¤ asioita haluat edistÃ¤Ã¤ tai ajaa tulevalla vaalikaudella','work_for')
gf.nyName(yle_data,'puolue','party')
yle_data = gf.scandinavian_letters(yle_data)

#2 DROP NA rows and party with n < 100
yle_data.dropna(axis=0,inplace=True)
print(yle_data.groupby("party").count())
filtered= yle_data.groupby('party').filter(lambda x: len(x) >= 150)
yle_data_cut = yle_data[yle_data['party'].isin(filtered.party)]
print(yle_data_cut.groupby("party").count())

#3 WRITE CLEANED DATA TO FOLDER
yle_data_cut.to_csv("/home/chpatola/Desktop/Skola/Python/cookie_nlp/data/interim/cleaned_data.csv",index=False, encoding="ISO-8859-1")


