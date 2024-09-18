import os

import pandas
import pandas as pd
import numpy as np
from itertools import groupby
import re


class General:
    def __init__(self, txt_dir, wav_dir):
        self.txt_dir = txt_dir
        self.wav_dir = wav_dir

    def data(self, divide_month, month, cuts, rct):
        general_data, txt_months, wav_months, txt_common, wav_common = pd.DataFrame(), [], [], [], []
        no_data_in = []
        txt_list = [txt for txt in os.listdir(self.txt_dir) if txt.endswith('.txt')]
        wav_list = [wav for wav in os.listdir(self.wav_dir)]

        if divide_month:
            month_txt = []
            for txt in txt_list:
                if txt.startswith(month):
                    month_txt.append(txt)
            for i in month_txt:
                for j in wav_list:
                    if i[0:31] == j[0:31]:
                        txt_common.append(i)
                        wav_common.append(j)
        else:
            for i in txt_list:
                for j in wav_list:
                    if i[0:31] == j[0:31]:
                        txt_common.append(i)
                        wav_common.append(j)

        # print("PRELIMINARY DATA CHECK FOR ALL ANNOTATION FOLDER:")
        txt_line_sum = 0
        csv_line_sum = 0
        for i, txt in enumerate(txt_common):
            with open(self.txt_dir + txt, 'r') as file:
                line_count = sum(1 for line in file)
            txt_line_sum += line_count
            data_original = pd.read_csv(self.txt_dir + txt, sep="\t", index_col=0, on_bad_lines="skip")
            csv_line_sum += len(data_original)

            # CORRECT COMMON MISSPELLINGS IN COLUMN NAMES
            if data_original.columns[0] == 'Selection':
                data_original = data_original.drop(columns=['Selection'])

            replacement_dict = {6: "Call Type", 7: 'Localization'}
            col_names = data_original.columns.tolist()
            col_names_new = col_names

            for idx, new_name in replacement_dict.items():
                if idx < len(col_names):
                    col_names_new[idx] = new_name
            data_original.columns = col_names_new

            # FLAG A FRAME THAT HAS NO DATA IN IT
            if len(data_original) == 0:
                no_data_in.append(txt)
            else:
                data = data_original[["Channel", "Begin Time (s)", "Call Type", "Localization"]]
                data.insert(1, 'WAV', [wav_common[i] for n in range(data.shape[0])], True)
                data = data.reset_index(drop=True)
                if i == 0:
                    general_data = data
                else:
                    general_data = pd.concat([general_data, data], ignore_index=True)

        # SORT BY DATE: convert WAV column to usable datetime variable column
        general_data.insert(0, 'DateTime', general_data['WAV'].apply(lambda x: x[:11]), True)
        general_data['DateTime'] = pd.to_datetime(general_data['DateTime'], format='%Y%m%d-%H')
        general_data = general_data.sort_values(by='DateTime', ascending=True)

        # REPETITION CHECKING: gets rid of localizations with too little or too many repetitions
        unique_chunks = general_data['WAV'].unique()
        gen_data = pandas.DataFrame()
        for chunk in unique_chunks:
            using = general_data[general_data['WAV'] == chunk]
            count = [len(list(c)) for i, c in groupby(np.ndarray.tolist(using["Localization"].values))]
            using.insert(5, 'Repetition', sum(([c] * c for c in count), []), True)
            gen_data = pd.concat([gen_data, using], ignore_index=True, sort=False)
        presence_data = gen_data
        gen_data = gen_data[(gen_data['Repetition'] > 2) & (gen_data['Repetition'] < 5)]
        gen_data['Call Type'] = gen_data['Call Type'].apply(lambda x: int(x))
        gen_data = gen_data.sort_values(by=['WAV', 'Localization'], ascending=[True, True])

        # CHECK LENGTHS AS FILE IS FORMATTED
        print("No data found in: \n", no_data_in)
        print("\nNumber of lines in all the txt files is", txt_line_sum)
        print("Length of csv file after dropping bad lines is", csv_line_sum)
        print("Length of general_data after removing repetitions <2 or >5 is", (gen_data.shape[0]))

        gen_data = gen_data.reset_index(drop=True)
        presence_data = presence_data.reset_index(drop=True)

        # GROUP BY DATE THEN LOCALIZATION TO GET CORRECT SUBSETS FOR ANALYSIS
        grouped = gen_data.groupby(['WAV', 'Localization'])
        group_sizes = grouped.size().reset_index(name='NumRows')

        counts = group_sizes['NumRows'].values
        species = gen_data.drop_duplicates(subset=["Localization", "Call Type"], keep='last')["Call Type"].values
        return gen_data, counts, species, presence_data
