import pandas as pd

class Util:

    def __init(self):
        pass

    def get_file(self, csv):
        file = pd.read_csv(csv)
        return file

    def convert_list(self, df, params):
        df_list = list(df[params])
        return df_list

    def high_or_low(self, df_list):
        sst_result = []
        for params in df_list:
            if params < -1:
                sst_result.append(-1)
            elif 1 <= params < 1.5:
                sst_result.append(1)
            elif params > 1.5:
                sst_result.append(2)
            else:
                sst_result.append(0)
        return sst_result
