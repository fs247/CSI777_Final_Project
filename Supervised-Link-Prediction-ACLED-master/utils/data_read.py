import pandas as pd

def load_data(path):
    data = pd.read_csv(path, index_col="data_id")
    data["event_date"] = pd.to_datetime(data["event_date"])
    return data


def load_interaction_codes(path):
    data = pd.read_csv(path)
    return data

