import pandas as pd


def create_agents(conflict_df):
    df = conflict_df.copy()

    df['agent1'] = df['actor1_category'] + '-' + df['actor1_country']
    df['agent2'] = df['actor2_category'] + '-' + df['actor1_country']

    return df[["event_date", "agent1", "agent2"]]


def get_month(conflict_df):
    df = conflict_df.copy()
    df["period"] = df.event_date.dt.year.astype(str)\
      + "-" + df.event_date.dt.month.astype(str)
    return df
