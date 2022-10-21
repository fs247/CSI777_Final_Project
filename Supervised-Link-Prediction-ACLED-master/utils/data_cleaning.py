import pandas as pd
import numpy as np
import re

def rename_category_columns(actor_category_df):
    return actor_category_df\
        .rename(columns={'Category_x': 'actor1_category'})\
        .rename(columns={'Category_y': 'actor2_category'})


def get_actor_categories(conflict_df, interaction_lookup):
    actor_category_df = conflict_df.copy
    actor_category_df = conflict_df\
                            .merge(interaction_lookup, 
                                   left_on="inter1", 
                                   right_on="code")\
                            .merge(interaction_lookup, 
                                   left_on="inter2", 
                                   right_on="code")
    return rename_category_columns(actor_category_df)


def subset_columns(conflict_df):
    return conflict_df[["event_date",
                        "country",
                        "actor1",
                        "actor1_category",
                        "actor2",
                        "actor2_category"]]


def get_country(string, countries):
    for country in countries:
        if country in string:
            return country

def country_extractor(conflict_df, countries):
    df = conflict_df.copy()
    df["actor1_country_temp"] = df["actor1"]\
        .apply(lambda x: get_country(x, countries))
    df["actor2_country_temp"] = df["actor2"]\
        .apply(lambda x: get_country(x, countries))

    df["actor1_country"] = np.where(df["actor1_country_temp"].isnull(),
                                    df["country"],
                                    df["actor1_country_temp"])

    df["actor2_country"] = np.where(df["actor2_country_temp"].isnull(),
                                    df["country"],
                                    df["actor2_country_temp"])

    return df[["event_date", "country", 
               "actor1", "actor1_country", "actor1_category", 
               "actor2", "actor2_country", "actor2_category"]]
    
    
    
    

    