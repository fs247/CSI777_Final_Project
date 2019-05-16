# -*- coding: utf-8 -*-
# %% [markdown]
# # Supervised Link Prediction with the Armed Conflict Location Event Database
# In this notebook, I will be turning relational data from the Armed Conflict
# Location & Event Data Project and turning it into several "Conflict"
# Undirected Graphs/Networks.
#
# The nodes in these graphs will be different "agents" in Africa defined as 
# the product of an actor type and the country that they operate in 
# e.g. Civilians - Ghana, Ethnic Militia - Angola
#
# The edges in these graphs will be defined as follows: 1 if there was at least
# one conflict during a time period between two agents and 0 otherwise. 
# The time periods will be different months spanning January 1997 - December 2018.
#
# By using various link prediction/topological
# features measures, I will create features about the edges that exist
# and don't exist within these graphs and I shall set my target
# (what I am trying to predict) as whether or not a conflict occured 
# (represented as an edge existing within the conflict graph) within a certain time frame (a month)
# ACLED (Armed Conflict Location & Event Data Project) is a disaggregated conflict
# analysis and crisis mapping project.
# ACLED collects and analyzes data on locations, dates and types of all reported
# armed conflict and protest events in developing countries.
# It can be found here: https://www.acleddata.com/
# %% [markdown]
# **NOTE**: This analysis has only been completed for Africa
# %% [markdown]
# ### Import utility functions
# %%
from utils.data_read import load_data, load_interaction_codes
from utils.data_cleaning import (
    get_actor_categories, subset_columns, country_extractor)
from utils.feature_engineering import create_agents, get_month
from ConflictGraph import ConflictGraph
from collections import OrderedDict
from itertools import product
from functools import reduce
import pandas as pd
import time
present = True
# %% [markdown]
# ### Load conflict data
# #### The Data Dictionary is as follows:
#
# **ISO**: A numeric code for each individual country <br>
# **EVENT_ID_CNTY**: An individual event identifier by
# number and country acronym. <br>
# **EVENT_ID_NO_CNTY**: An individual event numeric identifier. <br>
# **EVENT_DATE**: Recorded as Year/Month/Day. <br>
# **YEAR**: The year in which an event took place. <br>
# **TIME_PRECISION**: A numeric code indicating the level of certainty of
# the date coded for the event (1-3). <br>
# **EVENT_TYPE**: The type of event. <br>
# **SUB_EVENT_TYPE**: The type of sub-event. <br>
# **ACTOR1**: A named actor involved in the event. <br>
# **ASSOC_ACTOR_1**: The named actor associated with or identifying with
# ACTOR1 in one specific event. <br>
# **INTER1**: A numeric codeindicating the type of ACTOR1. <br>
# **ACTOR2**: The named actor involved in the event. If a dyadic event,
# there will also be an “Actor 1”. <br>
# **ASSOC_ACTOR_2**: The named actor associated with or identifying with
# ACTOR2 in one specific event. <br>
# **INTER2**: A numeric code indicating the type of ACTOR2. <br>
# **INTERACTION**: A numeric code indicating the interaction between types of
# ACTOR1 and ACTOR2.
# Coded  as  an  interaction  between  actor types, and  recorded as lowest joint
# number. <br>
# **REGION**: The region of the world where the event took place. <br>
# **COUNTRY**: The country in which the event took place. <br>
# **ADMIN1**: The largest sub-national administrative region
# in which the event took place.<br>
# **ADMIN2**: The second largest sub-national administrative region 
# in which the event took place. <br>
# **ADMIN3**: The  third largest sub-national administrative region 
# in which the event took place. <br>
# **LOCATION**: The location in which the event took place. <br>
# **LATITUDE**: The latitude of the location. <br>
# **LONGITUDE**: The longitude of the location. <br>
# **GEO_PRECISION**: A numeric code indicating the level of certainty of the
# location coded for the event. <br>
# **SOURCE**: The source(s) used to code the event. <br>
# **SOURCE SCALE**: The geographic scale of the sources used to code the event. <br>
# **NOTES**: A short description of the event. <br>
# **FATALITIES**: Number or estimate of fatalities due to event. <br>
# These are frequently different across reports
# %%
data = load_data('data/data.csv')
# %%
data.info()
# %%
data.head()
# %% [markdown]
# #### Load lookup codes
# The lookup codes for the inter columns to match them to category type
# Each actor has an associated code which represents the type of actor that they are. 
# For example: GIA: Armed Islamic Group is classified as a Rebel Force
# %%
interaction_lookup = load_interaction_codes('data/categorycodes.csv')
# %%
interaction_lookup
# %% [markdown]
# #### Determine the set of all countries in the data set
# %%
countries = set(data.country)
# %% [markdown]
# #### Define all of the possible actor countries
# %%
categories = list(interaction_lookup.Category)
# %% [markdown]
# ## Create a "conflict dataframe"
# Join the interaction lookup to each actor code in order to get the category
# of actor that they are.
# Also, extract the country that each actor belongs from. Conflicts may happen
# in a certain country, but the actors may not come from that country. <br>
# We will be ignoring Associated Actors from this analysis for ease!
# %%
conflict_df = data.pipe(lambda x: get_actor_categories(x, interaction_lookup))\
    .pipe(subset_columns)\
    .pipe(lambda x: country_extractor(x, countries))
# %%
conflict_df.head()
# %% [markdown]
# ## Create a dataframe of all realised conflicts between "Agents"
# This will be a dataframe of all conflicts have actually happened.
# We define an agent as a composite label encompassing an actor's
# country of origin and the actor category.
# This is to ensure that the network has the same amount of nodes
# %%
all_realised_conflicts = conflict_df.pipe(create_agents)\
    .pipe(get_month)
# %%
all_realised_conflicts.head()
# %% [markdown]
# Define a range of monthly time periods
# %%
periods = [str(x)+"-"+str(y) for x, y in
           product(range(1997, 2019), range(1, 13))]
# %% [markdown]
# ## Create Conflict Graphs
# For each time period, create a Conflict Graph with the conflicts that
# happened and didn't happen during that period.
# %%
def make_conflict_graphs(all_realised_conflicts, categories,
                         countries, periods):

    """

    For each time period, create a Conflict Graph with the conflicts that
    happened and didn't happen during that period

    """
    conflict_graphs = OrderedDict()
    counter = 0

    print('Creating Conflict Graphs....')

    for period in periods:
        counter += 1

        if counter % 20 == 0:
            print(str(counter) + " out of " + str(len(periods)))

        conflicts = all_realised_conflicts[
            all_realised_conflicts.period == period]

        cf = ConflictGraph(categories=categories,
                           countries=countries,
                           period=period)

        cf.set_conflicts(conflicts)
        conflict_graphs[period] = cf
    
    print('Conflict Graph Creation Complete')

    return conflict_graphs
# %% [markdown]
# Let's see what metrics have been included by taking a sample
# graph
# %%
conflicts_1997_11 = all_realised_conflicts[
    all_realised_conflicts.period == '1997-11']

cf = ConflictGraph(categories=categories,
                   countries=countries,
                   period='1997-11')

cf.set_conflicts(conflicts_1997_11)
# %% [markdown]
# We have extracted the jaccard coefficient, resource allocation and
# preferential attachment of each potential edge! (the product of all agents)
# %%
cf.get_all_metrics()\
    .sort_values('pref_attachment', ascending=False)\
    .head()
# %% [markdown]
# Lets see what the target dataframe looks like
# %%
cf.get_edge_labels()\
    .sort_values('target', ascending=False)\
    .head()
# %% [markdown]
# ## Explanation of the Features that will be extracted
# $$\Gamma(x) \text{ : The neighbours of the node x. In other words, the nodes that x is connected to.} $$
# $$\left |\Gamma(x)\right| \text{: The size of the neighbour set of x} $$
# $$ \Gamma(x) \cap \Gamma(y) \text{: The shared neighbours of x and y} $$
#
# The toplogical features we will be extracting for each edge are as follows: <br>
#
# ##### Resource Allocation Index with Community Information (Soundarajan_hopcroft):
#
# $$\sum_{w \in \Gamma(u) \cap \Gamma(v)} \frac{f(w)}{|\Gamma(w)|} \text{ where } f(w) = 1 \text{ if } w \text{ is in the same community as } u \text{ and } w \text{ and } 0 \text{ otherwise}$$
#
# ##### Jaccard Coefficient:
#
# $$\frac{\left| \Gamma(x)\cap\Gamma(y) \right|}{\left| \Gamma(x)\cup\Gamma(y) \right|} \text{ for the edge } (x, y)$$
#
# ##### Preferential Attachment
# $$\left| \Gamma(x) \right| \times \left| \Gamma(y) \right| \text{ for the edge } (x, y)$$
# %%
del data, conflict_df
# %% [markdown]
# # Feature Extraction
# The aim is: <br>
# For each 1 month window of time where conflicts have happened,
# we will extract a dataframe where the target is whether or not a
# conflict/edge existed during that time frame and we will extract the features
# are link prediction measures about the emerging graphs
# (a representation of all the interaction between agents up to a certain time) 
# up to 10 months
# in the past in 1 month windows
# %%
def full_merge(features, target):

    """

    Merge the dataframes with the link features to the
    link target (absence/presence)

    """

    result = reduce(
        lambda x, y: x.merge(y, on=["agent1", "agent2"]),
        features)
    result = result.merge(target, on=["agent1", "agent2"])
    return result
# %%
def make_training_data(graphs, n_prev=12):

    """

    Implement a sliding window approach for extracting features
    and targets

    Example:
    {G1, G2, .... G12} -> G13
    {G2, G3, .... G13} -> G14

    Keyword Arguments:
    graphs -- a dictionary of Conflict Graphs
    n_prev -- The amount of time periods to slide the window
              by


    """

    keys = list(graphs.keys())
    indexes = range(len(graphs)-n_prev)

    train_dfs = []

    start = time.time()

    print('Creating Training Data...')
    for index in indexes:

        if index % 10 == 0:
            print(str(index) + " out of " + str(len(indexes)))
            print(str(time.time() - start) + " seconds elapsed")

        # Select n_prev graphs in a sliding window for feature extraction
        selected_keys_X = keys[index:index+n_prev][::-1]
        # Select a graph with a period 1 after the final graph chosen in
        # selected_keys_X
        selected_keys_Y = keys[index+n_prev]

        # Extract features/metrics for each period selected in
        # selected_keys_X
        features = (graphs[key].get_all_metrics(lag=idx+1)
                    for (idx, key) in enumerate(selected_keys_X))

        # Get the target labels for a graph 1 time period after the
        # features
        target = graphs[selected_keys_Y].get_edge_labels()

        # Merge
        result = full_merge(features, target)

        train_dfs.append(result)

        del features, target, result, selected_keys_X, selected_keys_Y

    print('Training Data Created!')

    return train_dfs
# %% [markdown]
# # Creation of the Training Data via Graph Feature Extraction
# 1. Create all of the conflict graphs from 1997-1 to 2018-12
# 2. Apply a sliding window approach to get the edge labels for a certain period
# and then the graph features for the previous 12 periods
# 3. Concatenate the output from all of the sliding windows
# 4. Save as a parquet for the modelling phase
# %%
if not present:
    conflict_graphs = make_conflict_graphs(all_realised_conflicts,
                                           categories,
                                           countries,
                                           periods)
    train = make_training_data(conflict_graphs)
    train_df = pd.concat(train, ignore_index=True)
    train_df.to_parquet('df.parquet.gzip', compression='gzip')
# %%
if present:
    train_df = pd.read_parquet('df.parquet.gzip')
    train_df.head()
