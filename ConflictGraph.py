from networkx import Graph
from functools import reduce, lru_cache
import networkx as nx
import pandas as pd


class ConflictGraph(Graph):

    """

    A Graph/Network of conflicts within a certain time period

    """

    def __init__(self, categories, countries, period):
        super().__init__()

        self.categories = categories
        self.countries = countries
        self.period = period

    def __sort_tuple(self, tup):

        """

        Return a tuple after sorting a tuple

        The default behaviour of sorted() returns a list

        """

        return tuple(sorted(tup))

    @lru_cache(maxsize=2)
    def __create_all_possible_nodes(self):

        """

        Create all possible nodes in the network via the product of
        the actor categories and the actor countries

        """

        return [(x + '-' + y) for x in self.categories for y in self.countries]

    @lru_cache(maxsize=2)
    def __get_node_countries(self):
        return {x + '-' + y: y for x in self.categories
                for y in self.countries}

    @lru_cache(maxsize=2)
    def __create_all_possible_edges(self):

        """

        Generate all possible conflicts between agents via permutation

        """

        # Get all nodes for each timestep
        all_potential_nodes = self.__create_all_possible_nodes()

        # Create all possible combinations of nodes as tuples
        all_possible_conflicts = ((x, y) for x in all_potential_nodes
                                  for y in all_potential_nodes)

        # Order each tuple
        all_possible_conflicts_ordered = [self.__sort_tuple(edge) for edge
                                          in all_possible_conflicts]
        # Drop duplicate tuples
        all_possible_conflicts_no_dups = set(all_possible_conflicts_ordered)

        # Convert to dataframe
        all_possible_edges = pd.DataFrame(all_possible_conflicts_no_dups)
        all_possible_edges.columns = ['agent1', 'agent2']

        return all_possible_edges

    def __create_edges(self):

        """

        Create the Graph's "positive" edges (to be imported into a nx.Graph)
        Positive meaning that there was a conflict

        Only looks at the first conflict within the data as the target
        is to predict if at least one conflict will happen in the time
        period

        """

        if len(self.conflicts) == 0:
            return []

        edges = self.conflicts.copy()
        edges = edges.groupby(['agent1', 'agent2'])\
                     .first()\
                     .reset_index()

        # Return as list of tuples
        return edges[["agent1", "agent2"]]\
            .to_records(index=False)\
            .tolist()

    @lru_cache(maxsize=12)
    def get_edge_labels(self):

        """

        Create all edge labels, positive and negative

        Caching of 12 previous computations has been implemented

        """

        all_possible_edges = self.__create_all_possible_edges()

        positive_edges = pd.DataFrame(self.__create_edges())
        positive_edges.columns = ["agent1", "agent2"]

        # Edges that exist get a label of 1
        positive_edges["target"] = 1

        # Join on all possible edges, those with no match, get a 0 label
        edges = positive_edges.merge(all_possible_edges,
                                     how="right",
                                     on=["agent1", "agent2"])\
                              .fillna(0)

        edges["period"] = self.period

        return edges

    def get_nodes(self):
        return self.__create_all_possible_nodes()

    def get_edges(self):
        return self.__create_edges()

    def _add_nodes(self):

        """

        Add all possible agents as nodes to the Graph

        """

        nodes = self.__create_all_possible_nodes()
        self.add_nodes_from(nodes)

    def _set_node_attributes(self):
        values = self.__get_node_countries()
        nx.set_node_attributes(self, name='country', values=values)

    def _add_edges(self):

        """

        Add edges (extracted from the conflict dataframe) to
        the Graph

        """

        edges = self.__create_edges()
        if len(edges) == 0:
            pass
        else:
            self.add_edges_from(edges)

    def set_conflicts(self, conflicts):

        """

        Initialise the conflict graph with a dataframe of conflicts
        Add Nodes and Edges to the Graph

        Keyword Arguments:
        conflicts -- A dataframe of ACLED Conflicts

        """

        self.conflicts = conflicts
        self.all_possible_edges = self.__create_all_possible_edges()\
            .to_records(index=False)\
            .tolist()

        self._add_nodes()
        self._set_node_attributes()
        self._add_edges()

    def __set_suffix(self, lag=None):

        """

        Create a suffix to add onto the end of
        a feature column to represent the lag it has
        relative to the target columns

        """

        suffix = ""
        if lag:
            suffix = "_"+str(lag)+"periods_prev"

        return suffix

    def get_preferential_attachment(self, lag=None):

        """

        Extract the preferential attachment scores for all
        possible edges in this graph

        """

        suf = self.__set_suffix(lag)

        metric = nx.preferential_attachment(self, self.all_possible_edges)
        metric_df = pd.DataFrame(metric)
        metric_df.columns = ["agent1", "agent2", "pref_attachment"+suf]
        return metric_df

    def get_resource_allocation(self, lag=None):

        """

        Extract the resource allocation (with community info) 
        scores for all possible edges in this graph

        """

        suf = self.__set_suffix(lag)

        metric = nx.ra_index_soundarajan_hopcroft(self,
                                                  self.all_possible_edges,
                                                  community='country')
        metric_df = pd.DataFrame(metric)
        metric_df.columns = ["agent1", "agent2", "resource_alloc_com"+suf]
        return metric_df

    def get_jaccard_coefficient(self, lag=None):

        """

        Extract the jaccard coefficient
        scores for all possible edges in this graph

        """

        suf = self.__set_suffix(lag)

        metric = nx.jaccard_coefficient(self, self.all_possible_edges)
        metric_df = pd.DataFrame(metric)
        metric_df.columns = ["agent1", "agent2", "jaccard_coef"+suf]
        return metric_df

    @lru_cache(maxsize=24)
    def get_all_metrics(self, lag=None):

        """

        Extract the preferential attachment, resource allocation
        and jaccard coefficient scores for each potential edge.

        Caching of 12 previous computations has been implemented

        Keyword Arguments:
        lag -- A note/flag of how many periods prior to the edge label
        that these features are.

        """

        pref_attachment = self.get_preferential_attachment(lag)
        resource_allocation = self.get_resource_allocation(lag)
        jaccard_coefficient = self.get_jaccard_coefficient(lag)

        metrics = [pref_attachment, resource_allocation, jaccard_coefficient]

        on = ["agent1", "agent2"]

        all_metrics = reduce(
            lambda x, y: pd.merge(x, y, how='inner', on=on), metrics)

        return all_metrics
