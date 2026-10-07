from metropolis import initial_convergence, sample_different_groupings
import networkx as nx

if __name__ == "__main__":
    G = nx.read_gml("foodweb_undirected.gml")

    number_of_groups = 3

    weight, groupings = initial_convergence(nx.to_numpy_array(G), number_of_groups, 100)

    accepted_groupings, accepted_groupings_weights = sample_different_groupings(
        nx.to_numpy_array(G),
        weight,
        groupings,
        number_of_groups,
        100000,
        0.7,
        number_of_changes=20)

    print(f'Number of accepted groupings {len(accepted_groupings)}')

