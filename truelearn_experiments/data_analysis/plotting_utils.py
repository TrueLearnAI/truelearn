import itertools
import networkx as nx

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

ML_TOPICS_MAPPING = {21369: "ML",
                     30563: "RL",
                     966: "Algo",
                     7285: "CV",
                     28762: "Prob",
                     24183: "NLP",
                     40522: "W2V"}

ML_TOPIC_SET = frozenset(ML_TOPICS_MAPPING.keys())


def is_topic_eligible(user_topics):
    inter = set(user_topics).intersection(ML_TOPIC_SET)
    return bool(len(inter) > 4 and 7285 in inter and 24183 in inter)


def do_graph_analysis(user_profile, mapping):
    mapping = {
        ("RL", "Prob"): .45,
        ("RL", "CV"): .61,
        ("RL", "Algo"): .14,
        ("RL", "NLP"): .7,
        ("RL", "ML"): .81,
        ("Prob", "W2V"): .3,
        ("Prob", "Algo"): .23,
        ("Prob", "NLP"): .3,
        ("Prob", "ML"): .7,
        ("CV", "W2V"): .7,
        ("CV", "ML"): .61,
        ("W2V", "NLP"): .92,
        ("W2V", "ML"): .4,
        ("Algo", "ML"): .4,
        ("NLP", "ML"): .55
    }
    topic_to_id_mapping = {topic: id for id, topic in ML_TOPICS_MAPPING.items()}
    user_trimmed = {}
    missing_topics = set()

    for topic in ML_TOPIC_SET:
        try:
            user_trimmed[ML_TOPICS_MAPPING[topic]] = user_profile[topic]
        except KeyError:
            missing_topics.add(ML_TOPICS_MAPPING[topic])

    for topic, dist in user_trimmed.items():
        mu = dist[0]
        sigma = np.sqrt(dist[1])

        print(topic)

        sample = np.random.normal(mu, sigma, 100000)
        ax = sns.displot(sample, kind="kde", fill=True)

        plt.show()

    nodes = list(set(user_trimmed.keys()).union(missing_topics))

    combos = itertools.combinations(nodes, 2)

    # build graph
    g = nx.Graph()
    g.add_nodes_from(nodes)

    edges = []
    for src, dst in combos:
        rel = mapping.get((src, dst), mapping.get((dst, src), None))
        if rel is not None:
            g.add_edge(src, dst, weight=rel)
            print(rel)

    # node sizes
    exl = [(u, v) for (u, v, d) in g.edges(data=True) if d["weight"] >= 0.8 and u != "W2V" and v != "W2V"]
    elarge = [(u, v) for (u, v, d) in g.edges(data=True) if 0.8 > d["weight"] >= 0.6 and u != "W2V" and v != "W2V"]
    emed = [(u, v) for (u, v, d) in g.edges(data=True) if 0.6 > d["weight"] >= 0.4 and u != "W2V" and v != "W2V"]
    esmall = [(u, v) for (u, v, d) in g.edges(data=True) if 0.4 > d["weight"] >= 0.2 and u != "W2V" and v != "W2V"]
    exsmall = [(u, v) for (u, v, d) in g.edges(data=True) if 0.2 > d["weight"] >= .1 and u != "W2V" and v != "W2V"]

    wexl = [(u, v) for (u, v, d) in g.edges(data=True) if d["weight"] >= 0.8 and (u == "W2V" or v == "W2V")]
    welarge = [(u, v) for (u, v, d) in g.edges(data=True) if 0.8 > d["weight"] >= 0.6 and (u == "W2V" or v == "W2V")]
    wemed = [(u, v) for (u, v, d) in g.edges(data=True) if 0.6 > d["weight"] >= 0.4 and (u == "W2V" or v == "W2V")]
    wesmall = [(u, v) for (u, v, d) in g.edges(data=True) if 0.4 > d["weight"] >= 0.2 and (u == "W2V" or v == "W2V")]

    pos = nx.spring_layout(g)

    options = {"node_size": 1000, "alpha": 0.7}

    nx.draw_networkx_nodes(g, pos, node_color="b", **options)

    # edges
    nx.draw_networkx_edges(g, pos, edgelist=exl, edge_color="g", width=6)
    nx.draw_networkx_edges(g, pos, edgelist=elarge, edge_color="g", width=5)
    nx.draw_networkx_edges(g, pos, edgelist=emed, edge_color="g", width=4)
    nx.draw_networkx_edges(g, pos, edgelist=esmall, edge_color="g", width=3)
    nx.draw_networkx_edges(g, pos, edgelist=exsmall, edge_color="g", width=1)

    nx.draw_networkx_edges(g, pos, edgelist=wexl, edge_color="orange", style="dashed", alpha=0.5, width=6)
    nx.draw_networkx_edges(g, pos, edgelist=welarge, edge_color="orange", style="dashed", alpha=0.5, width=5)
    nx.draw_networkx_edges(g, pos, edgelist=wemed, edge_color="orange", style="dashed", alpha=0.5, width=4)
    nx.draw_networkx_edges(g, pos, edgelist=wesmall, edge_color="orange", style="dashed", alpha=0.5, width=3)

    print()
    nx.draw_networkx_labels(g, pos, font_size=20, font_family="sans-serif")
