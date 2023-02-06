import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

import random
import itertools
from pprint import pprint


def intOrEmpty(x):
    try:
        return int(x)
    except Exception as e:
        return ""

def keep_only_full_families(G):
    G = G.copy()

    prev = (0, 0)
    curr = (len(G.nodes()), len(G.edges()))
    if (prev != curr):
        # remove parent-child relation if child don't have both parents
        for e in [e for e in G.edges() if len(G.in_edges(e[1])) == 1]:
            G.remove_edge(*e)
        
        # remove nodes withot relation
        for n in [n for n in G.nodes() if len(G.in_edges(n)) + len(G.out_edges(n)) == 0]:
            G.remove_node(n)

        prev = curr
        curr = (len(G.nodes()), len(G.edges()))

    return G

def count_triplets(G):
    fms = 0
    fmd = 0
    for n, data in [(n, data) for n, data in G.nodes(data=True) if len(G.in_edges(n)) == 2]:
        if data["sex"] == "M":
            fms += 1
        else:
            fmd += 1
    return fms, fmd

def remove_less_represented_nodes(G, lim=1):
    G = keep_only_full_families(G)
    for n in [n for n, data in G.nodes(data=True) if data["n_images"] < lim]:
        G.remove_node(n)
    G = keep_only_full_families(G)

    return G

def plot_graph(G, pdf_file_path):
    COLOR_MALE = "#34aeeb"
    COLOR_FEMALE = "#de3e7e"

    node_colors = [COLOR_MALE if n[1]["sex"] == "M" else COLOR_FEMALE for n in G.nodes(data=True)]
    pos = nx.drawing.nx_agraph.graphviz_layout(G, prog='dot')
    labels = {n[0]: f"{n[0]}\n{n[1]['name']}" for n in G.nodes(data=True)}

    plt.clf()
    plt.figure(num=None, figsize=(200, 1.5), frameon=False, clear=True, tight_layout=True)

    nx.draw_networkx_nodes(G, pos=pos, node_color=node_colors, node_size=50, linewidths=0.0, alpha=0.5)
    nx.draw_networkx_edges(G, pos=pos, node_size=50, width=0.1, arrowsize=5, style="dashed", alpha=1)
    nx.draw_networkx_labels(G, pos=pos, labels=labels, font_size=2)

    plt.tight_layout()
    plt.savefig(pdf_file_path, format="pdf", orientation="landscape")
    plt.close()


def main():
    persons_df = pd.read_csv(".cache/nokdb/nokdb-persons.csv")
    persons_df["father_pid"] = persons_df["father_pid"].map(intOrEmpty)
    persons_df["mother_pid"] = persons_df["mother_pid"].map(intOrEmpty)

    images_df = pd.read_csv(".cache/nokdb/nokdb-images.csv")

    images_per_person_df = images_df.groupby("pid", as_index=False)["iid"].count().rename(columns={"iid": "n_images"})
    persons_df = pd.merge(persons_df, images_per_person_df, on="pid")


    persons_df.head()


    G = nx.DiGraph()

    # Add nodes
    for index, row in persons_df.iterrows():
        G.add_node(int(row["pid"]), sex=row["sex"], n_images=row["n_images"], name=row["name"], pid=int(row["pid"]))

    # Add edges
    for index, row in persons_df.iterrows():
        if row["father_pid"]:
            G.add_edge(int(row["father_pid"]), int(row["pid"]))
        if row["mother_pid"]:
            G.add_edge(int(row["mother_pid"]), int(row["pid"]))

    # Remove nodes without data
    for n in [n for n, data in G.nodes(data=True) if "name" not in data or "sex" not in data or "n_images" not in data or "pid" not in data]:
        G.remove_node(n)


    families = [
        (1, count_triplets(remove_less_represented_nodes(G, 1))),
        (2, count_triplets(remove_less_represented_nodes(G, 2))),
        (3, count_triplets(remove_less_represented_nodes(G, 3))),
        (4, count_triplets(remove_less_represented_nodes(G, 4))),
        (5, count_triplets(remove_less_represented_nodes(G, 5))),
        (6, count_triplets(remove_less_represented_nodes(G, 6))),
    ]

    for lim, (fms, fmd) in families:
        plt.bar(lim, fms,               label="Father-Mother-Son" if lim == 1 else "_nolegend_", color="tab:blue")
        plt.bar(lim, fmd, bottom=fms,   label="Father-Mother-Daughter" if lim == 1 else "_nolegend_", color="tab:pink")

    plt.title("Number of triplets")
    plt.ylabel("Count")
    plt.xlabel("Limit")
    plt.legend()

    # plot_graph(remove_less_represented_nodes(G, 1), "kin-trees-1.pdf")
    # plot_graph(remove_less_represented_nodes(G, 5), "kin-trees-5.pdf")



    Gx = remove_less_represented_nodes(G, 5)
    components = list(nx.connected_components(Gx.to_undirected()))

    test_i  = set(random.sample(range(len(components)), int(len(components)*0.1)))
    val_i   = set(random.sample(set(range(len(components))) - test_i, int(len(components)*0.1)))
    train_i = set(range(len(components))) - test_i - val_i

    train_pids  = set().union(*[components[i] for i in train_i])
    val_pids    = set().union(*[components[i] for i in val_i])
    test_pids   = set().union(*[components[i] for i in test_i])


    train_samples = []
    val_samples =   []
    test_samples =  []
    for n, data in Gx.nodes(data=True):
        if(len(Gx.in_edges(n)) != 2): continue

        parents = list(map(lambda x: x[0], Gx.in_edges(n)))
        p1 = Gx.nodes[parents[0]]
        p2 = Gx.nodes[parents[1]]

        is_train = n in train_pids
        is_val   = n in val_pids

        c_pid = n
        f_pid = None
        m_pid = None
        if p1["sex"] == "M":
            f_pid = p1["pid"]
            m_pid = p2["pid"]
        else:
            f_pid = p2["pid"]
            m_pid = p1["pid"]

        f_iids = list(map(lambda x: (f_pid, x), images_df[images_df["pid"] == f_pid]["iid"].tolist()))
        m_iids = list(map(lambda x: (m_pid, x), images_df[images_df["pid"] == m_pid]["iid"].tolist()))
        c_iids = list(map(lambda x: (c_pid, x), images_df[images_df["pid"] == c_pid]["iid"].tolist()))
        
        new_samples = list(map(
            lambda x: [x[0][0], x[0][1], x[1][0], x[1][1], x[2][0], x[2][1]],
            itertools.product(f_iids, m_iids, c_iids)
        ))
        
        if is_train:
            train_samples += new_samples
        elif is_val:
            val_samples += new_samples
        else:
            test_samples += new_samples



    pd.DataFrame.from_records(train_samples, columns=["f_pid", "f_iid", "m_pid", "m_iid", "c_pid", "c_iid"]).to_csv("dataset/nokdb-samples-al5-train.csv", index=False)

    pd.DataFrame.from_records(val_samples , columns=["f_pid", "f_iid", "m_pid", "m_iid", "c_pid", "c_iid"]).to_csv("dataset/nokdb-samples-al5-validation.csv", index=False)

    pd.DataFrame.from_records(test_samples , columns=["f_pid", "f_iid", "m_pid", "m_iid", "c_pid", "c_iid"]).to_csv("dataset/nokdb-samples-al5-test.csv", index=False)

if __name__ == "__main__":
    main()