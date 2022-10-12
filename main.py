import numpy as np
import pandas as pd
import networkx as nx
import random
import tqdm
from gensim.models.word2vec import Word2Vec


def create_graph(file_name):
    edges = pd.read_csv(file_name).values.tolist()
    graph = nx.from_edgelist(edges)
    return graph


def walk_transformer(walk, length):
    transformed_walk = []
    for step in range(length+1):
        neighbors = [y for i, y in enumerate(walk[step:]) if i % length == 0]
        transformed_walk.append(neighbors)
    return transformed_walk


class FirstOrderRandomWalker:
    def __init__(self, graph, walk_length, walk_number):
        self.graph = graph
        self.walk_length = walk_length
        self.walk_number = walk_number
        self.walks = []

    def do_walk(self, node):
        walk = [node]
        for _ in range(self.walk_length-1):
            nebs = [node for node in self.graph.neighbors(walk[-1])]
            if len(nebs) > 0:
                walk = walk + random.sample(nebs, 1)
        walk = [str(w) for w in walk]
        return walk

    def do_walks(self):
        print("\nModel initialized.\nRandom walks started.")
        for iteration in range(self.walk_number):
            print("\nRandom walk round: "+str(iteration+1) +
                  "/"+str(self.walk_number)+".\n")
            for node in self.graph.nodes():
                walk_from_node = self.do_walk(node)
                self.walks.append(walk_from_node)
        return self.walks


class SecondOrderRandomWalker:
    def __init__(self, nx_G, is_directed, walk_length, walk_number, P, Q):
        self.G = nx_G
        self.nodes = nx.nodes(self.G)
        print("Edge weighting.\n")
        for edge in tqdm(self.G.edges()):
            self.G[edge[0]][edge[1]]['weight'] = 1.0
            self.G[edge[1]][edge[0]]['weight'] = 1.0
        self.is_directed = is_directed
        self.walk_length = walk_length
        self.walk_number = walk_number
        self.p = P
        self.q = Q

    def node2vec_walk(self, start_node):
        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        walk = [start_node]

        while len(walk) < self.walk_length:
            cur = walk[-1]
            cur_n = sorted(G.neighbors(cur))
            if len(cur_n) > 0:
                if len(walk) == 1:
                    walk.append(
                        cur_n[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    pre = walk[-2]
                    next = cur_n[alias_draw(
                        alias_edges[(pre, cur)][0], alias_edges[(pre, cur)][1])]
                    walk.append(next)
            else:
                break
        walk = [str(w) for w in walk]
        return walk

    def do_walks(self):
        G = self.G
        walks = []
        nodes = list(G.nodes())
        for walk_it in range(self.walk_number):
            print("\nRandom walk round: "+str(walk_it+1) +
                  "/"+str(self.walk_number)+".\n")
            random.shuffle(nodes)
            for node in tqdm(nodes):
                walks.append(self.node2vec_walk(start_node=node))

        return walks

    def get_alias_edge(self, src, dst):
        G = self.G
        p = self.p
        q = self.q

        unnormalized_probs = []
        for dst_nbr in sorted(G.neighbors(dst)):
            if dst_nbr == src:
                unnormalized_probs.append(G[dst][dst_nbr]['weight']/p)
            elif G.has_edge(dst_nbr, src):
                unnormalized_probs.append(G[dst][dst_nbr]['weight'])
            else:
                unnormalized_probs.append(G[dst][dst_nbr]['weight']/q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [
            float(u_prob)/norm_const for u_prob in unnormalized_probs]

        return alias_setup(normalized_probs)

    def preprocess_transition_probs(self):
        G = self.G
        is_directed = self.is_directed

        alias_nodes = {}
        print("")
        print("Preprocesing.\n")
        for node in tqdm(G.nodes()):
            unnormalized_probs = [G[node][nbr]['weight']
                                  for nbr in sorted(G.neighbors(node))]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [
                float(u_prob)/norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = alias_setup(normalized_probs)

        alias_edges = {}
        triads = {}

        if is_directed:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
        else:
            for edge in tqdm(G.edges()):
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
                alias_edges[(edge[1], edge[0])] = self.get_alias_edge(
                    edge[1], edge[0])

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

        return


def alias_setup(probs):
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)
    smaller = []
    larger = []

    for kk, prob in enumerate(probs):
        q[kk] = K*prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q


def alias_draw(J, q):
    K = len(J)

    kk = int(np.floor(np.random.rand()*K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]


class WalkletMachine:
    def __init__(self, input_file="data/food_edges.csv", walk_type="first", window_size=8, dimensions=128, walk_length=2, walk_number=5, min_count=10, workers=3, output_file="output.embedding"):
        self.input_file = input_file
        self.walk_type = walk_type
        self.window_size = window_size
        self.dimensions = dimensions
        self.workers = workers
        self.min_count = min_count
        self.output_file = output_file
        self.walk_length = walk_length
        self.walk_number = walk_number
        self.P = 1
        self.Q = 1

        self.graph = create_graph(input_file)
        if walk_type == "first":
            self.walker = FirstOrderRandomWalker(
                self.graph, self.walk_length, self.walk_number)
        else:
            self.walker = SecondOrderRandomWalker(
                self.graph, False, self.walk_length, self.walk_number, self.P, self.Q)
            self.walker.preprocess_transition_probs()
        self.walks = self.walker.do_walks()
        del self.walker
        self.create_embedding()
        self.save_model()

    def walk_extracts(self, length):

        good_walks = [walk_transformer(walk, length) for walk in self.walks]
        good_walks = [w for walks in good_walks for w in walks]
        return good_walks

    def get_embedding(self, model):

        embedding = []
        for node in range(len(self.graph.nodes())):
            embedding.append(list(model[str(node)]))
        embedding = np.array(embedding)
        return embedding

    def create_embedding(self):

        self.embedding = []
        for index in range(1, self.window_size+1):
            print("\nOptimization round: "+str(index) +
                  "/"+str(self.window_size)+".")
            print("Creating documents.")
            clean_documents = self.walk_extracts(index)
            print("Fitting model.")

            model = Word2Vec(clean_documents,
                             vector_size=self.dimensions,
                             window=1,
                             min_count=self.min_count,
                             sg=1,
                             workers=self.workers)

            new_embedding = self.get_embedding(model)
            self.embedding = self.embedding + [new_embedding]
        self.embedding = np.concatenate(self.embedding, axis=1)

    def save_model(self):

        print("\nModels are integrated to be multi scale.\nSaving to disk.")
        self.column_names = ["x_" + str(x)
                             for x in range(self.embedding.shape[1])]
        self.embedding = pd.DataFrame(
            self.embedding, columns=self.column_names)
        self.embedding.to_csv(self.output_file, index=None)


def main():
    WalkletMachine()


if __name__ == "__main__":
    main()
