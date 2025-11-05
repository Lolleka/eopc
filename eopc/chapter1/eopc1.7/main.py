#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Union, Dict, Set
from collections import defaultdict

class Node:
    def __init__(self, value:int, x: float = 0.0, y: float = 0.0):
        self.x:float = x
        self.y:float = y
        self.value:int = value
        self.neighbors:List[Node] = []
        self.b = 0 # betweenness

class Network:
    def __init__(self, l:int, z:int=2, p:float = 0.0):
        if z % 2 != 0:
            raise ValueError("z must be even")
        self.nodes:List[Node] = []
        for i in range(l):
            pos = np.exp(1j*2*np.pi*i/l)
            self.nodes.append(Node(i, pos.real, pos.imag))
        # connect nodes
        for i in range(l):
            for j in range(1,z//2+1):
                self.addEdge(self.nodes[i], self.nodes[(i + j) % l])
                self.addEdge(self.nodes[i], self.nodes[(i - j) % l])
        # add p random edges
        for i in range(int(np.round(p*l*z//2))):
            i1 = np.random.randint(0, l)
            i2 = np.random.randint(0, l)
            if i1 != i2 and not self.hasEdge(self.nodes[i1], self.nodes[i2]):
                self.addEdge(self.nodes[i1], self.nodes[i2])

    def hasNode(self, node:Node) -> bool:
        return node in self.nodes

    def addNode(self, node:Node):
        if not self.hasNode(node):
            self.nodes.append(node)

    def hasEdge(self, node1:Node, node2:Node) -> bool:
        return node1 in node2.neighbors and node2 in node1.neighbors

    def addEdge(self, node1:Node, node2:Node):
        if not self.hasNode(node1):
            self.addNode(node1)
        if not self.hasNode(node2):
            self.addNode(node2)
        if not self.hasEdge(node1, node2):
            node1.neighbors.append(node2)
            node2.neighbors.append(node1)

    def getNodes(self) -> List[Node]:
        return self.nodes

    def getEdges(self) -> List[Tuple[Node, Node]]:
        edges = []
        for node in self.nodes:
            for neighbor in node.neighbors:
                i = node.value if node.value < neighbor.value else neighbor.value
                j = node.value if node.value > neighbor.value else neighbor.value
                edges.append((i, j))
        # sort edges by first value and then by second value
        edges.sort(key=lambda x: (x[0], x[1]))
        # remove duplicates
        seen = set()
        result = []
        for edge in edges:
            if edge not in seen:
                result.append(edge)
                seen.add(edge)
        return result

    def getNeighbors(self, node:Node) -> List[Node]:
        return node.neighbors

    def plot(self, ax=None, lw=1.0):
        if ax is None:
            fig, ax = plt.subplots()
        for node in self.nodes:
            ax.plot(node.x, node.y, 'o')
        for edge in self.getEdges():
            node1 = self.nodes[edge[0]]
            node2 = self.nodes[edge[1]]
            ax.plot([node1.x, node2.x], [node1.y, node2.y], 'k-', lw=lw)


def findPathLengthsFromNode(
    network:Network,
    node1:Node,
    exclude_nodes:Set[Node]=set()) -> Dict[Node, int]:
    assert network.hasNode(node1)
    # use BFS to find all nodes reachable from node1
    currentShell = [node1]
    l = 0
    lengths = {node1: l}
    while len(currentShell) > 0:
        nextShell = []
        for node in currentShell:
            for neighbor in network.getNeighbors(node):
                if neighbor not in lengths:
                    lengths[neighbor] = l + 1
                    nextShell.append(neighbor)
        l += 1
        currentShell = nextShell
    return {node: length for node, length in lengths.items() if node not in exclude_nodes}

def findAllPathLengths(network:Network) -> List[int]:
    lengths = {}
    visited = set()
    for node in network.getNodes():
        lengths[node] = list(findPathLengthsFromNode(network, node, visited).values())
        visited.add(node)
    all_lengths = sum(lengths.values(), [])
    return all_lengths

def findAveragePathLength(network:Network) -> float:
    return np.mean(findAllPathLengths(network))

def section_a():
    network = Network(6, 4, 0)
    _, ax = plt.subplots()
    network.plot(ax)
    plt.show()

def section_b1():
    l = 1000
    z = 2
    _, axes = plt.subplots(4,2)
    for p, ax in zip([0.02, 0.2, 0.5, 0.8], axes):
        network = Network(l, z, p)
        lengths = findAllPathLengths(network)
        network.plot(ax[0], lw=0.2)
        ax[1].hist(lengths, bins=np.max(lengths)+1, density=True, rwidth=0.8)
    plt.show()

def section_b2():
    l = 100
    z = 2
    p = 0.1
    n = 10
    for i in range(n):
        network = Network(l, z, p)
        lengths = findAllPathLengths(network)
        print(np.mean(lengths))

def section_c():
    l = 50
    z = 2
    n = 20
    lp0 = findAveragePathLength(Network(l, z, 0))
    print(lp0)
    p = np.logspace(-3, 1,n)
    l = np.array(list(map(lambda p: findAveragePathLength(Network(l, z, p)), p)))/lp0
    # plot on semilog scale
    plt.subplots()
    plt.semilogx(p, l)
    plt.show()

def section_d1():
    _, ax = plt.subplots(3,1)
    l = 50; z = 2; p = 0.1
    network = Network(l, z, p)
    network.plot(ax[0], lw =0.2)
    l = 1000; z = 10; p = 0.1
    network = Network(l, z, p)
    network.plot(ax[1], lw =0.2)
    l = 1000; z = 10; p = 0.001
    network = Network(l, z, p)
    network.plot(ax[2], lw =0.2)
    plt.show()

def section_d2():
    l = 100
    z = 2
    # _, ax = plt.subplots(2,2)
    fig, ax = plt.subplots()
    def inner(l, z, ax, n = 100):
        l0 = findAveragePathLength(Network(l, z, 0))
        print(l0)
        p = np.logspace(-3, 1,n)
        l_vs_p = np.array(list(map(lambda p: findAveragePathLength(Network(l, z, p)), p))) * np.pi*z/l
        ax.semilogx(p*l*z/2, l_vs_p)

    # inner(100, 2, ax[0,0])
    # inner(200, 2, ax[0,1])
    # inner(100, 4, ax[1,0])
    # inner(200, 4, ax[1,1])
    inner(100, 2, ax)
    inner(200, 2, ax)
    inner(100, 4, ax)
    inner(200, 4, ax)
    plt.show()

def findGeodesics(network:Network, node_j:Node):
    distances = defaultdict(set)
    d = 0
    distances[d].add(node_j)
    predecessors = defaultdict(set)
    unvisited = set(network.getNodes())
    unvisited.remove(node_j)
    while len(unvisited) > 0:
        for node_k in distances[d]:
            for node_l in node_k.neighbors:
                if node_l in unvisited:
                    distances[d+1].add(node_l)
                    unvisited.remove(node_l)
                predecessors[node_l].add(node_k)
        d = d + 1
    return distances, predecessors

def findBetweenness(network:Network):
    b_score = {node_k: 0 for node_k in network.getNodes()}
    all_nodes = network.getNodes()
    for node_j in all_nodes:
        distances, predecessors = findGeodesics(network, node_j)
        b_j = {node_k: 1 for node_k in all_nodes}
        # starting from fathest nodes, propagate b to predecessors
        for d in sorted(distances.keys(), reverse=True):
            for node_k in distances[d]:
                for node_p in predecessors[node_k]:
                    b_j[node_p] += b_j[node_k]/len(predecessors[node_k])
        for node_k, b_val in b_j.items():
            b_score[node_k] += b_val
    return b_score


def section_f():
    # we calculate betweennes using the algorithm
    # from PRE 64, 016132 (Newmann 2001)
    l = 100
    z = 2
    p = 0.05
    plt.subplots()
    plt.hist(list(findBetweenness(Network(l, z, p)).values()), density=True, rwidth=0.8)
    plt.show()


if __name__ == '__main__':
    section_a()
    section_b1()
    section_b2()
    section_c()
    section_d1()
    section_d2()
    section_f()
