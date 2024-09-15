import random
import math
import timeit
import matplotlib.pyplot as plt

class minHeap:
    def __init__(self, data):
        self.nodes = []
        self.weights = []
        self.index = {}
        for node, weight in data:
            self.nodes.append(node)
            self.weights.append(weight)
            self.index[node] = len(self.nodes) - 1

        self.length = len(data)
        self.build_heap()

    def find_left_index(self, index):
        return 2 * index + 1

    def find_right_index(self, index):
        return 2 * index + 2

    def find_parent_index(self, index):
        return (index - 1) // 2

    def swap(self, i, j):
        self.nodes[i], self.nodes[j] = self.nodes[j], self.nodes[i]
        self.weights[i], self.weights[j] = self.weights[j], self.weights[i]
        self.index[self.nodes[i]], self.index[self.nodes[j]] = i, j

    def heapify(self, index):
        smallest_known_index = index
        left_index = self.find_left_index(index)
        right_index = self.find_right_index(index)

        if left_index < self.length and self.weights[left_index] < self.weights[index]:
            smallest_known_index = left_index
        if right_index < self.length and self.weights[right_index] < self.weights[smallest_known_index]:
            smallest_known_index = right_index
        if smallest_known_index != index:
            self.swap(index, smallest_known_index)
            self.heapify(smallest_known_index)

    def build_heap(self):
        for i in range(self.length // 2 - 1, -1, -1):
            self.heapify(i)

    def swim(self, index):
        parent_index = self.find_parent_index(index)

        while index > 0 and self.weights[index] < self.weights[parent_index]:
            self.swap(index, parent_index)
            index = parent_index
            parent_index = self.find_parent_index(index)

    def insert_value(self, node, weight):
        self.nodes.append(node)
        self.weights.append(weight)
        self.length += 1
        self.index[node] = self.length - 1  # Correct index assignment
        self.swim(self.length - 1)

    def update(self, node, new_weight):
        if node not in self.index:
            return None

        node_index = self.index[node]
        old_weight = self.weights[node_index]
        self.weights[node_index] = new_weight

        if new_weight < old_weight:
            self.swim(node_index)
        else:
            self.heapify(node_index)

    def extract_min(self):
        if self.length == 0:
            return 
        
        out = (self.nodes.pop(0), self.weights.pop(0))

        for k, v in self.index.items():
            if v == 0:
                del self.index[k]
                break  
        for k in self.index:
            self.index[k] -= 1

        self.length -= 1
        return out
    
class WeightEdge:
    def __init__(self, u, v, weight) -> None:
        self.u = u
        self.v = v
        self.weight = weight
    
    def get_weight(self):
        return self.weight
    
    def start(self):
        return self.u
    
    def end(self):
        return self.v

    def other(self, node):
        if self.start() == node:
            return self.end()
        return self.start()
    
class Graph:
    def __init__(self, w_edges):
        self.graph = {}
        self.numE = 0         # Represents number of edges
        for edge in w_edges:
            v = edge.start()
            u = edge.end()
            if v not in self.graph.keys():
                self.graph[v]=[]
            self.graph[v].append(edge)

            if u not in self.graph.keys():
                self.graph[u]=[]
            self.graph[u].append(edge)
            # We are adding the same WeightEdge object to both the vertices
            self.numE +=2

        self.numV = len(self.graph.keys())  # Represents number of vertices

    def get_edges(self, node):
        edges = []
        if node not in self.graph:
            raise IndexError("vertex not exist")
        for e in self.graph[node]:
            edges.append([e.start(), e.end(), e.get_weight()])

        return edges
    
    def add_edge(self,node1, node2, w):
        edge = WeightEdge(node1, node2, w)

        if node1 not in self.graph.keys():
            self.graph[node1]=[]
            self.numV += 1
        self.graph[node1].append(edge)
        
        if node2 not in self.graph.keys():
            self.graph[node2]=[]
            self.numV += 1
        self.graph[node2].append(edge)

    def remove_edge(self, node):
        self.graph[node] = []
        for edges in self.graph.values():
            edges[:] = [edge for edge in edges if edge.start() != node and edge.end() != node]
                
    def get_graphList(self):
        all_edges = {}
        for key in self.graph.keys():
            edge = self.graph[key]
            for e in edge:
                if key not in all_edges:
                    all_edges[key]=[]
                all_edges[key].append([e.start(), e.end(), e.get_weight()])
        return all_edges
    
    def dijkstra(self, s, k=None):
        if k == None: 
            k = self.numV + 2

        dist = {}
        path = {}
        relaxLimit = {}
        for n in self.graph.keys():
            dist[n] = math.inf
            path[n] = []
            relaxLimit[n] = k

        W_edges = []
        for v in self.graph.keys():
            W_edges.append((v, math.inf))
        
        pq = minHeap(W_edges)

        pq.update(s, 0)
        dist[s] = 0
        relaxLimit[s] -=1

        while pq.length > 0:
            node, d = pq.extract_min()
            for e in self.graph[node]:
                adjNode = e.other(node)
                weight = e.get_weight()

                if (d + weight < dist[adjNode] and relaxLimit[adjNode] > 0):
                    dist[adjNode] = d + weight
                    pq.update(adjNode, dist[adjNode])
                    relaxLimit[adjNode] -= 1
                    path[adjNode] = path[node] + [node]

        return dist, path
    
    def bellmanFord(self, s, k=None):
        if k == None: 
            k = self.numV

        dist = {}
        path = {}
        relaxLimit = {}
        for n in self.graph.keys():
            dist[n] = math.inf
            path[n] = []
            relaxLimit[n] = k

        dist[s] = 0
        relaxLimit[s] -=1

        for i in range(self.numV - 1):
            for node in self.graph.keys():
                for e in self.graph[node]:
                    adjNode = e.other(node)
                    weight = e.get_weight()

                    if (dist[node] + weight < dist[adjNode] and relaxLimit[adjNode] > 0):
                        dist[adjNode] = dist[node] + weight
                        relaxLimit[adjNode] -= 1
                        path[adjNode] = path[node] + [node]

        # Check for negative cycles 
        for node in self.graph.keys():
            for e in self.graph[node]:
                adjNode = e.other(node)
                weight = e.get_weight()

                if (dist[node] + weight < dist[adjNode] and relaxLimit[adjNode] > 0):
                    print("Negative cycle detected")
                    return dist, path

        return dist, path
    
def A_Star(graph, source, destination, heuristic):
    predecessors = {}
    costs = {}
    
    W_edges = []
    for v in graph.graph.keys():
        costs[v] = math.inf
        predecessors[v] = None
        W_edges.append((v, math.inf))

    open_set = minHeap(W_edges)
    costs[source] = 0
    open_set.update(source, heuristic[source]) 
    
    while not open_set.length == 0:
        current_node = open_set.extract_min()[0]
        
        if current_node == destination:
            break

        for edge in graph.graph[current_node]:
            neighbor = edge.other(current_node)
            weight = edge.get_weight()
            new_cost = costs[current_node] + weight 
            if new_cost < costs[neighbor]:
                costs[neighbor] = new_cost
                predecessors[neighbor] = current_node
                open_set.update(neighbor, new_cost + heuristic[neighbor])  
    
    path = []
    current_node = destination
    while current_node is not None:
        path.append(current_node)
        current_node = predecessors[current_node]
    path.reverse()
    
    return predecessors, path

def euclidean(x1,x2,y1,y2):
    return ((x2 - x1)**2 + (y2 - y1)**2)**(1/2)

import csv

stations = {}
with open("london_stations.csv", newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  

    for row in reader:
        station_id = int(row[0])
        stations[station_id] = [int(row[0]), float(row[1]), float(row[2])]

Weight_edges = []
edges_list = []
connections = []
with open("london_connections.csv", newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  
    
    for row in reader:
        node1 = stations[int(row[0])]
        node2 = stations[int(row[1])]

        w = euclidean(node1[1],node2[1],node1[2],node2[2])

        Weight_edges.append(WeightEdge(node1[0], node2[0], w))
        edges_list.append([node1[0], node2[0], w])

        connections.append([int(row[0]), int(row[1]), int(row[2]), int(row[3])])

def heuristic(source):
    source_node = stations[source]
    
    heuristic_distances = {}

    heuristic_distances[source] = 0

    for node in stations.values():
        distance = euclidean(source_node[1], node[1], source_node[2], node[2])
        heuristic_distances[node[0]] = distance

    return heuristic_distances

### All pairs shortest paths using DIJKSTRA
def shortestPathDIJKSTRA(g):
    path = {}
    for node1 in stations:
        path_node = g.dijkstra(node1)[1]
        for p in path_node:
            if (node1,p) in path.keys() or (p,node1) in path.keys():
                continue
            path[(node1,p)] = path_node[p]

    return path

### All pairs shortest paths using A*
def shortestPathASTAR(g):
    path = {}
    for edge in edges_list:
        source = edge[0]
        dest = edge[1]
        path_node = A_Star(g, source, dest, heuristic(source))[1]

        path[(source,dest)] = path_node
    
    for node1 in stations:
        for node2 in stations:
            if ((node1, node2) not in path.keys()) and ((node2, node1) not in path.keys()):
                path[(node1,node2)] = []

    return path

# EXPERIMENT FOR ALL SHORTEST PATH FOR THE TWO FUNCTIONS
def draw_2_graphs(y1, y2, title):
    x_values = [i for i in range(1, 10)]
    plt.figure()
    plt.plot(x_values, y1[1:], label="dijkstra", marker='o')
    plt.plot(x_values, y2[1:], label="A*", marker='s')
    plt.xlabel('Iterations')
    plt.ylabel('Run Times')
    plt.title(title)
    plt.legend()
    plt.show()

runs = 10
DIJKSTRAtime = []
ASTARtime = []

for i in range(1, runs + 1):      
    g1 = Graph(Weight_edges)
    g2 = Graph(Weight_edges)

    start = timeit.default_timer()
    shortestPathDIJKSTRA(g1)
    stop = timeit.default_timer()
    DIJKSTRAtime.append(stop-start)

    start = timeit.default_timer()
    shortestPathASTAR(g2)
    stop = timeit.default_timer()
    ASTARtime.append(stop-start)


print("Average time for DIJKSTRA :", sum(DIJKSTRAtime) / len(DIJKSTRAtime))
print("Average time for A* :", sum(ASTARtime) / len(ASTARtime))

draw_2_graphs(DIJKSTRAtime, ASTARtime , "DIJKSTRA v/s A*")

## Stations on the same line. 
def draw_2_graphs(y1, y2, title):
    x_values = [i for i in range(len(samelinePairs) - 1)]
    plt.figure()
    plt.plot(x_values, y1[1:], label="dijkstra", marker='o')
    plt.plot(x_values, y2[1:], label="A*", marker='s')
    plt.xlabel('Stattions')
    plt.ylabel('Run Times')
    plt.title(title)
    plt.legend()
    plt.show()

sameline = {}
for e in connections:
    if e[2] not in sameline.keys():
        sameline[e[2]] = []

    if e[0] not in sameline[e[2]]:
        sameline[e[2]].append(e[0])
    if e[1] not in sameline[e[2]]:
        sameline[e[2]].append(e[1])

# FOR SIMPLICTY WE WILL ONLY TEST ON STATIONS ON LINES 1.
samelineSMALL = {key: value for key, value in sameline.items() if key in [1]}

samelinePairs = []
for k in samelineSMALL:
    for s1 in samelineSMALL[k]:
        for s2 in samelineSMALL[k]:
            if s1 != s2 and ([s1,s2] not in samelinePairs or [s2,s1] not in samelinePairs):
                samelinePairs.append([s1,s2])


DIJKSTRAtime = []
ASTARtime = []

g = Graph(Weight_edges)

for e in samelinePairs: 
        start = timeit.default_timer()
        # Using dijkstra algo to find the path between the two edges.
        (g.dijkstra(e[0])[1])[e[1]]
        stop = timeit.default_timer()
        DIJKSTRAtime.append(stop-start)

        start = timeit.default_timer()
        # Using A* algo to find the path between the two edges.
        A_Star(g, e[0], e[1], heuristic(e[0]))[1]
        stop = timeit.default_timer()
        ASTARtime.append(stop-start)

print("Average time for DIJKSTRA :", sum(DIJKSTRAtime) / len(DIJKSTRAtime))
print("Average time for A* :", sum(ASTARtime) / len(ASTARtime))

draw_2_graphs(DIJKSTRAtime, ASTARtime , "DIJKSTRA v/s A*")

## Stations on the adjacent line. 
def draw_2_graphs(y1, y2, title):
    x_values = [i for i in range(len(adjlinePairs) - 1)]
    plt.figure()
    plt.plot(x_values, y1[1:], label="dijkstra", marker='o')
    plt.plot(x_values, y2[1:], label="A*", marker='s')
    plt.xlabel('Stattions')
    plt.ylabel('Run Times')
    plt.title(title)
    plt.legend()
    plt.show()

# FOR SIMPLICTY WE WILL ONLY TEST ON STATIONS ON LINES 2 which transfer to LINE 6.
adjline = []
for s1 in connections:
    for s2 in connections:
        if s1 != s2:
            if s1[1] == s2[0] and (s1[2] == 2 and s2[2] == 6): 
                if ([s1,s2] not in adjline or [s2,s1] not in adjline):
                    adjline.append([s1,s2])

samelineSMALLDICT = {key: value for key, value in sameline.items() if key in [6]}

# FOR SIMPLICTY WE WILL ONLY TEST ON STATIONS ON LINES 2 which transfer to LINE 6 THEN TRANSFER TO ANY OTHER NODE IN 6.
adjlinePairs = []
for pair in adjline:
    s1 = pair[0][0]

    for val in samelineSMALLDICT.values():
        for s in val:
            adjlinePairs.append([s1,s])

DIJKSTRAtime = []
ASTARtime = []

g = Graph(Weight_edges)

for e in adjlinePairs: 
        start = timeit.default_timer()
        # Using dijkstra algo to find the path between the two edges.
        (g.dijkstra(e[0])[1])[e[1]]
        stop = timeit.default_timer()
        DIJKSTRAtime.append(stop-start)

        start = timeit.default_timer()
        # Using A* algo to find the path between the two edges.
        A_Star(g, e[0], e[1], heuristic(e[0]))[1]
        stop = timeit.default_timer()
        ASTARtime.append(stop-start)

print("Average time for DIJKSTRA :", sum(DIJKSTRAtime) / len(DIJKSTRAtime))
print("Average time for A* :", sum(ASTARtime) / len(ASTARtime))

draw_2_graphs(DIJKSTRAtime, ASTARtime , "DIJKSTRA v/s A*")

## Stations on the Transfer. 
def draw_2_graphs(y1, y2, title):
    x_values = [i for i in range(len(transferPairs) - 1)]
    plt.figure()
    plt.plot(x_values, y1[1:], label="dijkstra", marker='o')
    plt.plot(x_values, y2[1:], label="A*", marker='s')
    plt.xlabel('Stattions')
    plt.ylabel('Run Times')
    plt.title(title)
    plt.legend()
    plt.show()

sameline = {}
for e in connections:
    if e[2] not in sameline.keys():
        sameline[e[2]] = []

    if e[0] not in sameline[e[2]]:
        sameline[e[2]].append(e[0])
    if e[1] not in sameline[e[2]]:
        sameline[e[2]].append(e[1])

adjline = []
for s1 in connections:
    for s2 in connections:
        if s1 != s2:
            if s1[1] == s2[0] and (s1[2] != s2[2]): 
                if ([s1,s2] not in adjline or [s2,s1] not in adjline):
                    adjline.append([s1,s2])

def samelineFUNC(line):
    samelineSMALLDICT = {key: value for key, value in sameline.items() if key in [line]}
    return samelineSMALLDICT

def found(s1, list):
    for s2 in connections:
        if s1 != s2:
            if s1[1] == s2[0] and (s1[2] != s2[2]): 
                if ([s1,s2] not in l or [s2,s1] not in l):
                    return s2

    s2 = random.choice(list)
    
    while s1[0] == s2[0]:
        s2 = random.choice(list)

    return s2


transfers = []
for start in adjline:
    l = []
    s1 = start[0]

    for n in range(20):
        list = [c for c in connections if c[2] == s1[2]]
        s2 = found(s1, list)

        l.append([s1,s2])
        s1 = s2
    transfers.append(l)


transferPairs = []
for s in transfers:
    print(s)
    start = s[0][0][0]
    end = s[-1][1][1]

    transferPairs.append([start, end])
# OUTPUT SHOWS THAT MULTIPLE TRANSFERS TAKES PLACE.

DIJKSTRAtime = []
ASTARtime = []

g = Graph(Weight_edges)
for e in transferPairs: 
        start = timeit.default_timer()
        # Using dijkstra algo to find the path between the two edges.
        (g.dijkstra(e[0])[1])[e[1]]
        stop = timeit.default_timer()
        DIJKSTRAtime.append(stop-start)

        start = timeit.default_timer()
        # Using A* algo to find the path between the two edges.
        A_Star(g, e[0], e[1], heuristic(e[0]))[1]
        stop = timeit.default_timer()
        ASTARtime.append(stop-start)

print("Average time for DIJKSTRA :", sum(DIJKSTRAtime) / len(DIJKSTRAtime))
print("Average time for A* :", sum(ASTARtime) / len(ASTARtime))

draw_2_graphs(DIJKSTRAtime, ASTARtime , "DIJKSTRA v/s A*")

#NUMBER OF LINES IN THE SHORTEST PATH
sameline = {}
for e in connections:
    if e[2] not in sameline.keys():
        sameline[e[2]] = []

    if e[0] not in sameline[e[2]]:
        sameline[e[2]].append(e[0])
    if e[1] not in sameline[e[2]]:
        sameline[e[2]].append(e[1])

adjline = []
for s1 in connections:
    for s2 in connections:
        if s1 != s2:
            if s1[1] == s2[0] and (s1[2] != s2[2]): 
                if ([s1,s2] not in adjline or [s2,s1] not in adjline):
                    adjline.append([s1,s2])

def samelineFUNC(line):
    samelineSMALLDICT = {key: value for key, value in sameline.items() if key in [line]}
    return samelineSMALLDICT

def found(s1, list):
    for s2 in connections:
        if s1 != s2:
            if s1[1] == s2[0] and (s1[2] != s2[2]): 
                if ([s1,s2] not in l or [s2,s1] not in l):
                    return s2

    s2 = random.choice(list)
    
    while s1[0] == s2[0]:
        s2 = random.choice(list)

    return s2


transfers = []
# FOR SIMPLICITY WE ONLY CONSIDER 3 CASES OF STATIONS WITH MULTIPLE TRANSFER LINES
for start in adjline:
    l = []
    s1 = start[0]

    for n in range(20):
        list = [c for c in connections if c[2] == s1[2]]
        s2 = found(s1, list)

        l.append([s1,s2])
        s1 = s2
    transfers.append(l)


transferPairs = []
for s in transfers:
    start = s[0][0][0]
    end = s[-1][1][1]

    transferPairs.append([start, end])
# OUTPUT SHOWS THAT MULTIPLE TRANSFERS TAKES PLACE.

def drawgraphs(y1, title):
    x_values = [i for i in range(len(numberLine) - 1)]
    plt.figure()
    plt.plot(x_values, y1[1:], label="Number of Lines", marker='o')
    plt.xlabel('Random Travels')
    plt.ylabel('Number of Lines')
    plt.title(title)
    plt.legend()
    plt.show()


numberLine = []

g = Graph(Weight_edges)
for e in transferPairs: 
    numberLine.append(len(A_Star(g, e[0], e[1], heuristic(e[0]))[1]))

print("Average number of Lines :", sum(numberLine) / len(numberLine))

drawgraphs(numberLine , "DIJKSTRA v/s A*")