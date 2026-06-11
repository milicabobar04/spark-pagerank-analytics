import json
import random
from collections import defaultdict

class GraphNode:
       def __init__(self, node_id: int, out_neighbors: list[int], metadata: dict = None):
        self.node_id: int = node_id
        self.out_neighbors: list[int] = list(out_neighbors)
        self.out_degree: int = len(self.out_neighbors)
        self.metadata: dict = metadata if metadata is not None else {}

       def __hash__(self) -> int:
        return hash(self.node_id)
       
       def to_dict(self) -> dict:
        return {
            "id": self.node_id,
            "outNeighbors": self.out_neighbors
        }

        @classmethod
        def from_dict(cls, data: dict) -> "GraphNode":
            return cls(
                node_id=data["id"],
                out_neighbors=data["outNeighbors"]
            )
       

# edge-list -> list GraphNode (pomocna funkcija)
def edges_to_nodes(edges : list[tuple[int,int]], all_nodes_ids: set[int] = None) -> list[GraphNode]:
   if all_nodes_ids is None:
      all_nodes_ids = set()
      for src, dst in edges:
         all_nodes_ids.add(src)
         all_nodes_ids.add(dst)
  
   adjacency: dict[int, list[int]] = defaultdict(list)
   for src, dst in edges:
      adjacency[src].append(dst)

   nodes = []
   for node_id in sorted(all_nodes_ids):
      nodes.append(GraphNode(node_id, adjacency.get(node_id, [])))

   return nodes

# Generisanje sintetickog grafa
def generate_graph(N: int, E: int, seed: int = 42) -> list[GraphNode]:
   """
        N : int
            Broj čvorova 
        E : int
            Željeni broj veza.
        seed : int
    """
   if N <= 0:
      raise ValueError("Broj čvorova N mora biti pozitivan.")
   if E < 0:
      raise ValueError("Broj veza E ne smije biti negativan.")
   
   # Maksimalan broj mogucih veza u grafu bez petlji
   max_edges = N * (N - 1)
   if E > max_edges:
      E = max_edges
   
   rng = random.Random(seed)

   # Ako trazimo manje od 50% posto maksimalnog broja cvorova dodajemo jednu po jednu vezu
   # Ako trazimo vise od 50% posto generisemo sve moguce i uklanjamo jednu po jednu vezu
   edges = set()

   if E <= max_edges * 0.5:
      while len(edges) < E:
         src = rng.randint(0, N - 1)
         dst = rng.randint(0, N - 1)
         if src != dst:
            edges.add((src, dst))
   else:
      all_possible = [(u, v) for u in range(N) for v in range(N) if u != v]
      rng.shuffle(all_possible)
      edges = set(all_possible[:E])

   all_nodes_ids = set(range(N))
   return edges_to_nodes(list(edges), all_nodes_ids)
      
# Generisanje grafa iz edge-list
def load_graph_from_edgelist(filepath: str, delimiter: str = None) -> list[GraphNode]:
   edges = []
   all_node_ids = set()

   with open(filepath, "r") as f:
      for line_number, line in enumerate(f, start = 1):
         line = line.strip()

         if not line or line.startswith("#"):
            continue
         
         parts = line.split(delimiter)

         if len(parts) != 2:
            raise ValueError(f"Neispravan format na liniji {line_number}: '{line}'. "
                  f"Očekivano: 'srcId dstId'.")
         try:
            src = int(parts[0])
            dst = int(parts[1])
         except ValueError:
            raise ValueError(
                   f"Neispravan tip podataka na liniji {line_number}: '{line}'. "
                   f"srcId i dstId moraju biti cijeli brojevi."
               )
         edges.append((src,dst))
         all_node_ids.add(src)
         all_node_ids.add(dst)
   return edges_to_nodes(edges, all_node_ids)


# Serijalizacija i deserijalizacije (JSON)
def save_graph_to_json(nodes: list[GraphNode], filepath: str) -> None:
   data = {
      "nodes" : [node.to_dict() for node in nodes]
   }
   with open(filepath, "w") as f:
      json.dump(data, f, indent=2)



def load_graph_from_json(filepath: str) -> list[GraphNode]:
   with open(filepath, "r") as f:
      data = json.load(f)

   if "nodes" not in data:
      raise ValueError("JSON fajl mora sadržavati 'nodes' ključ.")
   return [GraphNode.from_dict(node_data) for node_data in data["nodes"]]
