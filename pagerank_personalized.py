import json
import time
import argparse
import random
from pyspark.sql import SparkSession
from graph import generate_graph, GraphNode

def create_spark_session(app_name: str = "PersonalizedPageRank"):
    spark = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.executor.extraJavaOptions", "-Djava.security.manager=allow")
        .config("spark.driver.extraJavaOptions",   "-Djava.security.manager=allow")
        .config("spark.default.parallelism", "8")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    return spark

def run_pagerank_generic(
    sc,
    nodes_list: list,
    seed_nodes: set = None,  
    d: float = 0.85,
    max_iter: int = 100,
    epsilon: float = 1e-6,
    top_k: int = 10,
    verbose: bool = True,
    num_partitions: int = 8
) -> dict:
    
    total_start = time.time()
    N = len(nodes_list)
    initial_rank = 1.0 / N

    # Staticna struktura grafa
    # (nodeId, [outNeighbors])
    graph_rdd = (
        sc.parallelize(
            [(node.node_id, node.out_neighbors) for node in nodes_list],
            numSlices=num_partitions
        )
        .partitionBy(num_partitions)
        .persist() 
    )

    # Rankovi cvorova
    # (nodeId, rank)
    ranks = (
        sc.parallelize(
            [(node.node_id, initial_rank) for node in nodes_list],
            numSlices=num_partitions
        )
        .partitionBy(num_partitions)
    )

    # Cvorovi bez izlaza
    # (nodeId, True)
    dangling_rdd = (
        sc.parallelize(
            [(node.node_id, True) for node in nodes_list if node.out_degree == 0],
            numSlices=num_partitions
        )
        .partitionBy(num_partitions)
        .persist()
    )

    graph_rdd.count()

    is_personalized = seed_nodes is not None
    
    if is_personalized:
        num_seeds = len(seed_nodes)
        if num_seeds == 0:
            raise ValueError("Seed set for Personalized PageRank cannot be empty.")
        seed_nodes_bd = sc.broadcast(seed_nodes)
        
        # If v in S: 1/|S|, else 0
        def get_teleport_prob(node_id):
            return 1.0 / num_seeds if node_id in seed_nodes_bd.value else 0.0
    else:
        # Standard PR: Uniform 1/N for everyone
        def get_teleport_prob(node_id):
            return 1.0 / N

    converged = False
    iterations_done = 0
    iteration_times = []

    for iteration in range(max_iter):
        iter_start = time.time()

        # Suma rankova cvorava izlaznog stepena 0
        dangling_mass = (
            dangling_rdd
            .join(ranks)
            .map(lambda x: x[1][1])
            .sum()
        )

        # Doprinosi susjedima
        contributions = (
            graph_rdd
            .join(ranks)
            .flatMap(lambda x: (
                [(neighbor, x[1][1] / len(x[1][0])) for neighbor in x[1][0]]
            ) if x[1][0] else [])
        )
        
        # Sabiramo sve doprinose po cvoru
        sum_incoming = contributions.reduceByKey(lambda a, b: a + b)

        # New Rank = (1-d)*T(v) + d * ( Incoming + DanglingMass * T(v) )
        # Factorized: T(v) * [ (1-d) + d*DanglingMass ] + d * Incoming

        base_teleport_mass = (1.0 - d) + d * dangling_mass

        # Azuriramo rankove
        ranks_and_diff = (
            ranks
            .leftOuterJoin(sum_incoming)
            .map(lambda x: (
                x[0], 
                (
                    get_teleport_prob(x[0]) * base_teleport_mass + 
                    d * (x[1][1] if x[1][1] is not None else 0.0), # Novi rank
                    x[1][0]                                        # Stari rank
                )
            ))
            # (nodeId, (new_rank, old_rank))
            # -> (nodeId, (new_rank, abs(new - old)))
            .mapValues(lambda x: (
                x[0],             
                abs(x[0] - x[1])  
            ))
        )
        
        ranks_and_diff.cache()
        ranks_and_diff.count()  

        total_diff = (
            ranks_and_diff
            .map(lambda x: x[1][1])
            .sum()
        )

        new_ranks = ranks_and_diff.mapValues(lambda x: x[0])

        ranks.unpersist()
        ranks = new_ranks

        iter_time = time.time() - iter_start
        iteration_times.append(iter_time)
        iterations_done += 1

        if verbose:
            print(f"   Iter {iteration+1:>3} | Diff: {total_diff:.8f} | Time: {iter_time:.4f}s")

        if total_diff < epsilon:
            converged = True
            break

    total_time = time.time() - total_start
    top_k_results = ranks.takeOrdered(top_k, key=lambda x: -x[1])
    
    if is_personalized:
        seed_nodes_bd.unpersist()
    
    graph_rdd.unpersist()
    dangling_rdd.unpersist()

    return {
        "top_k": top_k_results,
        "iterations": iterations_done,
        "total_time": total_time,
        "converged": converged
    }

# J(A, B) = |A ∩ B| / |A ∪ B|
def calculate_overlap(list_a, list_b):
    set_a = set(x[0] for x in list_a)
    set_b = set(x[0] for x in list_b)
    
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    
    return intersection / union if union > 0 else 0.0

def run_experiment_batch(sc, num_graphs=10, N=1000, E=5000, top_k=10, num_partitions=8):
    print(f"\nPOKRETANJE EKSPERIMENTA ({num_graphs} grafova)...")
    results = []

    for i in range(num_graphs):
        seed_gen = 42 + i
        print(f"\n--- Graf {i+1}/{num_graphs} (Seed {seed_gen}) ---")
        
        nodes = generate_graph(N, E, seed=seed_gen)
        
        all_ids = [n.node_id for n in nodes]
        random_seeds = set(random.sample(all_ids, k=min(5, len(all_ids))))
        print(f"Seed Nodes za PPR: {random_seeds}")
        
        print("Running Standard PR...")
        std_res = run_pagerank_generic(sc, nodes, seed_nodes=None, top_k=top_k, verbose=True, num_partitions=num_partitions)
        
        print("Running Personalized PR...")
        ppr_res = run_pagerank_generic(sc, nodes, seed_nodes=random_seeds, top_k=top_k, verbose=True, num_partitions=num_partitions)

        overlap = calculate_overlap(std_res["top_k"], ppr_res["top_k"])
        print(f"Preklapanje : {overlap:.4f}")
        
        results.append({
            "graph_id": i,
            "seeds": list(random_seeds),
            "std_top_k": std_res["top_k"],
            "ppr_top_k": ppr_res["top_k"],
            "overlap_score": overlap
        })

    return results

def main():
    parser = argparse.ArgumentParser(description="Standard vs Personalized PageRank")
    
    # Ulaz
    parser.add_argument("--generate-n", type=int, default=1000, help="Broj čvorova")
    parser.add_argument("--generate-e", type=int, default=5000, help="Broj veza")
    parser.add_argument("--partitions", type=int, default=8, help="Broj particija")
    
    # PPR opcije
    parser.add_argument("--seeds", type=str, help="Lista seed ID-eva razdvojena zarezom")
    parser.add_argument("--seeds-file", type=str, help="Putanja do fajla sa seed ID-evima")
    
    # Eksperiment
    parser.add_argument("--run-experiment", action="store_true", help="Pokreni batch eksperiment")
    parser.add_argument("--output", type=str, default="results.json", help="Output JSON fajl")

    args = parser.parse_args()
    spark = create_spark_session()
    sc = spark.sparkContext

    if args.run_experiment:
        experiment_data = run_experiment_batch(
            sc, 
            num_graphs=10, 
            N=args.generate_n, 
            E=args.generate_e,
            num_partitions=args.partitions
        )
        
        avg_overlap = sum(r["overlap_score"] for r in experiment_data) / len(experiment_data)
        
        final_output = {
            "summary": {"average_overlap": avg_overlap, "num_graphs": 10},
            "details": experiment_data
        }
        
        with open(args.output, "w") as f:
            json.dump(final_output, f, indent=2)
            
        print(f"\nTestiranje zavrseno. Prosječno preklapanje: {avg_overlap:.4f}")
        print(f"Rezultati sačuvani u: {args.output}")

    else:
        nodes = generate_graph(args.generate_n, args.generate_e)
        
        seed_set = None
        if args.seeds:
            seed_set = set(map(int, args.seeds.split(',')))
        elif args.seeds_file:
            with open(args.seeds_file, 'r') as f:
                seed_set = set(int(line.strip()) for line in f if line.strip())
        
        if seed_set:
            available_ids = set(n.node_id for n in nodes)
            if not seed_set.issubset(available_ids):
                print("Upozorenje: Neki seed čvorovi ne postoje u grafu i biće ignorisani.")
                seed_set = seed_set.intersection(available_ids)

        print("\n--- STANDARD PAGERANK ---")
        std_res = run_pagerank_generic(sc, nodes, seed_nodes=None, num_partitions=args.partitions)
        
        if seed_set:
            print(f"\n--- PERSONALIZED PAGERANK (Seeds: {seed_set}) ---")
            ppr_res = run_pagerank_generic(sc, nodes, seed_nodes=seed_set, num_partitions=args.partitions)
            
            print("\n--- POREĐENJE (Top-10) ---")
            print(f"{'Rank':<5} {'Std Node':<10} {'Std Score':<12} | {'PPR Node':<10} {'PPR Score':<12}")
            print("-" * 60)
            
            for i in range(10):
                std_node, std_score = std_res["top_k"][i]
                ppr_node, ppr_score = ppr_res["top_k"][i]
                print(f"{i+1:<5} {std_node:<10} {std_score:.6f}     | {ppr_node:<10} {ppr_score:.6f}")
                
            overlap = calculate_overlap(std_res["top_k"], ppr_res["top_k"])
            print(f"\n Sličnost Top-10 rezultata: {overlap:.4f}")
        else:
            print("\nNisu zadati seed čvorovi. Prikazujem samo Standard PR.")
            for i, (nid, score) in enumerate(std_res["top_k"], 1):
                print(f"{i}. Node {nid}: {score:.6f}")

    spark.stop()

if __name__ == "__main__":
    main()