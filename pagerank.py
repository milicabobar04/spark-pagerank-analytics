import json
import time
import sys
import os
import argparse
import shutil
from pyspark.sql import SparkSession
from graph import (
    GraphNode,
    generate_graph,
    load_graph_from_edgelist,
    save_graph_to_json,
)
from pagerank_optimized import run_pagerank_optimized

def create_spark_session(app_name: str = "PageRank"):
    spark = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.executor.extraJavaOptions", "-Djava.security.manager=allow")
        .config("spark.driver.extraJavaOptions",   "-Djava.security.manager=allow")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    
    return spark

# PR(v) = (1-d)/N + d * ( sumIncoming(v) + danglingMass/N )
def run_pagerank(
        sc,
        nodes_list: list,      
        d: float = 0.85,       
        max_iter: int = 100,   
        epsilon: float = 1e-6, 
        top_k: int = 10,       
        checkpoint_interval: int = 3,  
) -> dict:
    total_start = time.time()
    N = len(nodes_list)
    initial_rank = 1.0 / N

    # Staticna struktura grafa
    graph_rdd = sc.parallelize([
       (node.node_id, node.out_neighbors)
       for node in nodes_list
    ]).cache()

    # Rankovi cvorova
    ranks = sc.parallelize([
        (node.node_id, initial_rank)
        for node in nodes_list
    ]).cache()

    # Dangling cvorovi
    dangling_rdd = sc.parallelize([
        (node.node_id, True)
        for node in nodes_list
        if node.out_degree == 0
    ]).cache()

    iteration_times = []
    converged = False
    iterations_done = 0

    for iteration in range(max_iter):
        iter_start = time.time()
        
        # Dangling masa
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
                [(neighbor, x[1][1] / len(x[1][0]))
                 for neighbor in x[1][0]]
            ) if len(x[1][0]) > 0 else [])
        )

        sum_incoming = contributions.reduceByKey(lambda a, b: a + b)

        ranks_and_diff = (
            # (node_id, (old_rank, sum_incoming_rank))
            ranks
            .leftOuterJoin(sum_incoming)
            .mapValues(lambda x: (
                x[0],                                 
                x[1] if x[1] is not None else 0.0     
            ))
            # -> (node_id, (old_rank, sum_incoming_rank | 0.0))
            .mapValues(lambda x: {
                "new_rank": (1.0 - d) / N + d * (x[1] + dangling_mass / N),
                "old_rank": x[0]
            })
            # -> (node_id, (new_rank, |new_rank - old_rank|))
            .mapValues(lambda data: (
                data["new_rank"],                       
                abs(data["new_rank"] - data["old_rank"])
            ))
        )

        ranks_and_diff.cache()
        ranks_and_diff.count()  


        total_diff = ranks_and_diff.map(lambda x: x[1][1]).sum()
        new_ranks = ranks_and_diff.mapValues(lambda x: x[0])

        old_ranks = ranks
        ranks = new_ranks
        old_ranks.unpersist()

        iter_time = time.time() - iter_start
        iteration_times.append(iter_time)
        iterations_done += 1

        print(
           f"  Iteracija {iteration + 1:>4d} | "
           f"diff = {total_diff:.10f} | "
           f"danglingMass = {dangling_mass:.6f} | "
           f"vrijeme = {iter_time:.4f}s"
        )

        if total_diff < epsilon:
               converged = True
               break
        
    total_time = time.time() - total_start
    top_k_results = ranks.takeOrdered(top_k, key=lambda x: -x[1])
    graph_rdd.unpersist()
    dangling_rdd.unpersist()


    return {
        "top_k":            top_k_results,
        "iterations":       iterations_done,
        "iteration_times":  iteration_times,
        "total_time":       total_time,
        "converged":        converged,
        "ranks_rdd":        ranks
    }

def save_ranks_to_json(results: dict, filepath: str) -> None:
    output = {
        "top_k":            [[nid, rank] for nid, rank in results["top_k"]],
        "iterations":       results["iterations"],
        "total_time":       round(results["total_time"], 4),
        "converged":        results["converged"],
        "iteration_times":  [round(t, 4) for t in results["iteration_times"]],
    }
    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)


def validate_pagerank_results(
          sc,                     
          ranks_rdd,               # RDD: (nodeId, rank)
          N: int,                  # ukupan broj cvorova
          dangling_node_ids: set,  # set dangling node ID-eva
          tolerance: float = 1e-4, # tolerancija za sumu rangova
) -> dict:
     # Svi rangovi nenegativni
     negative_count = ranks_rdd.filter(lambda x: x[1] < 0).count()
     all_non_negative = (negative_count == 0)

     # Suma rangova priblizno jednaka 1.0
     sum_of_ranks = ranks_rdd.map(lambda x: x[1]).sum()
     sum_valid = abs(sum_of_ranks - 1.0) < tolerance

     # Broj rangova jedank N
     rank_count = ranks_rdd.count()
     count_matches = (rank_count == N)

     # Provjera dangling cvorova
     if dangling_node_ids:
        dangling_ranks = (
            ranks_rdd
            .filter(lambda x: x[0] in dangling_node_ids)
            .collect()  #
        )
     else:
        dangling_ranks = []

     all_checks_passed = (
        all_non_negative
        and sum_valid
        and count_matches
    )
    
     return {
        "all_non_negative":  all_non_negative,
       "sum_of_ranks":      sum_of_ranks,
        "sum_valid":         sum_valid,
        "count_matches":     count_matches,
        "rank_count":        rank_count,
        "dangling_ranks":    dangling_ranks,
        "all_checks_passed": all_checks_passed,
    }

def run_dangling_test(sc):
    print("\n" + "="*50)
    print("TESTA ZA DANGLING ČVOROVE")
    print("="*50)

    
    test_nodes = [
        GraphNode(0, [1]),
        GraphNode(1, []),   
        GraphNode(2, [0])
    ]

    print("Struktura grafa:")
    print("  Node 0 -> [1]")
    print("  Node 1 -> [] (DANGLING!)")
    print("  Node 2 -> [0]")

    print("\nPokrećem PageRank...")
    results = run_pagerank_optimized(
        sc, 
        test_nodes, 
        d=0.85, 
        max_iter=20,     
        epsilon=1e-6,
        top_k=3
    )

    ranks = results["top_k"]
    ranks.sort(key=lambda x: x[0]) 

    print("\nRezultati po čvorovima:")
    total_sum = 0.0
    for node_id, rank in ranks:
        print(f"  Node {node_id}: {rank:.6f}")
        total_sum += rank

    print("-" * 30)
    print(f"  UKUPNA SUMA RANKOVA: {total_sum:.6f}")

    if abs(total_sum - 1.0) < 1e-4:
        print("\nTEST PROŠAO!")
        print("   Suma je ≈ 1.0. Dangling masa se ispravno vraća u sistem.")
    else:
        print("\nTEST PAO!")
        print("   Suma značajno odstupa od 1.0. Negdje gubiš masu.")

def print_validation_report(validation: dict) -> None:
    print("  VALIDACIJA REZULTATA")
    
    status = "PASS" if validation["all_non_negative"] else "FAIL"
    print(f"\n[1] Svi rangovi nenegativni:         {status}")
    
    status = "PASS" if validation["sum_valid"] else "FAIL"
    print(f"\n[2] Suma rangova ≈ 1.0:              {status}")
    print(f"    Suma: {validation['sum_of_ranks']:.10f}")
    
    status = "PASS" if validation["count_matches"] else "FAIL"
    print(f"\n[3] Broj rangova == N:               {status}")
    print(f"    Rangovi: {validation['rank_count']}")
    
    if validation["all_checks_passed"]:
        print("SVE PROVJERE PROŠLE")

def main():
    parser = argparse.ArgumentParser(
        description= "PageRank algoritam sa Apache Spark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Primjeri:
    # Edge-list fajl:
    spark-submit pagerank.py --input graph.csv --output results.json

    # Generisani graf:
    spark-submit pagerank.py --generate-n 1000 --generate-e 5000 --output results.json

    # Sa svim parametrima:
    spark-submit pagerank.py --input graph.csv --output results.json \\
      --graph-output graph.json --damping 0.85 --max-iter 100 \\
      --epsilon 0.000001 --top-k 20
        """
    )
    # Ulaz
    input_group = parser.add_argument_group('Ulazni podaci')
    input_group.add_argument(
        "--input",
        type=str,
        default=None,
        help="Putanja do edge-list  fajla (srcId dstId)"
    )
    input_group.add_argument(
        "--generate-n",
        type=int,
        default=None,
        metavar="N",
        help="Broj čvorova za sintetički graf"
    )
    input_group.add_argument(
        "--generate-e",
        type=int,
        default=None,
        metavar="E",
        help="Broj veza za sintetički graf"
    )
    input_group.add_argument(
        "--generate-seed",
        type=int,
        default=42,
        metavar="SEED",
        help="Seed za generisanje grafa (default: 42)"
    )

    # Izlaz
    output_group = parser.add_argument_group('Izlazni fajlovi')
    output_group.add_argument(
        "--output",
        type=str,
        default="pagerank_results.json",
        help="Izlazni JSON fajl za rezultate (default: pagerank_results.json)"
    )
    output_group.add_argument(
        "--graph-output",
        type=str,
        default=None,
        help="Putanja za čuvanje grafa u JSON (opciono)"
    )

    # PageRank parametri
    pr_group = parser.add_argument_group('PageRank parametri')
    pr_group.add_argument(
        "--damping", "-d",
        type=float,
        default=0.85,
        metavar="D",
        help="Damping faktor (default: 0.85)"
    )
    pr_group.add_argument(
        "--max-iter",
        type=int,
        default=100,
        metavar="ITER",
        help="Maksimalan broj iteracija (default: 100)"
    )
    pr_group.add_argument(
        "--epsilon",
        type=float,
        default=1e-6,
        metavar="EPS",
        help="Epsilon za konvergenciju (default: 0.000001)"
    )
    pr_group.add_argument(
        "--top-k",
        type=int,
        default=10,
        metavar="K",
        help="Broj top čvorova za prikaz (default: 10)"
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Pokreni validaciju rezultata"
    )
    # Dangling test
    parser.add_argument(
        "--test-dangling",
        action="store_true",
        help="Pokreni unit test na malom grafu za provjeru dangling logike"
    )

    args = parser.parse_args()

    spark = create_spark_session()
    sc = spark.sparkContext

    if args.test_dangling:
        run_dangling_test(sc)
        spark.stop()
        return

    if args.input is None and args.generate_n is None:
        parser.error("Mora se zadati ili --input ili --generate-n i --generate-e")

    if args.generate_n is not None and args.generate_e is None:
        parser.error("Ako je --generate-n zadat, mora biti i --generate-e")

    if args.input is not None:
        print(f"\nUčitavanje grafa iz: {args.input}")
        nodes = load_graph_from_edgelist(args.input)
    else:
        print(f"\nGenerisanje grafa:")
        print(f"       N = {args.generate_n} čvorova")
        print(f"       E = {args.generate_e} veza")
        print(f"       seed = {args.generate_seed}")
        nodes = generate_graph(args.generate_n, args.generate_e, args.generate_seed)

    num_edges = sum(n.out_degree for n in nodes)
    num_dangling = sum(1 for n in nodes if n.out_degree == 0)

    print(f"\nGraf učitan:")
    print(f"       Čvorovi:  {len(nodes)}")
    print(f"       Veze:     {num_edges}")
    print(f"       Dangling: {num_dangling}")

    if args.graph_output: 
       save_graph_to_json(nodes, args.graph_output)
       print(f"\nGraf sačuvan: {args.graph_output}")

    
    print(f"\nParametri PageRank:")
    print(f"       Damping:  {args.damping}")
    print(f"       MaxIter:  {args.max_iter}")
    print(f"       Epsilon:  {args.epsilon}")
    print(f"       Top-K:    {args.top_k}")
    print()

    results = run_pagerank(
        sc=sc,
        nodes_list=nodes,
        d=args.damping,
        max_iter=args.max_iter,
        epsilon=args.epsilon,
        top_k=args.top_k,
    )
    print(f"  REZULTATI")
    print(f"  Iteracije:       {results['iterations']}")
    print(f"  Konvergiralo:    {results['converged']}")
    print(f"  Ukupno vrijeme:  {results['total_time']:.4f}s")
    print(f"  Prosječno/iter:  {results['total_time']/results['iterations']:.4f}s")
    
    print(f"\n  Top-{args.top_k} čvorova:")
    print(f"  {'#':<5} {'NodeID':<10} {'PageRank':<15}")
    print(f"  {'-'*30}")
    for i, (node_id, rank) in enumerate(results["top_k"], 1):
        print(f"  {i:<5} {node_id:<10} {rank:<15.10f}")
    
    if args.validate:
        dangling_ids = set(n.node_id for n in nodes if n.out_degree == 0)
        validation = validate_pagerank_results(
            sc=sc,
            ranks_rdd=results["ranks_rdd"],
            N=len(nodes),
            dangling_node_ids=dangling_ids,
            tolerance=1e-4
        )
        print_validation_report(validation)

    save_ranks_to_json(results, args.output)
    print(f"\nRezultati sačuvani: {args.output}")

    spark.stop()

if __name__ == "__main__":
    main()