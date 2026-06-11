
import json
import time
from pyspark.sql import SparkSession
from graph import GraphNode


def create_spark_session(app_name: str = "PageRank-Optimized"):
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


def run_pagerank_optimized(
    sc,
    nodes_list: list,
    d: float = 0.85,
    max_iter: int = 100,
    epsilon: float = 1e-6,
    top_k: int = 10,
    num_partitions: int = 8,
) -> dict:
   
    total_start = time.time()

    N = len(nodes_list)
    initial_rank = 1.0 / N

    graph_rdd = (
        sc.parallelize(
            [(node.node_id, node.out_neighbors) for node in nodes_list],
            numSlices=num_partitions  
        )
        .partitionBy(num_partitions)  
        .persist()  
    )

    ranks = sc.parallelize(
        [(node.node_id, initial_rank) for node in nodes_list],
        numSlices=num_partitions
    ).partitionBy(num_partitions)

    dangling_ids = set(n.node_id for n in nodes_list if n.out_degree == 0)
    dangling_broadcast = sc.broadcast(dangling_ids) 
    
    dangling_rdd = (
        sc.parallelize(
            [(node.node_id, True) for node in nodes_list if node.out_degree == 0],
            numSlices=max(1, num_partitions // 2)  
        )
        .partitionBy(num_partitions)
        .persist()
    )

    iteration_times = []
    converged = False
    iterations_done = 0

    graph_rdd.count()

    for iteration in range(max_iter):
        iter_start = time.time()

        dangling_mass = (
            dangling_rdd
            .join(ranks) 
            .map(lambda x: x[1][1])
            .sum()
        )

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
            ranks
            .leftOuterJoin(sum_incoming)
            .mapValues(lambda x: (
                x[0],                                 
                x[1] if x[1] is not None else 0.0     
            ))
            .mapValues(lambda x: (
                (1.0 - d) / N + d * (x[1] + dangling_mass / N),
                x[0]
            ))
            .mapValues(lambda x: (
                x[0],               
                abs(x[0] - x[1])    
            ))
        )
        ranks_and_diff.cache()
        ranks_and_diff.count()

        total_diff = (
            ranks_and_diff
            .map(lambda x: x[1][1]) # Uzimamo diff
            .sum()
        )

        new_ranks = ranks_and_diff.mapValues(lambda x: x[0])

        ranks.unpersist()
        ranks = new_ranks

        iter_time = time.time() - iter_start
        iteration_times.append(iter_time)
        iterations_done += 1

        print(
            f"  [OPTIMIZED] Iter {iteration + 1:>4d} | "
            f"diff = {total_diff:.10f} | "
            f"vrijeme = {iter_time:.4f}s"
        )

        if total_diff < epsilon:
            converged = True
            break

    total_time = time.time() - total_start

    dangling_broadcast.unpersist()

    top_k_results = ranks.takeOrdered(top_k, key=lambda x: -x[1])

    graph_rdd.unpersist()
    dangling_rdd.unpersist()

    return {
        "top_k":            top_k_results,
        "iterations":       iterations_done,
        "iteration_times":  iteration_times,
        "total_time":       total_time,
        "converged":        converged,
    }