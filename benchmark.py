import time
from pyspark.sql import SparkSession
from graph import generate_graph
from pagerank import run_pagerank as run_std
from pagerank_optimized import run_pagerank_optimized as run_opt

def run_benchmark():
    spark = (
        SparkSession.builder
        .appName("PageRank Optimized - testing")
        .config("spark.executor.extraJavaOptions", "-Djava.security.manager=allow")
        .config("spark.driver.extraJavaOptions",   "-Djava.security.manager=allow")
        .getOrCreate()
    )
    sc = spark.sparkContext
    sc.setLogLevel("ERROR") 

    # Definisemo test slucajeve
    # Format: (Ime, N, E)
    test_cases = [
        ("Test A (Manji)", 5000, 25000),
        ("Test B (Veci)",  30000, 150000)
    ]

    for label, n, e in test_cases:
        nodes = generate_graph(n, e, seed=42)
        # --- STANDARD RUN ---
        start_std = time.time()
        run_std(sc, nodes, max_iter=5, d=0.85, top_k=1)
        time_std = time.time() - start_std
        # --- OPTIMIZED RUN ---
        start_opt = time.time()
        run_opt(sc, nodes, max_iter=5, d=0.85, top_k=1)
        time_opt = time.time() - start_opt
       
        
        speedup = time_std / time_opt
        
        print(f"{label:<15} | {time_std:<10.4f} | {time_opt:<10.4f} | {speedup:<10.2f}x")

    spark.stop()

if __name__ == "__main__":
    run_benchmark()