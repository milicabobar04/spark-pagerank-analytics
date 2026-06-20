# Distributed PageRank with Apache Spark

## Project Overview
Implementation of PageRank and Personalized PageRank algorithms using Apache Spark RDD API for distributed processing of large graphs.

## Features
- Standard PageRank algorithm
- Personalized PageRank with seed nodes
- Graph generator and edge-list loader
- JSON serialization/deserialization
- Iterative convergence control (epsilon, max iterations)
- Top-K ranking results

## Implementation
- Spark RDD-based distributed computation
- Handling dangling nodes
- Parallel iterative updates
- Fully scalable graph processing

## Output
- Ranked nodes
- Top-K results
- Convergence metrics
