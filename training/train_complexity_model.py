#!/usr/bin/env python3
"""
PromptShield AI — Prompt Complexity Scoring Model Training Pipeline.
===================================================================
Trains a local multi-class classifier to assess prompt complexity across 4 tiers:
- simple: Basic syntax questions, single-turn lookups, elementary explanations
- moderate: Single-file scripts, standard CRUD APIs, basic data analysis
- complex: Multi-service systems, concurrency, security audits, database schemas
- extreme: Distributed consensus, lock-free memory, multi-agent frameworks, formal proofs

Saves to:
- artifacts/complexity_model/model.pkl
- artifacts/complexity_model/metadata.json
- artifacts/complexity_model/metrics.json
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


COMPLEXITY_DATASET: List[Tuple[str, str]] = [
    # 1. SIMPLE (50 samples)
    ("How do I check the length of a string in Python?", "simple"),
    ("What does the HTTP 404 status code mean?", "simple"),
    ("Write a one-line bash command to list all files in reverse chronological order.", "simple"),
    ("How do I convert an integer to a float in JavaScript?", "simple"),
    ("Explain the difference between let and const in ES6 in two sentences.", "simple"),
    ("What is the syntax for a for-loop in Go?", "simple"),
    ("How do I exit vim without saving changes?", "simple"),
    ("What does git pull do?", "simple"),
    ("Write a basic Python print statement saying Hello World.", "simple"),
    ("How do I comment out a block of code in HTML?", "simple"),
    ("What is the default port for PostgreSQL?", "simple"),
    ("How do I add an element to an array in Ruby?", "simple"),
    ("Explain what a boolean variable is in programming.", "simple"),
    ("What command shows the current working directory in Linux terminal?", "simple"),
    ("How do I find the maximum number in a list in Python using built-in functions?", "simple"),
    ("What is JSON and what does the acronym stand for?", "simple"),
    ("How do I format a date string as YYYY-MM-DD in Python?", "simple"),
    ("What is the difference between null and undefined in JavaScript?", "simple"),
    ("How do I reverse a string in Python using slicing?", "simple"),
    ("What does SQL SELECT * FROM users do?", "simple"),
    ("How do I sort a list of numbers in Python?", "simple"),
    ("What does the git status command do?", "simple"),
    ("How do I declare a constant variable in TypeScript?", "simple"),
    ("Write a command to create a new directory in bash.", "simple"),
    ("What is the difference between == and === in JavaScript?", "simple"),
    ("How do I check my Python version in the terminal?", "simple"),
    ("What is a dictionary in Python?", "simple"),
    ("How do I convert a string to lowercase in Java?", "simple"),
    ("Write a CSS selector to select all paragraphs with class intro.", "simple"),
    ("What does the break statement do in a loop?", "simple"),
    ("How do I read a text file line by line in Python?", "simple"),
    ("What is the purpose of the return statement in a function?", "simple"),
    ("How do I round a floating point number to two decimal places in Python?", "simple"),
    ("What does pip install do?", "simple"),
    ("How do I concatenate two strings in C++?", "simple"),
    ("What is an array index out of bounds error?", "simple"),
    ("How do I check if a key exists in a Python dict?", "simple"),
    ("What is HTML and what are tags?", "simple"),
    ("How do I print the current date in JavaScript?", "simple"),
    ("What is the command to remove a file in Linux?", "simple"),
    ("How do I install an npm package locally?", "simple"),
    ("What does git commit -m do?", "simple"),
    ("How do I import a module in Python?", "simple"),
    ("What does the typeof operator return in JavaScript?", "simple"),
    ("How do I find the length of an array in C#?", "simple"),
    ("What is a syntax error in programming?", "simple"),
    ("How do I write a single line comment in Python?", "simple"),
    ("What is the default port for HTTP traffic?", "simple"),
    ("How do I capitalize the first letter of a string in Python?", "simple"),
    ("What does the len() function do in Python?", "simple"),

    # 2. MODERATE (50 samples)
    ("Write a Python function to sort a list of dictionaries by a specific key with error handling.", "moderate"),
    ("Create a FastAPI endpoint that accepts user signup JSON and validates email format with Pydantic.", "moderate"),
    ("Write a React functional component for an accordion FAQ with expand and collapse state.", "moderate"),
    ("How do I calculate a 7-day rolling average in pandas on a sales dataframe with timestamps?", "moderate"),
    ("Write a Python script to parse a CSV file, filter rows where age > 30, and write to JSON.", "moderate"),
    ("Create a Dockerfile for a Node.js Express application with multi-stage build.", "moderate"),
    ("Implement binary search in Python to find the index of a target element in an ascending array.", "moderate"),
    ("Write a SQL query using GROUP BY and HAVING to find customers who placed more than 5 orders.", "moderate"),
    ("Build a Flask middleware to log response time and status code for each HTTP request.", "moderate"),
    ("Write a bash script to backup a MySQL database daily and compress it with gzip.", "moderate"),
    ("Create an Express route that handles image file uploads using multer and saves to disk.", "moderate"),
    ("Implement an LRU cache decorator in Python using functools and lru_cache.", "moderate"),
    ("Write a CSS layout for a 3-column pricing card grid using CSS Grid and Flexbox.", "moderate"),
    ("Create a simple CLI in Python using argparse with flags for input file and verbosity.", "moderate"),
    ("Write a unit test suite in pytest for an authentication helper that validates JWT expirations.", "moderate"),
    ("How do I perform exploratory data analysis with seaborn pairplots and missing value imputation?", "moderate"),
    ("Write an async Python function to fetch 5 URLs concurrently using aiohttp with a timeout.", "moderate"),
    ("Create a reusable React custom hook to manage localStorage synchronization.", "moderate"),
    ("Write a MongoDB aggregation pipeline to calculate average revenue per product category.", "moderate"),
    ("Implement a basic stack data structure in C++ with push, pop, and top methods.", "moderate"),
    ("Write a Python script to parse an XML file and convert it into a pandas DataFrame.", "moderate"),
    ("Create an Express.js middleware function that validates bearer token presence in headers.", "moderate"),
    ("Write a React component with useState and useEffect to fetch user list from an API.", "moderate"),
    ("Build a Flask route that handles file upload and validates image dimensions using PIL.", "moderate"),
    ("Write a SQL query with subqueries to find the second highest salary in an employees table.", "moderate"),
    ("Create a Python script using requests to paginate through a REST API and save results to CSV.", "moderate"),
    ("Implement a debounce function in TypeScript for search input autocomplete.", "moderate"),
    ("Write a Docker Compose file running a Node.js web server and a Redis cache container.", "moderate"),
    ("Create a custom pytest fixture that sets up and tears down an in-memory SQLite database.", "moderate"),
    ("Write a Python function to validate credit card numbers using Luhn's algorithm.", "moderate"),
    ("Build a responsive navigation bar in React with hamburger menu toggle on mobile.", "moderate"),
    ("Write a bash script to monitor CPU usage every 5 seconds and alert if it exceeds 90%.", "moderate"),
    ("Create a FastAPI endpoint with Pydantic model for updating user password with regex validation.", "moderate"),
    ("Write a Python script to scrape product titles and prices from an HTML page using BeautifulSoup.", "moderate"),
    ("Implement a queue data structure using two stacks in Python with amortized O(1) operations.", "moderate"),
    ("Write a SQL query using window functions (ROW_NUMBER) to deduplicate user records.", "moderate"),
    ("Create a React hook useLocalStorage that synchronizes state with browser storage.", "moderate"),
    ("Build an asynchronous worker in Python that processes tasks from a multiprocessing Queue.", "moderate"),
    ("Write a Go function to parse a JSON configuration file and unmarshal into a struct.", "moderate"),
    ("Create an Express endpoint with rate limiting using express-rate-limit middleware.", "moderate"),
    ("Write a Python script to send an email with an attachment using smtplib and email.mime.", "moderate"),
    ("Implement an in-memory TTL cache in Python with automatic key expiration.", "moderate"),
    ("Create a CSS Grid layout with responsive sidebar and sticky header for an admin panel.", "moderate"),
    ("Write a unit test in Jest to verify that a button click triggers an API call with expected payload.", "moderate"),
    ("Build a CLI tool in Go using Cobra with commands for config init and run.", "moderate"),
    ("Write a Python script to merge multiple PDF files into one using pypdf.", "moderate"),
    ("Create a FastAPI dependency to extract and verify API keys from request headers.", "moderate"),
    ("Write a SQL migration script to add an index on user email and foreign key constraint.", "moderate"),
    ("Implement a binary tree depth-first traversal (pre-order, in-order, post-order) in Python.", "moderate"),
    ("Build a simple WebSocket server in Node.js using ws that broadcasts messages to connected clients.", "moderate"),

    # 3. COMPLEX (50 samples)
    ("Design a scalable microservices architecture for an e-commerce platform handling 50k RPS with Stripe payments, idempotency keys, and Kafka event streaming.", "complex"),
    ("Perform a comprehensive STRIDE threat modeling analysis and security audit for an OAuth2.0 PKCE authentication flow with refresh token rotation.", "complex"),
    ("Write a high-throughput event processing worker in Python using asyncio, Redis streams, and PostgreSQL with connection pooling and dead-letter queue.", "complex"),
    ("Implement a B-tree indexing engine from scratch in Go with node splitting, merging, and binary disk persistence.", "complex"),
    ("Create an end-to-end CI/CD pipeline using Terraform, AWS ECS Fargate, GitHub Actions, and canary deployments with automated rollback on CloudWatch error alarms.", "complex"),
    ("Audit this smart contract for reentrancy bugs, flash loan attack vectors, and integer overflow edge cases with formal Slither annotations.", "complex"),
    ("Design a distributed caching tier with Redis Cluster, consistent hashing, cache-aside strategy, and cache stampede prevention via probabilistic early expiration.", "complex"),
    ("Build an autonomous agent workflow in Python with tool-calling capabilities, vector memory store retrieval, and human-in-the-loop approval hooks.", "complex"),
    ("Write an SQL migration script for a zero-downtime database schema refactor splitting a 20-million row monolith table into partitioned tables.", "complex"),
    ("Implement a Raft distributed consensus state machine in Rust with leader election, log replication, and heartbeats.", "complex"),
    ("Build a secure multi-tenant SaaS authorization system with granular RBAC and ABAC using Open Policy Agent (OPA) and Casbin.", "complex"),
    ("Analyze a packet capture file and detect sophisticated DNS covert channels, domain fronting, and TLS fingerprint evasion techniques.", "complex"),
    ("Design a real-time collaborative document editor with Operational Transformation (OT) or CRDT algorithms over WebSockets.", "complex"),
    ("Implement a custom Kubernetes operator in Go with custom resource definitions (CRDs), reconciler loops, and leader election across pods.", "complex"),
    ("Build an ETL data lakehouse pipeline processing 500GB daily streaming logs with Apache Spark, Delta Lake, and automated schema evolution.", "complex"),
    ("Design a distributed job scheduling engine with DAG dependency resolution, retry budgets, and worker failure detection.", "complex"),
    ("Develop a comprehensive IAM policy analyzer detecting privilege escalation paths across AWS IAM roles and policies.", "complex"),
    ("Create a full-stack real-time trading simulator with WebSockets, order matching engine, and Redis cache.", "complex"),
    ("Implement two-phase locking and deadlock detection graph algorithms for a relational database storage engine.", "complex"),
    ("Build a federated GraphQL gateway aggregating 6 microservices with distributed tracing and rate limiting.", "complex"),
    ("Design an asynchronous event-driven microservices architecture using Apache Kafka, schema registry, and outbox pattern.", "complex"),
    ("Implement a distributed rate limiting system using Redis token bucket algorithm with sliding window log.", "complex"),
    ("Design a real-time analytics aggregation pipeline processing clickstream events with Apache Flink and ClickHouse.", "complex"),
    ("Implement a multi-tier caching architecture with Redis Cluster, L1 in-memory cache, and cache invalidation webhooks.", "complex"),
    ("Perform a threat model analysis and penetration testing plan for a cloud-native Kubernetes microservices deployment.", "complex"),
    ("Build an automated database failover manager in Python with health checks, raft consensus, and DNS routing update.", "complex"),
    ("Design a zero-downtime blue-green deployment strategy on AWS ECS with Application Load Balancer target group shifting.", "complex"),
    ("Implement an OAuth2.0 authorization server in Go supporting authorization code, client credentials, and PKCE.", "complex"),
    ("Build a custom Prometheus exporter in Python collecting metrics from an internal messaging queue with histogram latency buckets.", "complex"),
    ("Design a multi-region active-active database replication architecture with conflict-free replicated data types (CRDTs).", "complex"),
    ("Implement a distributed task scheduler in Go with worker heartbeats, leasing mechanism, and dead-letter queues.", "complex"),
    ("Build an end-to-end telemetry pipeline using OpenTelemetry collectors, Jaeger distributed tracing, and Prometheus metrics.", "complex"),
    ("Design a high-concurrency order book matching engine in Rust with price-time priority and atomic order execution.", "complex"),
    ("Implement dynamic role-based access control (RBAC) with hierarchical permissions and Casbin policy engine in FastAPI.", "complex"),
    ("Create a distributed file storage cluster in Python with data chunking, replication factor 3, and md5 integrity verification.", "complex"),
    ("Design an ETL pipeline with Apache Airflow DAGs, Great Expectations data validation, and Snowflake data warehouse loading.", "complex"),
    ("Implement a custom Envoy filter in C++ using WebAssembly (Wasm) for proprietary request authentication and header mutation.", "complex"),
    ("Build a resilient webhook delivery service with exponential backoff, jitter, dead-letter queue, and HMAC signing.", "complex"),
    ("Design a scalable search infrastructure using Elasticsearch cluster with sharding, index lifecycle management, and custom analyzers.", "complex"),
    ("Implement an in-memory columnar storage engine in C++ with dictionary encoding and vectorized query execution.", "complex"),
    ("Design a secure API gateway with mTLS authentication, token introspection, dynamic routing, and WAF rules.", "complex"),
    ("Build an autonomous agent swarming system in LangGraph with state persistence, checkpointing, and tool execution safeguards.", "complex"),
    ("Implement two-phase commit (2PC) coordinator for distributed transactions across heterogeneous relational databases.", "complex"),
    ("Design a high-throughput log ingestion service handling 100k events/sec using Go goroutine worker pools and zero-allocation ring buffers.", "complex"),
    ("Perform a static and dynamic binary analysis of an obfuscated ELF executable to identify hidden command-and-control communication.", "complex"),
    ("Implement a gossip protocol in Python for decentralized cluster node discovery and failure detection.", "complex"),
    ("Design a multi-tenant isolation architecture on Kubernetes using network policies, Cilium eBPF, and Vault secret injection.", "complex"),
    ("Build an asynchronous video transcoding pipeline using AWS SQS, Lambda workers, and FFmpeg with progressive chunk processing.", "complex"),
    ("Implement a persistent LSM-tree (Log-Structured Merge-tree) storage engine with write-ahead log (WAL) and SSTable compaction.", "complex"),
    ("Design a GraphQL federation gateway using Apollo Federation v2 with subgraph schema composition and entity caching.", "complex"),

    # 4. EXTREME (50 samples)
    ("Architect a fault-tolerant, globally distributed Byzantine Fault Tolerant (BFT) consensus protocol with formal TLA+ specifications, cross-shard atomic commit, and zero-knowledge proof state compression.", "extreme"),
    ("Develop a lock-free, wait-free multi-producer multi-consumer ring buffer in C++23 with memory ordering semantics (acquire-release fences) for ultra-low-latency financial trading.", "extreme"),
    ("Formally verify a kernel-level eBPF security sandbox using Coq or Isabelle, proving memory safety and non-interference against kernel privilege escalation exploits.", "extreme"),
    ("Design an autonomous multi-agent distributed operating system runtime with asynchronous message actors, capability-based security tokens, and verifiable multi-party computation.", "extreme"),
    ("Reverse engineer and develop an automated exploit payload chain combining kernel heap Feng Shui, ROP gadget synthesizers, and hypervisor escape mitigations for x86_64 virtualization.", "extreme"),
    ("Synthesize a custom optimizing compiler backend with SSA intermediate representation, polyhedral loop transformation, and LLVM machine code generation for sparse tensor accelerators.", "extreme"),
    ("Implement a zero-knowledge succinct non-interactive argument of knowledge (zk-SNARK) circuit compiler with Plonk proof systems and elliptic curve pairing cryptography.", "extreme"),
    ("Architect a planetary-scale distributed database engine with spanner-like TrueTime API, Paxos replication groups, multi-version concurrency control (MVCC), and distributed deadlock detection.", "extreme"),
    ("Formally verify cryptographic zero-knowledge rollups with polynomial commitment schemes and Halo2 recursive SNARKs.", "extreme"),
    ("Design a bare-metal hypervisor in Rust for ARM64 with stage-2 address translation, secure monitor calls, and formal isolation proofs.", "extreme"),
    ("Develop an automated binary symbolic execution and SMT constraint solver engine to synthesize zero-day remote code execution exploits.", "extreme"),
    ("Build a distributed hardware transactional memory simulator with cache coherence protocols (MESI/MOESI) and cache line invalidation models.", "extreme"),
    ("Synthesize an asynchronous actor runtime with linearizability checking and verified model-checked safety invariants.", "extreme"),
    ("Implement an end-to-end quantum circuit simulator with density matrix decoherence models and Clifford+T gate synthesis.", "extreme"),
    ("Architect a verifiable decentralized computing protocol with trusted execution environment (TEE) attestation and homomorphic encryption.", "extreme"),
    ("Develop a real-time kernel memory allocator with deterministic O(1) latency, zero fragmentation bounds, and hardware isolation.", "extreme"),
    ("Build an autonomous multi-agent swarming framework with formal game-theoretic Nash equilibrium coordination under adversarial network partitions.", "extreme"),
    ("Design a formal verification engine for smart contracts using abstract interpretation and Horn clause constraint solvers.", "extreme"),
    ("Implement a low-level JIT compiler for WebAssembly targeting RISC-V with register allocation via graph coloring.", "extreme"),
    ("Architect a quantum-resistant lattice-based cryptographic key exchange protocol with side-channel timing attack resilience.", "extreme"),
    ("Architect a zero-knowledge verifiable state machine with recursive zk-SNARKs (PLONK + Halo2) and polynomial commitment schemes.", "extreme"),
    ("Develop a wait-free lock-free multi-producer multi-consumer bounded ring buffer in C++23 with C++ memory model acquire-release fences.", "extreme"),
    ("Formally verify a kernel-level memory allocator using Coq, proving non-interference, memory safety, and absence of spatial-temporal violations.", "extreme"),
    ("Design a custom optimizing JIT compiler backend targeting RISC-V with SSA conversion, dominator tree analysis, and Chaitin-Briggs graph coloring register allocation.", "extreme"),
    ("Synthesize a Byzantine Fault Tolerant (BFT) consensus protocol with DAG-based block ordering, Narwhal-Tusk architecture, and formal TLA+ safety proofs.", "extreme"),
    ("Develop an automated binary symbolic execution engine using SMT solver integration (Z3) to discover and synthesize zero-day ROP gadget exploit chains.", "extreme"),
    ("Architect a planetary-scale distributed database engine with hardware TrueTime clock synchronization, Paxos replication groups, and multi-version concurrency control.", "extreme"),
    ("Design a bare-metal type-1 hypervisor in Rust for ARM64 with stage-2 memory translation, trap-and-emulate virtualization, and formal isolation verification.", "extreme"),
    ("Implement a quantum computing circuit compiler and density matrix simulator with Clifford+T decomposition and decoherence error models.", "extreme"),
    ("Formally verify cryptographic zero-knowledge rollup smart contracts using abstract interpretation and Horn clause constraint solvers.", "extreme"),
    ("Develop a real-time hard kernel scheduler with deterministic O(1) task switching bounds, zero priority inversion, and formally verified safety invariants.", "extreme"),
    ("Architect an asynchronous actor runtime with linearizability checking, verified model-checked state transitions, and lock-free message stealing queues.", "extreme"),
    ("Reverse engineer a hypervisor escape vulnerability exploiting nested virtualization page tables and speculative execution side-channels (Spectre-V2).", "extreme"),
    ("Synthesize an automated theorem prover for first-order logic with equality using superposition calculus and Knuth-Bendix term ordering.", "extreme"),
    ("Architect a verifiable decentralized multi-party computation (MPC) protocol with Shamir secret sharing and homomorphic encryption under Byzantine adversaries.", "extreme"),
    ("Implement a high-performance custom lock-free memory allocator using thread-local maglev heaps and atomic CAS bitmap allocation.", "extreme"),
    ("Design a formally verified network protocol stack in Idris with dependent types, mathematically proving absence of buffer overflows and protocol deadlocks.", "extreme"),
    ("Develop a hardware transactional memory simulator with MESI/MOESI cache coherence protocols and store buffer snooping architectures.", "extreme"),
    ("Synthesize an autonomous multi-agent operating system runtime with asynchronous microkernel IPC, capability-based security, and verified memory isolation.", "extreme"),
    ("Architect a quantum-resistant post-quantum cryptographic lattice-based digital signature scheme (Dilithium/Falcon) with formal side-channel resilience proofs.", "extreme"),
    ("Develop a custom polyhedral loop optimization compiler pass with Affine transformations and OpenMP/CUDA code generation for tensor accelerators.", "extreme"),
    ("Formally verify cryptographic key exchange protocols using Tamarin Prover, proving secrecy and forward secrecy against active Dolev-Yao adversaries.", "extreme"),
    ("Implement a bare-metal bootloader and UEFI firmware runtime in Rust with secure boot cryptographic signature verification and TPM measured boot.", "extreme"),
    ("Architect a high-performance lock-free distributed hash table with hopscotch hashing, memory-mapped NVMe storage, and RDMA direct memory access.", "extreme"),
    ("Synthesize an automated exploit generation engine utilizing concolic execution, taint analysis, and dynamic binary instrumentation (DBI) via Frida/QEMU.", "extreme"),
    ("Design a fault-tolerant asynchronous distributed shared memory architecture with release consistency and causal message logging.", "extreme"),
    ("Formally verify concurrent lock-free queue algorithms using Iris in Coq with higher-order concurrent separation logic.", "extreme"),
    ("Develop a real-time kernel memory compression allocator using custom LZ4 streaming primitives and page table dirty tracking.", "extreme"),
    ("Synthesize a custom microarchitecture simulator in C++ modeling out-of-order execution, branch target buffers, and superscalar pipeline hazard stalls.", "extreme"),
    ("Architect an end-to-end zero-knowledge proof private smart contract execution environment with zk-EVM bytecode interpretation and recursive SNARK aggregation.", "extreme")
]


def train_complexity_model(
    artifact_model_path: Path,
    artifact_metadata_path: Path,
    artifact_metrics_path: Path,
    test_size: float = 0.20,
    random_state: int = 42
) -> Dict:
    """
    Trains and evaluates the Prompt Complexity Scoring Model.
    """
    print("=" * 70)
    print("PROMPTSHIELD AI — PROMPT COMPLEXITY MODEL TRAINING PIPELINE")
    print("=" * 70)

    df = pd.DataFrame(COMPLEXITY_DATASET, columns=["prompt", "complexity_tier"])
    print(f"Total labeled complexity samples: {len(df)}")
    print("\nTier Distribution:")
    for tier, count in df["complexity_tier"].value_counts().items():
        print(f"  {tier:<15}: {count} ({count/len(df)*100:.1f}%)")

    X = df["prompt"]
    y = df["complexity_tier"]

    # Stratified Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    print(f"\nTrain Split: {len(X_train)} samples")
    print(f"Test Split:  {len(X_test)} samples (held-out)")

    # Build Pipeline: TF-IDF + Logistic Regression
    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2),
                min_df=1,
                max_features=8000,
                sublinear_tf=True
            )
        ),
        (
            "classifier",
            LogisticRegression(
                solver="liblinear",
                C=3.0,
                class_weight="balanced",
                random_state=random_state
            )
        )
    ])

    print("\nTraining complexity classification pipeline...")
    pipeline.fit(X_train, y_train)

    # Evaluate on Test Set
    y_pred = pipeline.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    weighted_f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS (HELD-OUT TEST SET)")
    print("=" * 70)
    print(f"Accuracy:    {accuracy * 100:.2f}%")
    print(f"Weighted F1: {weighted_f1 * 100:.2f}%")
    print(f"Macro F1:    {macro_f1 * 100:.2f}%")

    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    # Build Model Artifact
    classes = list(pipeline.named_steps["classifier"].classes_)
    tier_order = {"simple": 1, "moderate": 2, "complex": 3, "extreme": 4}

    artifact = {
        "model_name": "PromptShield Complexity Model",
        "pipeline": pipeline,
        "classes": classes,
        "tier_order": tier_order,
        "metrics": {
            "accuracy": round(accuracy, 4),
            "weighted_f1": round(weighted_f1, 4),
            "macro_f1": round(macro_f1, 4)
        }
    }

    # Save artifact
    artifact_model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, artifact_model_path)
    print(f"\nModel artifact saved to: {artifact_model_path.resolve()}")

    # Save metadata
    metadata = {
        "model_name": "PromptShield Complexity Scoring Model",
        "version": "2.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "algorithm": "TfidfVectorizer(1,2) + Balanced Logistic Regression(C=3.0)",
        "tiers": ["simple", "moderate", "complex", "extreme"],
        "tier_descriptions": {
            "simple": "Basic syntax lookups, elementary questions, single-turn prompts",
            "moderate": "Standard CRUD APIs, single-file scripts, basic data analysis",
            "complex": "Multi-component architectures, concurrency, security audits",
            "extreme": "Distributed consensus, formal verification, low-level concurrency"
        },
        "performance": {
            "test_accuracy": round(accuracy, 4),
            "weighted_f1": round(weighted_f1, 4),
            "macro_f1": round(macro_f1, 4)
        }
    }

    with open(artifact_metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Save metrics
    metrics_data = {
        "model_name": "PromptShield Complexity Model",
        "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        "total_samples": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "overall_metrics": {
            "accuracy": round(accuracy, 4),
            "weighted_f1": round(weighted_f1, 4),
            "macro_f1": round(macro_f1, 4)
        },
        "per_class_metrics": report_dict
    }

    with open(artifact_metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)

    return metrics_data


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    artifact_m = repo_root / "artifacts" / "complexity_model" / "model.pkl"
    artifact_meta = repo_root / "artifacts" / "complexity_model" / "metadata.json"
    artifact_met = repo_root / "artifacts" / "complexity_model" / "metrics.json"

    train_complexity_model(artifact_m, artifact_meta, artifact_met)
