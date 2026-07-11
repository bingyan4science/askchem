import time
import programasweights as paw

fn = paw.function("32765bb3d684d7fa604d", n_gpu_layers=0)

fn("warm up")

queries = [
    "Who is Suzuki?",
    "What is benzene?",
    "How does NMR work?",
    "suzuki coupling reaction",
    "DOI 10.1021/jacs.5b00001",
    "palladium catalyzed cross coupling",
    "What solvents work for Grignard reactions?",
    "Tell me about lithium ion batteries",
]

header = "{:<50} {:>10}  {}".format("Query", "Time (ms)", "Result")
print(header)
print("-" * 85)

times = []
for q in queries:
    t0 = time.perf_counter()
    result = fn(q).strip().lower()
    elapsed = (time.perf_counter() - t0) * 1000
    times.append(elapsed)
    print("{:<50} {:>8.1f}ms  {}".format(q, elapsed, result))

print("-" * 85)
print("{:<50} {:>8.1f}ms".format("Average", sum(times) / len(times)))
print("{:<50} {:>8.1f}ms".format("Min", min(times)))
print("{:<50} {:>8.1f}ms".format("Max", max(times)))
