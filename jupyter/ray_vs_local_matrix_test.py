import torch
import ray
import time
import os

# --- Ray Matrix Test Configuration ---
RAY_ADDRESS = "ray://192.168.12.91:31001"
NUM_TASKS = 3
MATRIX_SIZE = 14000
ITERATIONS = 30
# ---------------------------------------

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

# ---------- Heavy Matrix Multiplication Function ----------
def local_matrix_multiply(task_id: int, size: int = 14000, iterations: int = 30):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"(Local) Task {task_id} - Running on {device}...")

    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)

    result = torch.zeros(size, size, device=device)
    start = time.time()

    for i in range(iterations):
        temp = torch.matmul(a, b)
        result += temp
        if i % 5 == 0:
            log(f"(Task {task_id}) Iteration {i}/{iterations}")

    duration = time.time() - start
    log(f"(Local) Task {task_id} - Time: {duration:.2f}s - Result sum: {result.sum().item():.2f}")
    return duration

# ---------- Ray Remote Function ----------
@ray.remote(num_gpus=1)
def ray_matrix_multiply(task_id: int, size: int = 14000, iterations: int = 30):
    return local_matrix_multiply(task_id, size, iterations)

# ---------- Test Runs ----------
def run_local_tasks(num_tasks=2, size=14000, iterations=30):
    log("=== LOCAL EXECUTION ===")
    times = []
    for i in range(num_tasks):
        times.append(local_matrix_multiply(i, size, iterations))
    avg = sum(times) / len(times)
    log(f"Average time (local): {avg:.2f}s")

def run_ray_tasks(num_tasks=2, size=14000, iterations=30):
    log("=== RAY EXECUTION ===")
    ray.init(address=RAY_ADDRESS)  # Specify the Ray cluster head
    futures = [ray_matrix_multiply.remote(i, size, iterations) for i in range(num_tasks)]
    results = ray.get(futures)
    avg = sum(results) / len(results)
    log(f"Average time (Ray): {avg:.2f}s")
    ray.shutdown()

# ---------- Main Execution Block ----------
if __name__ == "__main__":
    num_tasks = NUM_TASKS
    matrix_size = MATRIX_SIZE
    iterations = ITERATIONS

    # Run a single local task for comparison
    run_local_tasks(1, matrix_size, iterations)
    
    # Run multiple tasks on the Ray cluster
    run_ray_tasks(5, matrix_size, iterations)