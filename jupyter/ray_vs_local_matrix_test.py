import torch
import ray
import time

# --- Ray Matrix Test Configuration ---
RAY_ADDRESS = "ray://192.168.12.91:31001"
NUM_TASKS = 5
MATRIX_SIZE = 14000
ITERATIONS = 10
# ---------------------------------------

ray.shutdown()

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

# ---------- Función pesada de multiplicación ----------
def local_matrix_multiply(task_id: int, size: int = 14000, iterations: int = 30):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"(Local) Tarea {task_id} - Ejecutando en {device}...")

    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    result = torch.zeros(size, size, device=device)

    if device == "cuda":
        torch.cuda.synchronize()
    start = time.time()

    for i in range(iterations):
        temp = torch.matmul(a, b)
        result += temp
        if i % 5 == 0:
            log(f"(Tarea {task_id}) Iteración {i}/{iterations}")

    if device == "cuda":
        torch.cuda.synchronize()
    duration = time.time() - start

    log(f"(Local) Tarea {task_id} - Tiempo: {duration:.4f}s - Suma resultado: {result.sum().item():.2f}")
    return duration

# ---------- Ray ----------
@ray.remote(num_gpus=1)
def ray_matrix_multiply(task_id: int, size: int = 14000, iterations: int = 30):
    return local_matrix_multiply(task_id, size, iterations)

# ---------- Pruebas ----------
def run_local_tasks(num_tasks=2, size=14000, iterations=30):
    log("=== EJECUCIÓN LOCAL ===")
    times = []
    for i in range(num_tasks):
        times.append(local_matrix_multiply(i, size, iterations))
    avg = sum(times) / len(times)
    log(f"Tiempo promedio (local): {avg:.4f}s")
    return times, avg

def run_ray_tasks(num_tasks=2, size=14000, iterations=30):
    log("=== EJECUCIÓN CON RAY ===")
    ray.init(address=RAY_ADDRESS)  # Indicar head del cluster de ray
    futures = [ray_matrix_multiply.remote(i, size, iterations) for i in range(num_tasks)]
    results = ray.get(futures)
    avg = sum(results) / len(results)
    log(f"Tiempo promedio (ray): {avg:.4f}s")
    ray.shutdown()
    return results, avg

def print_comparison_table(local_times, ray_times):
    print("\n=== Tabla Comparativa de Tiempos (segundos) ===")
    print(f"{'Tarea':<6} | {'Local':>10} | {'Ray':>10}")
    print("-" * 32)
    for i, (lt, rt) in enumerate(zip(local_times, ray_times)):
        print(f"{i:<6} | {lt:10.4f} | {rt:10.4f}")
    print("-" * 32)
    print(f"{'Promedio':<6} | {sum(local_times)/len(local_times):10.4f} | {sum(ray_times)/len(ray_times):10.4f}\n")

# ---------- Main ----------
if __name__ == "__main__":
    num_tasks = NUM_TASKS
    matrix_size = MATRIX_SIZE
    iterations = ITERATIONS

    local_times, local_avg = run_local_tasks(num_tasks, matrix_size, iterations)
    ray_times, ray_avg = run_ray_tasks(num_tasks, matrix_size, iterations)

    print_comparison_table(local_times, ray_times)
