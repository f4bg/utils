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

@ray.remote(num_gpus=1)
def ray_matrix_multiply(task_id: int, size: int = 14000, iterations: int = 30):
    return local_matrix_multiply(task_id, size, iterations)

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
    ray.init(address=RAY_ADDRESS)
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

def print_statistics(local_times, ray_times):
    local_avg = sum(local_times) / len(local_times)
    ray_avg = sum(ray_times) / len(ray_times)
    local_total = sum(local_times)
    ray_total = sum(ray_times)

    diff = ray_avg - local_avg
    perc_change = (diff / local_avg) * 100
    speed_ratio = local_avg / ray_avg if ray_avg > 0 else float('inf')

    print("=== Estadísticas Comparativas ===")
    print(f"Tiempo promedio local   : {local_avg:.4f} s")
    print(f"Tiempo promedio Ray     : {ray_avg:.4f} s")
    print(f"Diferencia promedio     : {diff:+.4f} s")
    if perc_change >= 0:
        print(f"Ray es {perc_change:.2f}% más lento que local.")
    else:
        print(f"Ray es {abs(perc_change):.2f}% más rápido que local.")
    print(f"Velocidad relativa      : {speed_ratio:.2f}x (local/Ray)")
    print(f"Tiempo total local      : {local_total:.4f} s")
    print(f"Tiempo total Ray        : {ray_total:.4f} s\n")

if __name__ == "__main__":
    num_tasks = NUM_TASKS
    matrix_size = MATRIX_SIZE
    iterations = ITERATIONS

    local_times, local_avg = run_local_tasks(num_tasks, matrix_size, iterations)
    ray_times, ray_avg = run_ray_tasks(num_tasks, matrix_size, iterations)

    print_comparison_table(local_times, ray_times)
    print_statistics(local_times, ray_times)
