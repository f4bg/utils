import ray
import datetime # Import the datetime library

# --- Ray Connection Configuration ---
RAY_ADDRESS = "ray://192.168.12.91:31001"
# ---------------------------------------

# ANSI escape codes for colors and styles
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m' # Resets color and style
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

try:
    # Attempt to connect to the Ray cluster
    # 'ignore_reinit_error=True' allows calling ray.init() multiple times if necessary
    ray.init(address=RAY_ADDRESS, ignore_reinit_error=True)

    # --- Main report title ---
    print(f"\n{Colors.HEADER}{Colors.BOLD}=== RAY NODE RESOURCES REPORT ==={Colors.ENDC}")
    print(f"{Colors.UNDERLINE}Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}\n")

    # --- Global Cluster Resources Summary ---
    print(f"{Colors.BOLD}{Colors.BLUE}--- GLOBAL CLUSTER RESOURCES SUMMARY ---{Colors.ENDC}")
    cluster_resources = ray.cluster_resources()
    if cluster_resources:
        for resource, quantity in cluster_resources.items():
            # Format memory for better readability in the global summary
            if 'memory' in resource.lower() and isinstance(quantity, (int, float)):
                if quantity >= 10**9:
                    print(f"  - {Colors.GREEN}{resource}:{Colors.ENDC} {quantity / 10**9:.2f} GB")
                elif quantity >= 10**6:
                    print(f"  - {Colors.GREEN}{resource}:{Colors.ENDC} {quantity / 10**6:.2f} MB")
                else:
                    print(f"  - {Colors.GREEN}{resource}:{Colors.ENDC} {quantity} bytes")
            elif 'cpu' in resource.lower() or 'gpu' in resource.lower():
                print(f"  - {Colors.GREEN}{resource}:{Colors.ENDC} {int(quantity) if quantity == int(quantity) else quantity}")
            else:
                print(f"  - {Colors.GREEN}{resource}:{Colors.ENDC} {quantity}")
    else:
        print(f"  {Colors.WARNING}No global resources found.{Colors.ENDC}")

    # --- Separator after global summary ---
    print(f"\n{Colors.CYAN}{'='*50}{Colors.ENDC}\n")

    # Get detailed information for each individual node
    cluster_nodes = ray.nodes()

    print(f"{Colors.BOLD}{Colors.BLUE}--- NODE DETAILS ---{Colors.ENDC}\n")

    for i, node in enumerate(cluster_nodes):
        # Print a separator before each node, except the first one
        if i > 0:
            print(f"\n{Colors.CYAN}{'='*50}{Colors.ENDC}\n") # More distinctive node separator

        node_id_short = node['NodeID'][:8] # Shorten NodeID for better readability
        node_name = node.get('NodeManagerHostname', node.get('NodeName', 'N/A'))
        node_ip = node.get('NodeManagerAddress', 'N/A')
        # Check if the node is alive and apply color
        is_alive = f"{Colors.GREEN}Yes{Colors.ENDC}" if node.get('Alive', False) else f"{Colors.FAIL}No{Colors.ENDC}"

        # --- Title for each node ---
        print(f"{Colors.BOLD}{Colors.BLUE}--- NODE {i+1}: {node_name} ({node_ip}) ---{Colors.ENDC}")
        print(f"  Node ID: {node_id_short}...")
        print(f"  Status: {is_alive}")

        resources = node.get('Resources', {})
        print(f"{Colors.UNDERLINE}  Available Resources:{Colors.ENDC}")
        if resources:
            for resource_name, value in resources.items():
                # Format memory for better readability
                if 'memory' in resource_name.lower() and isinstance(value, (int, float)):
                    if value >= 10**9:
                        print(f"    - {Colors.GREEN}{resource_name}:{Colors.ENDC} {value / 10**9:.2f} GB")
                    elif value >= 10**6:
                        print(f"    - {Colors.GREEN}{resource_name}:{Colors.ENDC} {value / 10**6:.2f} MB")
                    else:
                        print(f"    - {Colors.GREEN}{resource_name}:{Colors.ENDC} {value} bytes")
                elif 'cpu' in resource_name.lower() or 'gpu' in resource_name.lower():
                     print(f"    - {Colors.GREEN}{resource_name}:{Colors.ENDC} {int(value) if value == int(value) else value}")
                else:
                    print(f"    - {Colors.GREEN}{resource_name}:{Colors.ENDC} {value}")
        else:
            print(f"    {Colors.WARNING}No specific resources listed for this node.{Colors.ENDC}")

        labels = node.get('Labels', {})
        if labels:
            print(f"{Colors.UNDERLINE}  Labels:{Colors.ENDC}")
            for label_name, label_value in labels.items():
                print(f"    - {Colors.CYAN}{label_name}:{Colors.ENDC} {label_value}")

except ray.exceptions.RaySystemError as e:
    print(f"\n{Colors.FAIL}Error connecting to Ray or getting node information. Ensure that the address {RAY_ADDRESS} is correct and that Ray is running. If using KubeRay, you might need a port-forward.{Colors.ENDC}")
    print(f"{Colors.FAIL}Details: {e}{Colors.ENDC}")
except Exception as e:
    print(f"\n{Colors.FAIL}An unexpected error occurred: {e}{Colors.ENDC}")
finally:
    if ray.is_initialized():
        ray.shutdown()
        print(f"\n{Colors.WARNING}--- Ray connection closed ---{Colors.ENDC}")