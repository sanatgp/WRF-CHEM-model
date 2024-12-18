import subprocess
import re
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class NUMANode:
    id: int
    cpus: List[int]
    memory: int  # in MB
    distances: Dict[int, int]  # distances to other NUMA nodes

class HardwareTopology:
    def __init__(self):
        self.numa_nodes: Dict[int, NUMANode] = {}
        self.total_cpus = 0
        self.parse_topology()

    def parse_topology(self):
        """Parse NUMA topology using numactl --hardware"""
        try:
            numa_output = subprocess.check_output(['numactl', '--hardware'], 
                                                universal_newlines=True)
            
            # Parse NUMA nodes
            node_pattern = r'node (\d+) cpus: ([\d ]+)'
            node_matches = re.finditer(node_pattern, numa_output)
            
            for match in node_matches:
                node_id = int(match.group(1))
                cpus = [int(cpu) for cpu in match.group(2).split()]
                self.numa_nodes[node_id] = NUMANode(
                    id=node_id,
                    cpus=cpus,
                    memory=0,  # Will be filled later
                    distances={}
                )
                self.total_cpus += len(cpus)

            # Parse distances
            distances_pattern = r'node distances:\n([\s\S]+?)(?:\n\n|\Z)'
            distances_match = re.search(distances_pattern, numa_output)
            
            if distances_match:
                distances_text = distances_match.group(1)
                lines = distances_text.strip().split('\n')[1:]  # Skip header
                
                for i, line in enumerate(lines):
                    distances = [int(x) for x in line.split()[1:]]
                    self.numa_nodes[i].distances = {
                        j: dist for j, dist in enumerate(distances)
                    }

        except subprocess.CalledProcessError as e:
            print(f"Error running numactl: {e}")
            raise

    def get_numa_distance(self, cpu1: int, cpu2: int) -> int:
        """Get NUMA distance between two CPUs"""
        node1 = self._get_numa_node_for_cpu(cpu1)
        node2 = self._get_numa_node_for_cpu(cpu2)
        return node1.distances[node2.id]

    def _get_numa_node_for_cpu(self, cpu: int) -> NUMANode:
        """Find which NUMA node contains a given CPU"""
        for node in self.numa_nodes.values():
            if cpu in node.cpus:
                return node
        raise ValueError(f"CPU {cpu} not found in any NUMA node")

class MPIGridConfig:
    def __init__(self, x_dim: int, y_dim: int, total_ranks: int):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.total_ranks = total_ranks
        self.rank_to_cpu: Dict[int, int] = {}

    def assign_ranks_to_cpus(self, topology: HardwareTopology):
        """Simple round-robin assignment of ranks to CPUs"""
        available_cpus = []
        for node in topology.numa_nodes.values():
            available_cpus.extend(node.cpus)
        
        for rank in range(self.total_ranks):
            self.rank_to_cpu[rank] = available_cpus[rank % len(available_cpus)]

    def get_neighbors(self, rank: int) -> List[int]:
        """Get neighboring ranks for a given rank in the grid"""
        x = rank % self.x_dim
        y = rank // self.x_dim
        neighbors = []

        # Check all 4 directions (can be extended to more)
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < self.x_dim and 0 <= new_y < self.y_dim:
                neighbor_rank = new_y * self.x_dim + new_x
                neighbors.append(neighbor_rank)

        return neighbors

class CommunicationPattern:
    def __init__(self, config: MPIGridConfig):
        self.config = config
        self.messages: List[Tuple[int, int, int]] = []  # (src, dst, size)

    def generate_halo_exchange_pattern(self, message_size: int):
        """Generate typical halo exchange communication pattern"""
        for rank in range(self.config.total_ranks):
            neighbors = self.config.get_neighbors(rank)
            for neighbor in neighbors:
                self.messages.append((rank, neighbor, message_size))

def main():
    # Initialize topology
    print("Parsing hardware topology...")
    topology = HardwareTopology()
    print(f"Found {len(topology.numa_nodes)} NUMA nodes")
    for node_id, node in topology.numa_nodes.items():
        print(f"Node {node_id}: CPUs {node.cpus}")
        print(f"Distances: {node.distances}")

    # Test with a sample configuration
    print("\nTesting 6x8 configuration...")
    config = MPIGridConfig(6, 8, 48)
    config.assign_ranks_to_cpus(topology)

    pattern = CommunicationPattern(config)
    pattern.generate_halo_exchange_pattern(1024 * 1024)  # 1MB messages

    # Analyze NUMA distances for communications
    total_cost = 0
    for src, dst, size in pattern.messages:
        src_cpu = config.rank_to_cpu[src]
        dst_cpu = config.rank_to_cpu[dst]
        distance = topology.get_numa_distance(src_cpu, dst_cpu)
        total_cost += distance * size

    print(f"\nTotal communication cost (arbitrary units): {total_cost}")
    print("This cost can be used to compare different process placements")

if __name__ == "__main__":
    main()
