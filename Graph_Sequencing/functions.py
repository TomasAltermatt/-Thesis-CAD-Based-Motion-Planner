import networkx as nx
from pathlib import Path
import IM_Generation.functions as imf
import time

def build_geometric_and_or_graph(matrices, assembly_manifest):
    """
    Takes the 6 directional Interference Matrices and builds a Directed 
    AND/OR Multigraph of all valid geometric disassembly sequences.
    Generates parallel arcs for different holding arms and directions.
    """
    # 1. Guarantee part names align exactly with matrix indices
    sorted_parts = sorted(assembly_manifest.items(), key=lambda x: x[1]["matrix_idx"])
    part_names = [name for name, data in sorted_parts]
    N = len(part_names)
    
    # 2. Initialize the Directed Multigraph (Allows parallel edges)
    G = nx.MultiDiGraph()
    
    # 3. The Root Node: All parts present (tuple of 1s)
    root_state = tuple([1] * N)
    G.add_node(root_state)
    
    # Queue for Breadth-First Search
    queue = [root_state]
    visited = set([root_state])
    
    # Explicitly define the available robotic arms for the parallel edges
    arms = ["Left", "Right"]
    
    print(f"\n[STARTING] Building AND/OR Multigraph for {N} parts...")
    
    while queue:
        current_state = queue.pop(0)
        
        # If the state is (0, 0, 0...), the assembly is fully disassembled. Stop here.
        if sum(current_state) == 0:
            continue
            
        # Get the indices of the parts that are still physically present in this state
        active_indices = [i for i, val in enumerate(current_state) if val == 1]
        
        # Test each active part to see if it can be extracted
        for i in active_indices:
            
            # Check all 6 extraction directions
            for direction, matrix in matrices.items():
                
                # Assume it's free until we find a collision
                collision_free = True
                
                # Check against all OTHER parts currently in the sub-assembly
                for j in active_indices:
                    if i != j:
                        # If there is a '2', part 'j' blocks part 'i' in this direction
                        if matrix[i, j] == 2:
                            collision_free = False
                            break 
                            
                # If no collisions were found, this is a valid extraction path!
                if collision_free:
                    # Create the new child state by removing part 'i'
                    new_state_list = list(current_state)
                    new_state_list[i] = 0
                    new_state = tuple(new_state_list)
                    
                    # If we haven't seen this specific sub-assembly state before, add it to the graph
                    if new_state not in visited:
                        visited.add(new_state)
                        queue.append(new_state)
                        G.add_node(new_state)
                        
                    # MULTIGRAPH LOGIC: Generate parallel arcs for each holding arm
                    for holding_arm in arms:
                        G.add_edge(
                            current_state, 
                            new_state, 
                            removed_part=part_names[i], 
                            removed_idx=i,
                            direction=direction,
                            holding_arm=holding_arm
                        )
                        
    print(f"--- Graph Complete! Generated {G.number_of_nodes()} Nodes and {G.number_of_edges()} Edges ---")
    return G

def print_geometric_assembly_sequences(G, N):
    """
    Prints all unique geometric assembly sequences from a MultiDiGraph,
    ignoring the parallel robotic arcs (direction/arms).
    """
    start_node = tuple([1] * N)  # Fully assembled
    end_node = tuple([0] * N)    # Fully disassembled
    
    # THE FIX: Cast to a standard DiGraph purely for the pathfinding.
    # This instantly collapses all parallel robotic edges into a single geometric edge,
    # preventing the combinatorial explosion of duplicate paths.
    G_simple = nx.DiGraph(G)
    
    all_paths = list(nx.all_simple_paths(G_simple, source=start_node, target=end_node))
    
    print(f"\nFound {len(all_paths)} unique geometric assembly sequences.")
    
    for path_idx, path in enumerate(all_paths):
        disassembly_sequence = []
        
        # Walk through the nodes in this specific sequence
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i+1]
            
            # We still pull the part name from the original Multigraph 'G'
            # because G_simple might strip out the edge dictionary attributes
            first_edge_key = list(G[u][v].keys())[0]
            removed_part = G[u][v][first_edge_key]['removed_part']
            
            disassembly_sequence.append(removed_part)
            
        # Reverse the disassembly list to get the Assembly sequence
        assembly_sequence = list(reversed(disassembly_sequence))
        
        # Print the clean sequence
        print(f"Sequence {path_idx + 1}: {' -> '.join(assembly_sequence)}")