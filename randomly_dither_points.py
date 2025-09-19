#!/usr/bin/env python3
import sys
import site

# Add user site-packages for scipy
site.USER_SITE = '/Users/yitong/.local/lib/python3.11/site-packages'
sys.path.append(site.USER_SITE)
import bpy
import bmesh
import sys
import site
import os
import json
import argparse
import numpy as np
import random
from mathutils import Vector
from scipy.interpolate import RBFInterpolator
from scipy.spatial.distance import pdist
import struct
from mathutils.bvhtree import BVHTree
from mathutils import Vector

def get_vertex_neighbors(obj, starting_indices, expansion_depth=5):
    """
    Expands a set of vertex indices to include their neighbors up to a given depth.
    
    Args:
        obj (bpy.types.Object): The object to get the mesh from.
        starting_indices (set): A set of initial vertex indices.
        expansion_depth (int): The number of neighbor rings to expand.
    
    Returns:
        set: The final set of unique vertex indices including all neighbors.
    """
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.verts.ensure_lookup_table()
    
    all_indices = set(starting_indices)
    
    # Use a queue for a breadth-first search
    queue = list(starting_indices)
    visited = set(starting_indices)
    
    for _ in range(expansion_depth):
        next_level_queue = []
        for vert_idx in queue:
            vert = bm.verts[vert_idx]
            # Iterate through the neighbors connected by edges
            for edge in vert.link_edges:
                neighbor_vert = edge.other_vert(vert)
                if neighbor_vert.index not in visited:
                    visited.add(neighbor_vert.index)
                    all_indices.add(neighbor_vert.index)
                    next_level_queue.append(neighbor_vert.index)
        queue = next_level_queue
        if not queue: # Stop if we've reached the end of the mesh
            break
            
    bm.free()
    return all_indices

from mathutils import kdtree


def find_nearest_indices(obj, positions):
    size = len(obj.data.vertices)
    kd = kdtree.KDTree(size)
    
    # Build KDTree with vertex world positions
    for i, v in enumerate(obj.data.vertices):
        kd.insert(obj.matrix_world @ v.co, i)
    kd.balance()
    
    nearest_indices = set()
    for pos in positions:
        _, index, _ = kd.find(pos)  # returns (co, index, dist)
        nearest_indices.add(index)
    
    return nearest_indices




def get_vertex_indices_with_bvh(obj, positions, tolerance=0.001):
    """
    Finds vertex indices corresponding to world positions using a BVH tree for fast lookup.
    
    Args:
        obj (bpy.types.Object): The Blender object.
        positions (list): A list of mathutils.Vector or similar objects.
        tolerance (float): The maximum distance to consider a match.
    
    Returns:
        set: A set of vertex indices.
    """
    # Create a BVH tree from the object's geometry
    verts = [v.co for v in obj.data.vertices]
    bvh = BVHTree.FromPolygons(verts, [(i,) for i in range(len(verts))])
    
    found_indices = set()
    for pos in positions:
        # Convert the world position to local coordinates for the BVH tree lookup
        local_pos = obj.matrix_world.inverted() @ pos
        
        # Find the nearest point in the BVH tree
        location, normal, index, distance = bvh.find_nearest(local_pos)
        
        if distance <= tolerance:
            # The 'index' returned by find_nearest corresponds to the original vertex index
            found_indices.add(index)
            
    return found_indices


# === ARGUMENT PARSING ===
def get_args():
    # Define a custom argument parser that ignores blender's args
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]  # Get all args after "--"
    else:
        argv = []  # No custom args found
    
    parser = argparse.ArgumentParser(description='Headless Blender RBF Deformation Script')
    
    # Required arguments
    parser.add_argument('--input', type=str, required=True, help='Path to input .blend file')
    
    # Configure dithering parameters via JSON
    parser.add_argument('--num_points_to_dither', type=int, default=20)
    parser.add_argument('--min_dither', type=float, default=0.5)
    parser.add_argument('--max_dither', type=float, default=5)
    parser.add_argument('--exempt_vertices', type=str, default=None)
    parser.add_argument('--output', type=str, default='temp_genes.json')
    parser.add_argument('--exempt_keywords', type=str, default='sparse_peripherals', 
                        help='Comma-separated list of keywords for vertex groups to exempt from dithering')
    
    args = parser.parse_args(argv)
    return args

# === HELPER FUNCTIONS ===
def get_vertex_world_position(obj, index):
    return obj.matrix_world @ obj.data.vertices[index].co

def random_offset(min_magnitude, max_magnitude):
    # Generate a normalized random direction vector
    direction = Vector([random.uniform(-1, 1) for _ in range(3)]).normalized()
    # Choose a random magnitude within the specified range
    magnitude = random.uniform(min_magnitude, max_magnitude)
    return direction * magnitude

def main():
    args = get_args()
    EXEMPT_KEYWORDS = [keyword.strip() for keyword in args.exempt_keywords.split(',')]
    # Load the .blend file
    bpy.ops.wm.open_mainfile(filepath=args.input)
    bpy.ops.object.mode_set(mode='OBJECT')
    obj = bpy.data.objects["eyes_open_mask"]

    # Apply transforms
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    mesh = obj.data
    verts = mesh.vertices
    exempt_vertex_indices = set()

    for vgroup in obj.vertex_groups:
        name = vgroup.name
        # Check if this group should be exempt from dithering
        is_exempt = any(key.lower() in name.lower() for key in EXEMPT_KEYWORDS)
        
        if is_exempt:
            verts_in_group = [v.index for v in verts if any(g.group == vgroup.index for g in v.groups)]
            exempt_vertex_indices.update(verts_in_group)
            print(f"Exempt group '{name}' has {len(verts_in_group)} vertices")
    
    if args.exempt_vertices is not None and os.path.exists(args.exempt_vertices):
        with open(args.exempt_vertices, 'r') as f:
            output_data = json.load(f)
        exempt_vertex_positions = [Vector(pos) for pos in output_data.get('exempt_vertices', [])]
        
        # 1. Find the nearest vertex for each position using the BVH tree
        nearest_indices = find_nearest_indices(obj, exempt_vertex_positions)
        
        # 2. Expand from these nearest vertices to include their neighbors
        newly_exempt_indices = get_vertex_neighbors(obj, nearest_indices)
        
        # Add these indices to the main exempt set
        exempt_vertex_indices.update(newly_exempt_indices)
        
        print(f"Added {len(newly_exempt_indices)} vertices to the exempt set from the previous dithered points.")


    
    all_vertex_indices = set(range(len(verts)))
    non_exempt_indices = list(all_vertex_indices - exempt_vertex_indices)

    print(f"Available non-exempt vertices: {len(non_exempt_indices)}")

    # Randomly sample points to dither
    num_to_sample = min(args.num_points_to_dither, len(non_exempt_indices))
    sampled_indices = random.sample(non_exempt_indices, num_to_sample)

    genes = []

    # Add randomly sampled vertices with dithering
    for vert_idx in sampled_indices:
        pos = get_vertex_world_position(obj, vert_idx)
        offset = random_offset(args.min_dither, args.max_dither)
        dithered_pos = pos + offset
        
        # genes.append((np.array(pos), np.array(dithered_pos)))
        genes.append((np.array(pos).tolist(), np.array(dithered_pos).tolist()))

        
    with open(args.output, 'w') as f:
        json.dump({"genes": genes}, f)



if __name__ == "__main__":
    # Add user site-packages for scipy
    if "site.USER_SITE" in os.environ:
        site.USER_SITE = os.environ["site.USER_SITE"]
    else:
        site.USER_SITE = '/Users/yitong/.local/lib/python3.11/site-packages'
    sys.path.append(site.USER_SITE)
    
    main()