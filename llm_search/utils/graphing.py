#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: graphing.py
Author: Mahd Afzal
Date: 2025-08-19
Version: 1.0
Description: 
    This module provides utilities for graph-based operations, including A* pathfinding and graph construction.
    For graph components that end up disconnected, it provides methods for identifying and bridging these gaps.
"""

import numpy as np
from queue import PriorityQueue
from shapely.geometry import LineString

class AStarNode:
    def __init__(self, id, image_point, world_point):
        self.id = id
        self.image_point = image_point
        self.world_point = world_point
        self.edges = set()

    def add_edge(self, other_node):
        self.edges.add(other_node)
    
    def __repr__(self):
        return f"Node(id={self.id}, image_point={self.image_point}, world_point={self.world_point}, edges={[n.id for n in self.edges]})"

def connected_components(adj):
    visited, comps = set(), []
    for nid in adj.keys():
        if nid in visited:
            continue
        stack = [nid]
        comp = []
        visited.add(nid)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nb in adj[cur].edges:
                if nb.id not in visited:
                    visited.add(nb.id)
                    stack.append(nb.id)
        comps.append(sorted(comp))
    return comps

def bridge_components(adjacency_list, centroids_world_xy, polygon, margin=1e-3):
    comps = connected_components(adjacency_list)
    print(f"Components before bridging: {len(comps)} -> sizes {[len(c) for c in comps]}")
    if len(comps) <= 1:
        return

    # Erode polygon slightly to keep bridges strictly inside
    safe_poly = polygon.buffer(-margin)

    while len(comps) > 1:
        best = None  # (dist2, a_id, b_id)
        A, B = comps[0], comps[1]

        # Search for the shortest valid in-polygon bridge between first two components
        for a_id in A:
            pa = centroids_world_xy[a_id]
            for b_id in B:
                pb = centroids_world_xy[b_id]
                seg = LineString([tuple(pa), tuple(pb)])
                if safe_poly.covers(seg):  # stays inside polygon
                    d2 = np.sum((pa - pb) ** 2)
                    if best is None or d2 < best[0]:
                        best = (d2, a_id, b_id)

        if best is None:
            # Relax: allow touching boundary if strictly inside failed
            for a_id in A:
                pa = centroids_world_xy[a_id]
                for b_id in B:
                    pb = centroids_world_xy[b_id]
                    seg = LineString([tuple(pa), tuple(pb)])
                    if polygon.covers(seg):
                        d2 = np.sum((pa - pb) ** 2)
                        if best is None or d2 < best[0]:
                            best = (d2, a_id, b_id)

        if best is None:
            print("No valid in-polygon bridge found; consider lowering pruning or increasing max_area.")
            break

        _, a_id, b_id = best
        # Add undirected graph edge
        adjacency_list[a_id].add_edge(adjacency_list[b_id])
        adjacency_list[b_id].add_edge(adjacency_list[a_id])

        # Recompute components
        comps = connected_components(adjacency_list)

def build_adjacency_list(polygon, tris, ids, centroids_img, centroids_world_xy):

    # World-space centroids (from earlier computation)
    centroids_world = centroids_world_xy  # shape (Nk, 2)

    # Create nodes keyed by triangle ID
    adjacency_list = {
        int(ids[i]): AStarNode(int(ids[i]), image_point=centroids_img[i], world_point=centroids_world[i])
        for i in range(len(tris))
    }

    # Build adjacency via shared undirected edges
    for i, (i0, i1, i2) in enumerate(tris):
        id_i = int(ids[i])
        for j, (j0, j1, j2) in enumerate(tris):
            if i == j:
                continue
            id_j = int(ids[j])
            # Check shared edges
            shared_edges = set([i0, i1, i2]) & set([j0, j1, j2])
            if len(shared_edges) == 2:  # exactly one edge shared
                adjacency_list[id_i].add_edge(adjacency_list[id_j])
                adjacency_list[id_j].add_edge(adjacency_list[id_i])
    
    # bridge components if needed
    if len(adjacency_list) > 1:
        bridge_components(adjacency_list, centroids_world_xy, polygon)
    
    return adjacency_list

def a_star_planner(start, goal, adjacency_list):
    """
    A* path planner using an adjacency list representation of the graph.
    
    :param start: Starting node ID
    :param goal: Goal node ID
    :param adjacency_list: Dictionary where keys are node IDs and values are AStarNode objects
    :return: List of node IDs representing the path from start to goal, or None if no path exists
    """

    open_set = PriorityQueue()
    open_set.put((0, start))
    came_from = {}
    g_score = {node: float('inf') for node in adjacency_list}
    g_score[start] = 0
    f_score = {node: float('inf') for node in adjacency_list}
    f_score[start] = heuristic(start, goal, adjacency_list)

    while not open_set.empty():
        current = open_set.get()[1]

        if current == goal:
            return reconstruct_path(came_from, current)

        for neighbor_node in adjacency_list[current].edges:
            neighbor = neighbor_node.id
            tentative_g_score = g_score[current] + 1  # Assuming uniform cost for edges

            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal, adjacency_list)
                open_set.put((f_score[neighbor], neighbor))

    return None  # No path found

def heuristic(node_a_id, node_b_id, adjacency_list):
    """
    Heuristic function for A* algorithm using Euclidean distance.
    
    :param node_a_id: First node ID
    :param node_b_id: Second node ID
    :param adjacency_list: Dictionary mapping node IDs to AStarNode objects
    :return: Estimated cost from node_a to node_b in squared euclidean distance
    """
    node_a = adjacency_list[node_a_id]
    node_b = adjacency_list[node_b_id]
    node_a_x, node_a_y = node_a.world_point
    node_b_x, node_b_y = node_b.world_point
    dist = (node_a_x - node_b_x) ** 2 + (node_a_y - node_b_y) ** 2
    return dist

def reconstruct_path(came_from, current):
    """
    Reconstructs the path from start to goal by backtracking through the came_from map.
    
    :param came_from: Dictionary mapping each node to its predecessor
    :param current: Current node ID (goal)
    :return: List of node IDs representing the path from start to goal
    """
    total_path = [current]
    while current in came_from:
        current = came_from[current]
        total_path.append(current)
    total_path.reverse()
    return total_path
