#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: mapf.py
Author: Mahd Afzal
Date: 2025-08-19
Version: 1.0
Description: 
    This module provides utilities for multi-agent pathfinding (MAPF) in a shared environment.
    Currently utilises a centralized conflict-based search (CBS) approach.
    This involves a high level Constraint Tree with a low level modified A* algorithm to consider constraints.
"""

import heapq
from dataclasses import dataclass, field

@dataclass(frozen=True)
class Constraint:
    agent: str
    ctype: str          # 'vertex' or 'edge'
    v: int = 0       # vertex (for vertex constraint)
    u: int = 0       # edge start
    w: int = 0       # edge end
    t: int = 0       # timestep

@dataclass
class CTNode:
    constraints: list = field(default_factory=list)   # list[Constraint]
    paths: dict = field(default_factory=dict)         # {agent: [triangles per timestep]}
    cost: int = 0                                     # sum of path lengths (time steps - 1)
    conflicts: int = 0

class CBSPlanner:
    def __init__(self, adjacency_list, goal_wait=5, weight=1.0, max_time=300, logger=None):
        self.graph = adjacency_list                  # {triangle_id: AStarNode with .edges}
        self.goal_wait = goal_wait
        self.weight = max(1.0, weight)               # ECBS factor
        self.max_time = max_time
        self.logger = logger

    # ---------- Public ----------
    def solve(self, agents, starts, goals):
        root = CTNode()
        for a in agents:
            path = self._low_level(a, starts[a], goals[a], [])
            if path is None:
                return False, {}
            root.paths[a] = path
        root.cost = self._compute_cost(root.paths)
        root.conflicts = self._count_conflicts(root.paths)

        open_heap = []
        heapq.heappush(open_heap, (root.cost, root.conflicts, id(root), root))

        while open_heap:
            _, _, _, node = heapq.heappop(open_heap)
            conflict = self._first_conflict(node.paths)
            if conflict is None:
                return True, node.paths  # success
            ctype, a1, a2, info, t = conflict

            # Branch: produce two children with added constraints
            for agent in (a1, a2):
                child = CTNode(constraints=list(node.constraints), paths=dict(node.paths))
                if ctype == 'vertex':
                    child.constraints.append(Constraint(agent=agent, ctype='vertex', v=info, t=t)) # type: ignore
                else:
                    u, v = info
                    child.constraints.append(Constraint(agent=agent, ctype='edge', u=u, w=v, t=t))
                # Replan only that agent
                original_goal = goals[agent]  # Store original goals
                new_path = self._low_level(agent,
                                           child.paths[agent][0],
                                           original_goal,
                                           child.constraints,
                                           goal_override=child.paths[agent][-1])  # we stored goal at end
                if new_path is None:
                    continue
                child.paths[agent] = new_path
                child.cost = self._compute_cost(child.paths)
                child.conflicts = self._count_conflicts(child.paths)
                heapq.heappush(open_heap, (child.cost, child.conflicts, id(child), child))
        return False, {}

    # ---------- Helpers ----------
    def _compute_cost(self, paths):
        return sum(len(p) - 1 for p in paths.values())

    def _count_conflicts(self, paths):
        cnt = 0
        horizon = max(len(p) for p in paths.values())
        for t in range(horizon):
            occ = {}
            # vertex
            for a, p in paths.items():
                v = p[min(t, len(p)-1)]
                if v in occ and occ[v] != a:
                    cnt += 1
                else:
                    occ[v] = a
            # edge
            for a, p in paths.items():
                v_t = p[min(t, len(p)-1)]
                v_t1 = p[min(t+1, len(p)-1)]
                for b, q in paths.items():
                    if b <= a:
                        continue
                    u_t = q[min(t, len(q)-1)]
                    u_t1 = q[min(t+1, len(q)-1)]
                    if v_t == u_t1 and v_t1 == u_t and v_t != v_t1:
                        cnt += 1
        return cnt

    def _first_conflict(self, paths):
        horizon = max(len(p) for p in paths.values())
        for t in range(horizon):
            # vertex
            occ = {}
            for a, p in paths.items():
                v = p[min(t, len(p)-1)]
                if v in occ:
                    return ('vertex', a, occ[v], v, t)
                occ[v] = a
            # edge
            agents = list(paths.keys())
            for i in range(len(agents)):
                for j in range(i+1, len(agents)):
                    a = agents[i]; b = agents[j]
                    p = paths[a]; q = paths[b]
                    v_t = p[min(t, len(p)-1)]
                    v_t1 = p[min(t+1, len(p)-1)]
                    u_t = q[min(t, len(q)-1)]
                    u_t1 = q[min(t+1, len(q)-1)]
                    if v_t == u_t1 and v_t1 == u_t and v_t != v_t1:
                        return ('edge', a, b, (v_t, v_t1), t)
        return None

    def _low_level(self, agent, start, goal, constraints, goal_override=None):
        """
        Time-expanded A* with constraints. Returns list of triangle IDs per timestep (includes waits).
        goal_override lets us reuse known goal if needed.
        """
        goal_node = goal_override if goal_override is not None else goal
        # Build fast constraint lookups
        forbid_vertex = {}
        forbid_edge = {}
        for c in constraints:
            if c.agent != agent:
                continue
            if c.ctype == 'vertex':
                forbid_vertex[(c.v, c.t)] = True
            else:
                forbid_edge[((c.u, c.w), c.t)] = True

        # Precompute simple admissible heuristic: zero or 1-step BFS distances
        dist_cache = self._single_source_shortest(goal_node)

        def h(n):
            return dist_cache.get(n, 0)

        start_state = (start, 0)
        g_cost = {start_state: 0}
        parent = {}

        open_heap = []
        f0 = h(start)
        heapq.heappush(open_heap, (f0, 0, start_state))

        best_goal_state = None
        best_goal_time = None

        max_allowed_time = self.max_time

        while open_heap:
            f, g, (v, t) = heapq.heappop(open_heap)
            # Goal handling: once at goal, allow waiting goal_wait steps; return first found (for w=1 optimal)
            if v == goal_node:
                # Extend with waits to allow others to clear conflicts
                final_len = t + self.goal_wait
                # Reconstruct path with appended waits
                path = self._reconstruct(parent, (v, t))
                last = path[-1]
                while len(path) < final_len + 1:
                    path.append(last)
                return path

            if t > max_allowed_time:
                continue

            # Expand neighbors + wait
            if v in self.graph:
                neighbors = [nb.id for nb in self.graph[v].edges]
            else:
                neighbors = []
            for nv in neighbors + [v]:  # include wait
                nt = t + 1
                # Constraints
                if forbid_vertex.get((nv, nt)):
                    continue
                if nv != v:
                    if forbid_edge.get(((v, nv), t)):
                        continue
                state = (nv, nt)
                tentative = g + 1
                if tentative < g_cost.get(state, 1e9):
                    g_cost[state] = tentative
                    parent[state] = (v, t)
                    fscore = tentative + self.weight * h(nv)
                    heapq.heappush(open_heap, (fscore, tentative, state))
        return None

    def _reconstruct(self, parent, state):
        seq = [state[0]]
        cur = state
        while cur in parent:
            cur = parent[cur]
            seq.append(cur[0])
        return list(reversed(seq))

    def _single_source_shortest(self, goal):
        # Reverse BFS from goal for unweighted graph distances
        dist = {goal: 0}
        dq = [goal]
        i = 0
        while i < len(dq):
            v = dq[i]; i += 1
            # Handle AStarNode structure: get neighbors from .edges attribute
            if v in self.graph:
                neighbors = [nb.id for nb in self.graph[v].edges]
                for nb in neighbors:
                    if nb not in dist:
                        dist[nb] = dist[v] + 1
                        dq.append(nb)
        return dist
