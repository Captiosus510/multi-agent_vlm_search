#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: vector.py
Author: Mahd Afzal
Date: 2025-08-19
Version: 1.0
Description: 
    This module provides a class for representing 7D vectors.
    In this codebase it is used to represent positions and orientations in 3D space.

    TODO: Extend this to support 7D vector operations.
"""

class Vector7D:
    def __init__(self, x: float, y: float, z: float, x_axis: float, y_axis: float, z_axis: float, rotation: float):
        self.x = x
        self.y = y
        self.z = z
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.z_axis = z_axis
        self.rotation = rotation