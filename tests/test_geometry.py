import unittest
import numpy as np
from src.geometry import order_quad, scale_polygon, box_mask, polygon_mask, overlap_over_detection, classify_section


class GeometryTests(unittest.TestCase):
    def test_order_quad(self):
        result = order_quad([[100,100],[0,100],[100,0],[0,0]])
        np.testing.assert_array_equal(result, [[0,0],[100,0],[100,100],[0,100]])

    def test_scale_polygon(self):
        result = scale_polygon([[0,0],[100,0],[100,50],[0,50]], (100,50), (200,100))
        np.testing.assert_array_equal(result, [[0,0],[200,0],[200,100],[0,100]])

    def test_overlap(self):
        shelf = polygon_mask([[0,0],[50,0],[50,100],[0,100]], (100,100))
        gap = box_mask([25,0,75,100], (100,100))
        self.assertAlmostEqual(overlap_over_detection(gap,shelf), .52, places=2)

    def test_perspective_sections_and_boundaries(self):
        quad=[[10,10],[110,0],[100,100],[0,90]]
        self.assertEqual(classify_section((20,50),quad),"left")
        self.assertEqual(classify_section((55,50),quad),"middle")
        self.assertEqual(classify_section((90,50),quad),"right")
        self.assertEqual(classify_section((36,48),quad,boundary_margin=.04),"unknown_section")
