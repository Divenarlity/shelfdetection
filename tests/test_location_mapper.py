import unittest
import numpy as np
from src.location_mapper import map_gap
from src.store_map import Camera, Shelf


class MapperTests(unittest.TestCase):
    def setUp(self):
        self.cam=Camera("CAM-01","A",200,100,[
            Shelf("A-R01-L01","A-R01",1,[[0,0],[100,0],[100,100],[0,100]]),
            Shelf("A-R02-L01","A-R02",1,[[100,0],[200,0],[200,100],[100,100]])])

    def test_best_overlap(self):
        result=map_gap([120,20,180,80],(100,200,3),self.cam,min_overlap=.2)
        self.assertEqual(result["shelf_id"],"A-R02-L01")

    def test_below_threshold_unknown(self):
        result=map_gap([95,20,105,80],(100,200,3),self.cam,min_overlap=.7)
        self.assertEqual(result["shelf_id"],"unknown_shelf")

    def test_manual_fallback(self):
        result=map_gap([10,20,30,80],(100,200,3),self.cam,[])
        self.assertEqual(result["mapping_source"],"manual_roi_fallback")

    def test_mask_and_manual_id(self):
        mask=np.zeros((100,200),np.uint8); mask[:,0:100]=1
        result=map_gap([10,20,30,80],(100,200,3),self.cam,[{"mask":mask,"confidence":.8}])
        self.assertEqual(result["mapping_source"],"shelf_mask_and_manual_id")
        self.assertEqual(result["shelf_id"],"A-R01-L01")
