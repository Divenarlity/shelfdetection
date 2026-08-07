import cv2
import numpy as np

from src.multi_view_geometry import triangulate_mask_pair
from src.pose_geometry import project_world_points


def meta(x):
    return {"resolution":[640,480],"camera_position_world":[x,0,0],
            "camera_rotation_xyzw":[0,0,0,1],
            "intrinsics":{"fx":500,"fy":500,"cx":320,"cy":240}}


def render_quad(points,metadata):
    pixels,valid=project_world_points(points,metadata);assert valid.all()
    mask=np.zeros((480,640),np.uint8)
    cv2.fillPoly(mask,[np.rint(pixels).astype(np.int32)],1)
    return mask.astype(bool)


def test_lidar_free_two_view_triangulation_recovers_shelf_plane():
    corners=[[-1,-1,5],[-1,1,5],[1,1,5],[1,-1,5]]
    a,b=meta(-.5),meta(.5)
    result=triangulate_mask_pair(render_quad(corners,a),a,render_quad(corners,b),b,max_reprojection_px=3)
    assert result is not None
    assert np.allclose(result["center_map"],[0,0,5],atol=.08)
    assert result["reprojection_error_px"]<3
