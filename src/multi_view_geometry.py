from __future__ import annotations

import cv2
import numpy as np

from .geometry import order_quad
from .pose_geometry import _quaternion_rotation


def camera_projection(metadata):
    intr=metadata["intrinsics"]; k=np.array([[intr["fx"],0,intr["cx"]],[0,intr["fy"],intr["cy"]],[0,0,1.]])
    world_from_camera=_quaternion_rotation(metadata["camera_rotation_xyzw"])
    optical_from_world=np.diag([1.,-1.,1.])@world_from_camera.T
    center=np.asarray(metadata["camera_position_world"],float)
    return k@np.c_[optical_from_world,-optical_from_world@center]


def mask_quad(mask):
    contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not contours:return None
    points=cv2.boxPoints(cv2.minAreaRect(max(contours,key=cv2.contourArea)))
    return order_quad(points)  # TL,TR,BR,BL


def triangulate_mask_pair(mask_a,meta_a,mask_b,meta_b,max_reprojection_px=12.0):
    qa,qb=mask_quad(mask_a),mask_quad(mask_b)
    if qa is None or qb is None:return None
    pa,pb=camera_projection(meta_a),camera_projection(meta_b)
    homogeneous=cv2.triangulatePoints(pa,pb,qa.T.astype(float),qb.T.astype(float))
    points=(homogeneous[:3]/homogeneous[3]).T
    if not np.isfinite(points).all():return None
    def reprojection(p,world):
        h=(p@np.c_[world,np.ones(len(world))].T).T
        return h[:,:2]/h[:,2:3]
    error=max(np.linalg.norm(reprojection(pa,points)-qa,axis=1).mean(),
              np.linalg.norm(reprojection(pb,points)-qb,axis=1).mean())
    baseline=np.linalg.norm(np.asarray(meta_a["camera_position_world"])-meta_b["camera_position_world"])
    if error>max_reprojection_px or baseline<1e-3:return None
    corners=points[[3,0,1,2]]  # BL,TL,TR,BR
    normal=np.cross(corners[1]-corners[0],corners[3]-corners[0])
    normal/=max(np.linalg.norm(normal),1e-9)
    center=corners.mean(axis=0)
    camera=np.asarray(meta_b["camera_position_world"])
    if np.dot(normal,camera-center)<0:normal=-normal
    return {"center_map":center.tolist(),"corners_map":corners.tolist(),
            "front_normal_map":normal.tolist(),"reprojection_error_px":float(error)}
