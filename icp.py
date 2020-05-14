import open3d as o3d
import numpy as np
import copy


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def icp(source,target,trans_init=np.eye(4)):
    
    sourcemesh=copy.deepcopy(source)
    targetmesh=copy.deepcopy(target)
    sourceply =  o3d.geometry.PointCloud()
    targetply =  o3d.geometry.PointCloud()
    sourcemesh.compute_vertex_normals()
    targetmesh.compute_vertex_normals()
    sourceply.points = sourcemesh.vertices
    targetply.points = targetmesh.vertices
    sourceply.normals = sourcemesh.vertex_normals
    targetply.normals = targetmesh.vertex_normals
    
    
    threshold = 0.02
    reg_p2p = o3d.registration.registration_icp(
            sourceply, targetply, threshold, trans_init,
            o3d.registration.TransformationEstimationPointToPlane())
    
    return reg_p2p.transformation

    