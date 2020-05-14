import numpy as np
import open3d as o3d
import copy

from icp import icp,draw_registration_result
from nricp import nonrigidIcp



#read source file
    
sourcemesh = o3d.io.read_triangle_mesh("data/source_test.obj")
targetmesh = o3d.io.read_triangle_mesh("data/target_half.obj")
sourcemesh.compute_vertex_normals()
targetmesh.compute_vertex_normals()






#first find rigid registration
# guess for inital transform for icp
initial_guess = np.eye(4)
affine_transform = icp(sourcemesh,targetmesh,initial_guess)


#creating a new mesh for non rigid transform estimation 
refined_sourcemesh = copy.deepcopy(sourcemesh)
refined_sourcemesh.transform(affine_transform)
refined_sourcemesh.compute_vertex_normals()


#non rigid registration
deformed_mesh = nonrigidIcp(refined_sourcemesh,targetmesh)



sourcemesh.paint_uniform_color([0.1, 0.9, 0.1])
targetmesh.paint_uniform_color([0.9,0.1,0.1])
deformed_mesh.paint_uniform_color([0.1,0.1,0.9])
o3d.visualization.draw_geometries([targetmesh,deformed_mesh])


