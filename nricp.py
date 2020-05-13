import numpy as np
import scipy.io
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy import sparse
from sklearn.neighbors import NearestNeighbors
import cv2
from sksparse.cholmod import cholesky_AAt

import open3d as o3d
import copy

from icp import icp,draw_registration_result


def spsolve_chol(sparse_X, dense_b):

    factor = cholesky_AAt(sparse_X.T)
    return factor(sparse_X.T.dot(dense_b)).toarray()



#read source file
    
sourcemesh = o3d.io.read_triangle_mesh("data/source_test.obj")
targetmesh = o3d.io.read_triangle_mesh("data/target_half.obj")
sourcemesh.compute_vertex_normals()
targetmesh.compute_vertex_normals()






#first find rigid registration

# guess for inital transform

affine_transform = icp(sourcemesh,targetmesh)


#draw_registration_result(sourcemesh, targetmesh, np.eye(4))



refined_sourcemesh = copy.deepcopy(sourcemesh)

refined_sourcemesh.transform(affine_transform)


target_vertices = np.array(targetmesh.vertices)
#target_vertices = np.hstack((target_vertices,np.ones((len(target_vertices),1))))
source_vertices = np.array(refined_sourcemesh.vertices)

#compute normals again for refined source mesh

refined_sourcemesh.compute_vertex_normals()

source_mesh_normals = np.array(refined_sourcemesh.vertex_normals)
target_mesh_normals = np.array(targetmesh.vertex_normals)



nbrs50 = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(target_vertices)


n_source_verts = source_vertices.shape[0]



sourcemesh_faces = np.array(sourcemesh.triangles)

l=[]
for i in sourcemesh_faces:
    s = np.sort(i)
    l.append(tuple([s[0],s[1]]))
    l.append(tuple([s[0],s[2]]))
    l.append(tuple([s[1],s[2]]))
    
edgeset = set(l)
n_source_edges = len(edgeset)
print("num edges:", n_source_edges)


M = sparse.lil_matrix((n_source_edges, n_source_verts), dtype=np.float32)

for i, t in enumerate(edgeset):
    M[i, t[0]] = -1
    M[i, t[1]] = 1

gamma = 1
G = np.diag([1, 1, 1, gamma]).astype(np.float32)


kron_M_G = sparse.kron(M, G)



#I = np.array(range(n_source_verts)).reshape(-1,1)
#J = 4*I

# using lil_matrix becaiuse chinging sparsity in csr is expensive 
D = sparse.lil_matrix((n_source_verts,n_source_verts*4), dtype=np.float32)
j_=0
for i in range(n_source_verts):
    D[i,j_:j_+3]=source_vertices[i,:]
    D[i,j_+3]=1
    j_+=4

n_source_normals = len(source_mesh_normals) #will be equal to n_source_verts
DN = sparse.lil_matrix((n_source_normals,n_source_normals*4), dtype=np.float32)
j_=0
for i in range(n_source_normals):
    DN[i,j_:j_+3]=source_mesh_normals[i,:]
    DN[i,j_+3]=1
    j_+=4


X_= np.concatenate((np.eye(3),np.array([[0,0,0]])),axis=0)
X = np.tile(X_,(n_source_verts,1))

targetmesh.paint_uniform_color([0.9,0.1,0.1])
refined_sourcemesh.paint_uniform_color([0.1,0.1,0.9])
o3d.visualization.draw_geometries([targetmesh,refined_sourcemesh])

# X for transformations and D for vertex info in sparse matrix

normalWeight=False


alphas = [100,50,35,20,15, 7,3,2,1]
betas = [15,14, 3, 2, 0.5, 0,0,0]


alpha_stiffness= alphas[0]

oldX = 10*X


#np.linalg.norm(())

alphas = np.linspace(200,1,20)

for num_,alpha_stiffness in enumerate(alphas):
    
    print("step- {}/20".format(num_))
    
    for i in range(3):
        
        wVec = np.ones((n_source_verts,1))
        
        vertsTransformed = D*X
        
        distances, indices = nbrs50.kneighbors(vertsTransformed)
        
        indices = indices.squeeze()
        
        matches = target_vertices[indices]
        
        mismatches = np.where(distances>0.03)[0]
        
        
        
        
        if normalWeight:
            
            normalsTransformed = DN*X
            
            corNormalsTarget = target_mesh_normals[indices]
            
            crossNormals = np.cross(corNormalsTarget, normalsTransformed)
            
            crossNormalsNorm = np.sqrt(np.sum(crossNormals**2,1))
            
            dotNormals = np.sum(corNormalsTarget*normalsTransformed,1)
            
            angles =np.arctan(dotNormals/crossNormalsNorm)
            
            wVec = wVec *(angles<np.pi/4).reshape(-1,1)
            
            
            

            
        wVec[mismatches] = 0
            
        
    #    mesh = np.hstack((mesh, np.ones([n, 1])))
        
        
        U = wVec*matches
        
        A = sparse.csr_matrix(sparse.vstack([alpha_stiffness * kron_M_G,   D.multiply(wVec) ]))
        
        
        B = sparse.lil_matrix((4 * n_source_edges + n_source_verts, 3), dtype=np.float32)
        
        B[4 * n_source_edges: (4 * n_source_edges +n_source_verts), :] = U
        
        
        
        oldX = X
        
        X = spsolve_chol(A, B)
    
        
vertsTransformed = D*X;

refined_sourcemesh.vertices = o3d.utility.Vector3dVector(vertsTransformed)



sourcemesh.paint_uniform_color([0.1, 0.9, 0.1])
targetmesh.paint_uniform_color([0.9,0.1,0.1])
refined_sourcemesh.paint_uniform_color([0.1,0.1,0.9])
o3d.visualization.draw_geometries([targetmesh,refined_sourcemesh])

#distances, indices = nbrs50.kneighbors(vertsTransformed)
#        
#indices = indices.squeeze()
#
#matches = target_vertices[indices]
#
#matches = matches[np.where(distances<0.01)[0]]

#listpairs = wVec==1
#vertsTransformed[np.where(distances<0.01)[0]] = matches[np.where(distances<0.01)[0]]


matcheindices = np.where(wVec > 0)[0]

vertsTransformed[matcheindices]=matches[matcheindices]

refined_sourcemesh.vertices = o3d.utility.Vector3dVector(vertsTransformed)


sourcemesh.paint_uniform_color([0.1, 0.9, 0.1])
targetmesh.paint_uniform_color([0.9,0.1,0.1])
refined_sourcemesh.paint_uniform_color([0.1,0.1,0.9])
o3d.visualization.draw_geometries([targetmesh,refined_sourcemesh])
