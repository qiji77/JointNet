import sys
sys.path.append("./Tool_list/")
#import pymesh
import numpy as np
#from smpl_webuser.serialization import load_model
#mesh_ref = pymesh.load_mesh("/home/djq19/workfiles/3D-CODED2/data/template/template_color.ply")
#import Pickle as pickle
import pickle
import os
import pandas as pd
from plyfile import PlyData,PlyElement
#from ply import *
from plywrite import *

trans = np.array([0,
                                  -2.1519510746002397,
                                  90.4766845703125]) / 100.0

class SMPLModel():
    def __init__(self, model_path):
        """
        SMPL model.

        Parameter:
        ---------
        model_path: Path to the SMPL model parameters, pre-processed by
        `preprocess.py`.

        """
        with open(model_path, 'rb') as f:
            params = pickle.load(f, encoding='iso-8859-1')

            self.J_regressor = params['J_regressor']
            self.weights = np.asarray(params['weights'])
            self.posedirs = np.asarray(params['posedirs'])
            self.v_template = np.asarray(params['v_template'])
            self.shapedirs = np.asarray(params['shapedirs'])
            self.f = np.asarray(params['f'])
            self.kintree_table = np.asarray(params['kintree_table'])
        print("J_regressor",self.J_regressor.shape)
        J_regressor=self.J_regressor.A
        np.save("J_regressor.npy",J_regressor)
        id_to_col = {
            self.kintree_table[1, i]: i for i in range(self.kintree_table.shape[1])
        }
        self.f.dtype='int32'
        self.parent = {
            i: id_to_col[self.kintree_table[0, i]]
            for i in range(1, self.kintree_table.shape[1])
        }

        self.pose_shape = [24, 3]
        self.beta_shape = [10]
        self.trans_shape = [3]

        self.pose = np.zeros(self.pose_shape)
        self.beta = np.zeros(self.beta_shape)
        self.trans = np.zeros(self.trans_shape)

        self.verts = None
        self.J = None
        self.R = None
        self.G = None

        self.update()

    def set_params(self, pose=None, beta=None, trans=None):
        """
        Set pose, shape, and/or translation parameters of SMPL model. Verices of the
        model will be updated and returned.

        Prameters:
        ---------
        pose: Also known as 'theta', a [24,3] matrix indicating child joint rotation
        relative to parent joint. For root joint it's global orientation.
        Represented in a axis-angle format.

        beta: Parameter for model shape. A vector of shape [10]. Coefficients for
        PCA component. Only 10 components were released by MPI.

        trans: Global translation of shape [3].

        Return:
        ------
        Updated vertices.

        """
        if pose is not None:
            self.pose = pose
        if beta is not None:
            self.beta = beta
        if trans is not None:
            self.trans = trans
        self.update()
        return self.verts

    def update(self):
        """
        Called automatically when parameters are updated.

        """
        # how beta affect body shape
        v_shaped = self.shapedirs.dot(self.beta) + self.v_template
        # joints location
        self.J = self.J_regressor.dot(v_shaped)
        pose_cube = self.pose.reshape((-1, 1, 3))
        # rotation matrix for each joint
        self.R = self.rodrigues(pose_cube)
        I_cube = np.broadcast_to(
            np.expand_dims(np.eye(3), axis=0),
            (self.R.shape[0] - 1, 3, 3)
        )
        lrotmin = (self.R[1:] - I_cube).ravel()
        # how pose affect body shape in zero pose
        v_posed = v_shaped + self.posedirs.dot(lrotmin)
        # world transformation of each joint
        G = np.empty((self.kintree_table.shape[1], 4, 4))
        G[0] = self.with_zeros(np.hstack((self.R[0], self.J[0, :].reshape([3, 1]))))
        for i in range(1, self.kintree_table.shape[1]):
            G[i] = G[self.parent[i]].dot(
                self.with_zeros(
                    np.hstack(
                        [self.R[i], ((self.J[i, :] - self.J[self.parent[i], :]).reshape([3, 1]))]
                    )
                )
            )
        # remove the transformation due to the rest pose
        G = G - self.pack(
            np.matmul(
                G,
                np.hstack([self.J, np.zeros([24, 1])]).reshape([24, 4, 1])
            )
        )
        # transformation of each vertex
        T = np.tensordot(self.weights, G, axes=[[1], [0]])
        rest_shape_h = np.hstack((v_posed, np.ones([v_posed.shape[0], 1])))
        v = np.matmul(T, rest_shape_h.reshape([-1, 4, 1])).reshape([-1, 4])[:, :3]
        self.verts = v + self.trans.reshape([1, 3])
        self.G = G

    def rodrigues(self, r):
        """
        Rodrigues' rotation formula that turns axis-angle vector into rotation
        matrix in a batch-ed manner.

        Parameter:
        ----------
        r: Axis-angle rotation vector of shape [batch_size, 1, 3].

        Return:
        -------
        Rotation matrix of shape [batch_size, 3, 3].

        """
        theta = np.linalg.norm(r, axis=(1, 2), keepdims=True)
        # avoid zero divide
        theta = np.maximum(theta, np.finfo(np.float64).tiny)
        r_hat = r / theta
        cos = np.cos(theta)
        z_stick = np.zeros(theta.shape[0])
        m = np.dstack([
            z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1],
            r_hat[:, 0, 2], z_stick, -r_hat[:, 0, 0],
            -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick]
        ).reshape([-1, 3, 3])
        i_cube = np.broadcast_to(
            np.expand_dims(np.eye(3), axis=0),
            [theta.shape[0], 3, 3]
        )
        A = np.transpose(r_hat, axes=[0, 2, 1])
        B = r_hat
        dot = np.matmul(A, B)
        R = cos * i_cube + (1 - cos) * dot + np.sin(theta) * m
        return R

    def with_zeros(self, x):
        """
        Append a [0, 0, 0, 1] vector to a [3, 4] matrix.

        Parameter:
        ---------
        x: Matrix to be appended.

        Return:
        ------
        Matrix after appending of shape [4,4]

        """
        return np.vstack((x, np.array([[0.0, 0.0, 0.0, 1.0]])))

    def pack(self, x):
        """
        Append zero matrices of shape [4, 3] to vectors of [4, 1] shape in a batched
        manner.

        Parameter:
        ----------
        x: Matrices to be appended of shape [batch_size, 4, 1]

        Return:
        ------
        Matrix of shape [batch_size, 4, 4] after appending.

        """
        return np.dstack((np.zeros((x.shape[0], 4, 3)), x))

    def save_to_obj(self, path):
        """
        Save the SMPL model into .obj file.

        Parameter:
        ---------
        path: Path to save.

        """
        print("save")
        with open(path, 'w') as fp:
            for v in self.verts:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))
            for f in self.f + 1:
                fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

m = SMPLModel('/data1/djq19/workfiles/3D-CODED2/data/basicModel_f_lbs_10_207_0_v1.0.0.pkl')
database_shape = np.load("/home/djq19/workfiles/3D-CODED2/data/smpl_data.npz")
database = np.load("/home/djq19/workfiles/VirtualBones/anim/anim1.npz")
betas = database_shape['femaleshapes']
def generate_surreal(pose, outmesh_path,trans_temp):
    """
    This function generation 1 human using a random pose and shape estimation from surreal
    """
    ## Assign gaussian pose
    #print(pose)
    beta = betas[19]
    m.pose[:] = pose.reshape(24,3)
    m.beta[:] = beta
    m.update()
    point_set = m.verts+trans_temp#m.R.astype(np.float64)
    
    #normalize
    m.f.dtype='int32'
    save(point_set,m,outmesh_path)
    '''mesh = pymesh.form_mesh(vertices=point_set, faces=m.f)
    mesh.add_attribute("red")
    mesh.add_attribute("green")
    mesh.add_attribute("blue")
    mesh.set_attribute("red", mesh_ref.get_attribute("vertex_red"))
    mesh.set_attribute("green", mesh_ref.get_attribute("vertex_green"))
    mesh.set_attribute("blue", mesh_ref.get_attribute("vertex_blue"))
    pymesh.meshio.save_mesh(outmesh_path, mesh, "red", "green", "blue", ascii=True)'''
    return

def generate_humandressed(pose, outmesh_path,trans_temp):#,rota,end_trans):
    """
    This function generation 1 human using a random pose and shape estimation from surreal
    """
    ## Assign gaussian pose
    #print(pose)
    beta = betas[19]
    m.pose[:] = pose.reshape(24,3)
    m.beta[:] = beta
    m.update()
    point_set = m.verts-trans_temp#m.R.astype(np.float64)
    print("point_set",point_set.shape)
    #point_set=np.matmul(rota,point_set.transpose()).transpose()
    #point_set=point_set-point_set[3503]+end_trans
    #normalize
    m.f.dtype='int32'
    save(point_set,m,outmesh_path)
    '''mesh = pymesh.form_mesh(vertices=point_set, faces=m.f)
    mesh.add_attribute("red")
    mesh.add_attribute("green")
    mesh.add_attribute("blue")
    mesh.set_attribute("red", mesh_ref.get_attribute("vertex_red"))
    mesh.set_attribute("green", mesh_ref.get_attribute("vertex_green"))
    mesh.set_attribute("blue", mesh_ref.get_attribute("vertex_blue"))
    pymesh.meshio.save_mesh(outmesh_path, mesh, "red", "green", "blue", ascii=True)'''
    return

def save( points,mesh,path):
        """
        Home-made function to save a ply file with colors. A bit hacky
        """
        to_write =points
        b = np.zeros((len(mesh.f), 4)) + 3
        b[:, 1:] = np.array(mesh.f)
        points2write = pd.DataFrame({
                'lst0Tite': to_write[:, 0],
                'lst1Tite': to_write[:, 1],
                'lst2Tite': to_write[:, 2],
            })
        write_ply(filename=path, points=points2write, as_text=True, text=False,
                          faces=pd.DataFrame(b.astype(int)),
                          color=False)


def generate_database_surreal(male):
    #TRAIN DATA
    nb_generated_humans = 500
    nb_generated_humans_val = 200
    if male:
        betas = database_shape['maleshapes']
        offset = 0
        offset_val = 0
    else:
        betas = database_shape['femaleshapes']
        offset = 0
        offset_val = nb_generated_humans_val

    poses = [i for i in database.keys() if "poses" in i]
    print(len(poses))
    
    params = []
    for i in range(nb_generated_humans):
        #pose, beta = get_random(poses, betas)
        
        print(beta)
        pose = database["poses"][i][:24]
        generate_surreal(pose, '/home/djq19/workfiles/VirtualBones/data/anim1_test1/' + str(offset + i) + '.ply')
        #generate_gaussian(pose, beta, './data/dataset_gaussian/' + str(offset + i) + '.ply')

    '''#VAL DATA
    for i in range(nb_generated_humans_val):
        pose, beta = get_random(poses, betas)
        #generate_surreal(pose, beta, './data/dataset-amass-val/' + str(offset_val + i) + '.ply')
        generate_gaussian(pose, beta, './data/dataset_gaussian_val/' + str(offset + i) + '.ply')'''

    return 0



if __name__ == '__main__':
    ### GENERATE MALE EXAMPLES
    
    generate_database_surreal(male=False)
    #generate_database_benthumans(male=True)
    '''
    ### GENERATE FEMALE EXAMPLES
    m = SMPLModel('./basicmodel_m_lbs_10_207_0_v1.0.0.pkl')
    database = np.load("./acting1_poses.npz")
    generate_database_surreal(male=False)
    #generate_database_benthumans(male=False)'''