import numpy as np
def reshape(traj, substeps):
    '''
    Reshape a trajectory of shape [m,D] to [substeps, m-substeps+1, D].
    For example, traj=[[1,1],[2,2],[3,3],[4,4],[5,5]], substeps=2, then, 
    return [[[1,1],[2,2],[3,3],[4,4]], [[2,2],[3,3],[4,4],[5,5]]]

    '''
    Y=[]
    for j in range(0,max(int(traj.shape[0]-substeps+1),1)):
        yj = traj[j: j+substeps]
        Y.append(yj)
    y_r=np.stack(Y, axis=1)
    return y_r

def l1(f):
    return np.abs(f).max(-1).mean()


substeps={  'AB': {'1': 2,        
                   '2': 3,
                   '3': 4,
                   '4': 5},
            'AM': {'1': 2,
                   '2': 2,
                   '3': 3,
                   '4': 4},
            'Ny': {'1': 3,
                   '2': 4,
                   '3': 5},
            'MS': {'1': 3,
                   '2': 3,
                   '3': 3},
            'BDF': {'1': 2,
                    '2': 3,
                    '3': 4,
                    '4': 5}
            }