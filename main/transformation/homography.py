"""
Homography fitting functions
You should write these
"""
import numpy as np
from common import homography_transform


def fit_homography(XY):
    '''
    Given a set of N correspondences XY of the form [x,y,x',y'],
    fit a homography from [x,y,1] to [x',y',1].
    
    Input - XY: an array with size(N,4), each row contains two
            points in the form [x_i, y_i, x'_i, y'_i] (1,4)
    Output -H: a (3,3) homography matrix that (if the correspondences can be
            described by a homography) satisfies [x',y',1]^T === H [x,y,1]^T

    '''
    #print(XY)
    #pre is the initial set of values (x, y) 
    #aft is outcome of transformation. (x', y')
    pre = XY[:,0:2]
    aft = XY[:,2:4]

    dim = len(pre) 

    #we want [x, y, 1] for all cases.
    pre = np.hstack((pre, np.ones((dim,1)))) 
    aft = np.hstack((aft, np.ones((dim,1))))

    #print(pre)
    #print(aft)

    A = np.zeros((2*dim, 9))

    #print(A)

    for i in range(dim):
        j = i * 2
        A[j, 3:6] = -1*pre[i]
        A[j, 6:] = aft[i][1]*pre[i]
        A[j+1, 0:3] = pre[i]
        A[j+1, 6:] = -1*aft[i][0]*pre[i]


    A = A.astype(np.float64)

    #print(A)

    ATA = np.dot(A.T,A)
    w,v = np.linalg.eig(ATA)
    target = np.argmin(w)
    h = v[:,target]
    h = h/h[8]
    H = h.reshape(3,3)
    # print(H)
    return H


def distance(XY, H):
    pre_x = XY[:, [0]]
    pre_y = XY[:, [1]]
    aft_x = XY[:, [2]]
    aft_y = XY[:, [3]]

    T = homography_transform(np.hstack((pre_x, pre_y)), H)
    dist = np.linalg.norm(T - np.hstack((aft_x, aft_y)), axis = 1)
    return dist

def RANSAC_fit_homography(XY, eps=1, nIters=1000):
    '''
    Perform RANSAC to find the homography transformation 
    matrix which has the most inliers
        
    Input - XY: an array with size(N,4), each row contains two
            points in the form [x_i, y_i, x'_i, y'_i] (1,4)
            eps: threshold distance for inlier calculation
            nIters: number of iteration for running RANSAC
    Output - bestH: a (3,3) homography matrix fit to the 
                    inliers from the best model.

    Hints:
    a) Sample without replacement. Otherwise you risk picking a set of points
       that have a duplicate.
    b) *Re-fit* the homography after you have found the best inliers
    '''
    bestH, bestCount, bestInliers = np.eye(3), -1, np.zeros((XY.shape[0],))
    bestRefit = np.eye(3)


    for iter in range(nIters):
        random_points = np.random.choice(XY.shape[0], size=4, replace=False)
        H = fit_homography(XY[random_points,:])
        dist = distance(XY, H)
        inliers = (dist < eps)
        count = np.sum(dist < eps)
        if count > bestCount:
            bestInliers = inliers
            bestCount = count
            bestH = H
            
    bestRefit = fit_homography(XY[bestInliers])

    return bestRefit

if __name__ == "__main__":
    #If you want to test your homography, you may want write any code here, safely
    #enclosed by a if __name__ == "__main__": . This will ensure that if you import
    #the code, you don't run your test code too
    pass
