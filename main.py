import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

import argparse

def get_homography_matrix(src, dst, size):
    A = None
    for idx in range(size):
        x = src[idx][0]
        y = src[idx][1]
        u = dst[idx][0]
        v = dst[idx][1]
        h1 = np.array([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        h2 = np.array([0, 0, 0, -x, -y, -1, v * x, v * y, v])
        if A is None:
            A = np.vstack((h1, h2))
        else:
            A = np.vstack((A, h1))
            A = np.vstack((A, h2))
    u, s, v = np.linalg.svd(A)
    h = np.reshape(v[8], (3, 3))
    h = (1 / h.item(8)) * h
    return h
def count_inliers(src, dst, H, threshold):
    n_samples = len(src)
    homo_src = np.hstack((src, np.ones((n_samples, 1))))
    homo_dst = np.hstack((dst, np.ones((n_samples, 1))))

    estimate_dst = np.dot(homo_src, H.T)
    estimate_dst[:, 0] /= estimate_dst[:, 2]
    estimate_dst[:, 1] /= estimate_dst[:, 2]
    estimate_dst[:, 2] = 1
    error = (estimate_dst - homo_dst) ** 2
    error = np.sum(error, axis=1)
    error = np.sqrt(error)
    return (error < threshold).sum()

def RANSAC(N, src, dst, threshold):
    n_samples = len(src)
    best_inliers = 0
    best_H = None
    for _ in range(N):
        random_idx = np.random.randint(n_samples, size=4)
        H = get_homography_matrix(src[random_idx], dst[random_idx], size = 4)
        n_inliers = count_inliers(src, dst, H, threshold)
        if (n_inliers > best_inliers):
            best_inliers = n_inliers
            best_H = H
    print('Best number of inliers', best_inliers)
    return best_H

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('-i1', '--image1', required=True, help = 'Directory to image 1')
    ap.add_argument('-i2', '--image2', required=True, help = 'Directory to image 2')
    ap.add_argument('-t', '--threshold', required=False, help = 'Threshold of RANSAC, default is 5.0')
    ap.add_argument('-o', '--output', required=False, help = 'Output for result image, default is output.jpg')
    args = vars(ap.parse_args())

    img1 = cv.imread(args['image1'], cv.IMREAD_GRAYSCALE)
    img2 = cv.imread(args['image2'], cv.IMREAD_GRAYSCALE)
    threshold = None
    if args['threshold'] is None:
        threshold = 5.0
    else:
        threshold = np.float32(args['threshold'])

    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)

    good = []
    for m,n in matches:
        if m.distance < 0.9*n.distance:
            good.append([m])
    # cv.drawMatchesKnn expects list of lists as matches.
    img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3)
    plt.show()
    matches = np.asarray(good)
    print('good matches', len(matches))
    if (len(matches[:,0]) >= 4):
        src = np.float32([ kp1[m.queryIdx].pt for m in matches[:,0] ]).reshape(-1,2)
        dst = np.float32([ kp2[m.trainIdx].pt for m in matches[:,0] ]).reshape(-1,2)
        H = RANSAC(5000, src, dst, threshold)
    else:
        raise AssertionError('Can’t find enough keypoints.')
    print(H)
    dst = cv.warpPerspective(img1,H,((img1.shape[1] + img2.shape[1]), img2.shape[0])) #wraped image
    dst[0:img2.shape[0], 0:img2.shape[1]] = img2 #stitched image
    cv.imwrite('output.jpg',dst)
    plt.imshow(dst)
    plt.show()
    #while True:
    #    if cv.waitKey(0) == ord('q'):
    #        break
    #cv.destroyAllWindows()
