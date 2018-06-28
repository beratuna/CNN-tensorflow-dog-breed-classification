import numpy as np
import random

def augmentation_func(imgvec):
    imgaug = np.zeros(imgvec.shape)
    for i in range(imgvec.shape[0]):
        toss = random.randint(0,19)
        if toss == 0:
            imgaug[i] = imgvec[i]
        else:
            tossed = random.randint(0,4)
            if tossed == 0:
                M   = cv2.getRotationMatrix2D((45, 45),30,1)
                img = cv2.warpAffine(imgvec[i],M,(90, 90))
                imgaug[i] = img
            elif tossed == 1:
                M   = np.float32([[1, 0, 10],[0, 1, 10]])
                img = cv2.warpAffine(imgvec[i], M, (90, 90))
                imgaug[i] = img
            elif tossed == 2:
                pts1 = np.float32([[20,20],[70,20],[20,70]])
                pts2 = np.float32([[20-15,20+0],[70-15,20-0],[20+15,70-0]])
                M = cv2.getAffineTransform(pts1,pts2)
                img = cv2.warpAffine(imgvec[i],M,(90,90))
                imgaug[i] = img
            elif tossed == 3:
                img = cv2.flip(imgvec[i], 0)
                imgaug[i] = img
            else:
                img = cv2.flip(imgvec[i], 1)
                imgaug[i] = img
    return(imgaug)