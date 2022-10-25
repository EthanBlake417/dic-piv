import cv2
import numpy as np
import matplotlib.pyplot as plt
from openpiv import pyprocess, piv

# the video is the jet PIV from Youtube
# https://www.youtube.com/watch?v=EeS1rYMZUxI&ab_channel=USUExperimentalFluidDynamicsLab
# all the rights reserved to the authors

vidcap = cv2.VideoCapture('IMG_0953.MOV')       # THIS is the movie calvin sent me
success, image1 = vidcap.read()
count = 0
U = []
V = []

while success and count < 10:
    success, image2 = vidcap.read()
    # cv2.imwrite("movie_images/frame%d.jpg" % count, image2)     # save frame as JPEG file
    print('Read a new frame: ', success)
    if success:
        x, y, u, v = piv.simple_piv(image1.sum(axis=2), image2.sum(axis=2), plot=True)
        image1 = image2.copy()
        count += 1
        U.append(u)
        V.append(v)


U = np.stack(U)
Umean = np.mean(U, axis=0)
V = np.stack(V)
Vmean = np.mean(V, axis=0)



fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(image1, alpha=0.7)
ax.quiver(x, y, Umean, Vmean, scale=200, color='r', width=.008)
plt.show()
plt.plot(np.mean(Umean, axis=1) * 20, y[:, 0], color='y', lw=3)
plt.show()