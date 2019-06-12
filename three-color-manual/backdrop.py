#!/usr/bin/python3

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
import imageio
import random, os

np.random.seed(0)

size = (600,1600)

img = np.zeros((size[0],size[1],3))

x, y = np.meshgrid(np.linspace(0,1,size[1]), np.linspace(0,1,size[0]))

w = 600

img[:,:,0] = x*y
img[:,:,1] = x*(1-y)
img[:,:,2] = (1-x)*y

specs = [
    (0.00, [0.000, 0.000, 0.000]),
    (0.10, [0.400, 0.000, 0.400]),
    (0.30, [0.000, 0.000, 1.000]),
    (0.40, [0.000, 0.800, 0.800]),
    (0.50, [0.000, 0.900, 0.000]),
    (0.60, [0.600, 1.000, 0.000]),
    (0.70, [1.000, 1.000, 0.000]),
    (0.80, [1.000, 0.451, 0.023]),
    (0.90, [1.000, 0.000, 0.000]),
    (1.00, [0.000, 0.000, 0.000]),
]
spectrum = np.array(list([x[1] for x in specs]))
spec_coords = np.array(list([x[0] for x in specs]))
def scalar_to_rgb(z):
    l = np.interp(z, spec_coords, spectrum[:,0])
    a = np.interp(z, spec_coords, spectrum[:,1])
    b = np.interp(z, spec_coords, spectrum[:,2])
    return np.moveaxis(np.array([l,a,b]), 0, -1)

nx = 7
ny = 4
colors = np.zeros((3*ny+2,2*nx+2,3))

X, Y = np.meshgrid(np.linspace(0,1,w), np.linspace(0,1,w))
XX = 1-X
YY = 1-Y
R = np.sqrt(X**2+Y**2)
phi = np.arctan2(Y,X)
phi[Y>X] = np.arctan2(X,Y)[Y>X]
middle = (X*Y)**0.5
wtl = 1 - R
wtl = (XX*YY)**1.6
a=4
wtl = XX*YY**a/(YY**a + Y**a + XX**a + X**a) + YY*XX**a/(YY**a + Y**a + XX**a + X**a)
# wtl = XX*YY + YY*X # middle*(XX*YY)**2 + (1-middle)*(1-R)

# plt.pcolor(middle)
# plt.colorbar()
# plt.figure()

# wtl[wtl<0] = 0
wtr = 1*wtl[:,::-1]
wbl = 1*wtl[::-1,:]
wbr = 1*wtl[::-1,::-1]

# plt.pcolor(wtl)
# plt.colorbar()
# plt.figure()

# norm = wtl + wtr + wbl + wbr
# print('big norm', norm[norm>1])
# wtl /= norm
# wtr /= norm
# wbl /= norm
# wbr /= norm
# wtl[norm>1] /= norm[norm>1]
# wtr[norm>1] /= norm[norm>1]
# wbl[norm>1] /= norm[norm>1]
# wbr[norm>1] /= norm[norm>1]

# wtl[norm<1] += 1-norm[norm<1]
# wtr[norm<1] += 1-norm[norm<1]
# wbl[norm<1] += 1-norm[norm<1]
# wbr[norm<1] += 1-norm[norm<1]
weights = np.array([[wtl, wtr], [wbl, wbr]])

# plt.pcolor(wtl)
# plt.colorbar()
# plt.show()
# exit(1)

def bad_corners(s):
    for i in range(s.shape[0]-1):
        for j in range(s.shape[1]-1):
            corners = [s[i,j], s[i+1,j], s[i+1,j+1], s[i,j+1]]
            while corners[0] > corners[1]:
                corners = [corners[1], corners[2], corners[3], corners[0]]
            while corners[0] > corners[3]:
                corners = [corners[3], corners[0], corners[1], corners[2]]
            mx = 0
            mn = 0
            for x in range(4):
                if corners[x] > corners[mx]:
                    mx = x
                if corners[x] < corners[mn]:
                    mn = x
            if mn != 0:
                return True
            if mx == 0:
                return True # reject squares of solid color
            for x in range(mx):
                if corners[x] > corners[x+1]:
                    return True
            for x in range(mx,3):
                if corners[x] < corners[x+1]:
                    return True
    return False # no bad corners

choices = np.linspace(0,1,3)
choices = np.append(choices, 0)
choices = np.append(choices, 1)
def choose(x):
    x = [[choices[elem % 3] for elem in y] for y in x]
    return np.array(x).T
def choose2(x):
    x = np.array(x)
    return 

scalars = choose([
    [1  ,   2  , 1  , 2  , 0  ,   1  ],
    [1  ,   2  , 1  , 2  , 1  ,   1  ],
    [0  ,   2  , 1  , 2  , 1  ,   2  ],
    [2  ,   2  , 1  , 2  , 1  ,   0  ],
    [0  ,   1  , 1  , 2  , 1  ,   1  ],
    [0  ,   1  , 1  , 2  , 1  ,   1  ],
    [1  ,   1  , 1  , 2  , 1  ,   1  ],
    [1  ,   1  , 1  , 2  , 1  ,   1  ],
    [2  ,   1  , 1  , 2  , 1  ,   1  ],
    [1  ,   1  , 1  , 2  , 1  ,   1  ],
])
for ix in range(nx+1):
    for iy in range(ny+1):
        ww = weights*1
        sc = scalars[iy:iy+2, ix:ix+2]
        goodrow = ww[0,0][:,0]*1
        for iii in range(4):
            if sc[0,0] == sc[1,0]:
                for jj in range(w):
                    row = ww[0,0][:,jj] + ww[1,0][:,jj]
                    extra = row - row[0]
                    norm = ww[1,1][:,jj] + ww[0,1][:,jj]
                    norm[norm==0] = 3
                    extra1 = extra*ww[1,1][:,jj]/norm
                    extra1[norm==3] = 0
                    ww[1,1][1:-1,jj] += extra1[1:-1]
                    ww[0,1][1:-1,jj] += (extra - extra1)[1:-1]
                    ww[0,0][1:-1,jj] = row[0]
                    ww[1,0][1:-1,jj] = row[0]
                ww[0,0] *= YY
                ww[1,0] *= Y
            sc = np.rot90(sc)
            ww = np.rot90(ww)
            for xxx in [0,1]:
                for yyy in [0,1]:
                    ww[yyy,xxx] = np.rot90(ww[yyy,xxx])
        for iii in range(4):
            if sc[1,1] == sc[1,0] and sc[1,1] == sc[0,1]:
                ww[0,0] = np.interp(R, np.linspace(0,1,w), goodrow)
                ww[0,0][ww[0,0]<0] = 0
                ww[1,0] = (1 - ww[0,0])/3
                ww[1,1] = (1 - ww[0,0])/3
                ww[0,1] = (1 - ww[0,0])/3
            sc = np.rot90(sc)
            ww = np.rot90(ww)
            for xxx in [0,1]:
                for yyy in [0,1]:
                    ww[yyy,xxx] = np.rot90(ww[yyy,xxx])
        square = scalar_to_rgb(ww[0,0]*sc[0,0] +
                               ww[1,0]*sc[1,0] +
                               ww[0,1]*sc[0,1] +
                               ww[1,1]*sc[1,1])
        offy = iy*w
        offx = ix*w
        img[max(0,offy):offy+w, max(0,offx):offx+w,:] = square[max(0,-offy):max(0,min(w,size[0]-offy)),
                                                               max(0,-offx):max(0,min(w,size[1]-offx)),
                                                                   :]

# rgb = lab_to_rgb(img)
rgb = img
rgb[rgb<0] = 0
rgb[rgb>1] = 1
rgb = (rgb*255).astype(np.uint8)
imageio.imwrite('backdrop.png', rgb)
# plt.imshow(img)
# plt.show()
