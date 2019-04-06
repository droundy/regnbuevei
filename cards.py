#!/usr/bin/python3

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
import imageio
import random, os, colour

size = (1575,975)

img = np.zeros((size[0],size[1],3))

x, y = np.meshgrid(np.linspace(0,1,size[1]), np.linspace(0,1,size[0]))

w = int(size[1]/2 - 22)

img[:,:,0] = x*y
img[:,:,1] = x*(1-y)
img[:,:,2] = (1-x)*y

def rgb_to_lab(x):
    return colour.XYZ_to_Lab(colour.sRGB_to_XYZ(x))
def lab_to_rgb(x):
    return colour.XYZ_to_sRGB(colour.Lab_to_XYZ(x))

spectrum = np.array([[0,0,0],
                     [0.4,0,0.4],
                     [0,0,1],
                     [0,0.5,1],
                     [0,1,0],
                     [1,1,0],
                     [0.975,0.451,0.023],
                     [1,0,0],
                     [0,0,0],
])
spec_coords = np.array([0,0.1,0.3,0.4,0.5,0.7, 0.8, 0.9, 1])
print(np.linspace(0.1,0.9,len(spectrum)-2))
print(len(spectrum), len(spec_coords))

def scalar_to_rgb(z):
    l = np.interp(z, spec_coords, spectrum[:,0])
    a = np.interp(z, spec_coords, spectrum[:,1])
    b = np.interp(z, spec_coords, spectrum[:,2])
    return np.moveaxis(np.array([l,a,b]), 0, -1)

nx = 3
ny = 4
colors = np.zeros((3*ny+2,2*nx+2,3))

X, Y = np.meshgrid(np.linspace(0,1,w), np.linspace(0,1,w))
XX = 1-X
YY = 1-Y
R = np.sqrt(X**2+Y**2)
phi = np.arctan2(Y,X)
phi[Y>X] = np.arctan2(X,Y)[Y>X]
wtl = 1 - R

wtl[wtl<0] = 0
wtr = wtl[:,::-1]
wbl = wtl[::-1,:]
wbr = wtl[::-1,::-1]

plt.pcolor(wtl)
plt.colorbar()
plt.figure()

norm = wtl + wtr + wbl + wbr
print('big norm', norm[norm>1])
wtl[norm>1] /= norm[norm>1]
wtr[norm>1] /= norm[norm>1]
wbl[norm>1] /= norm[norm>1]
wbr[norm>1] /= norm[norm>1]

# wtl[norm<1] += 1-norm[norm<1]
# wtr[norm<1] += 1-norm[norm<1]
# wbl[norm<1] += 1-norm[norm<1]
# wbr[norm<1] += 1-norm[norm<1]
weights = np.array([[wtl, wtr], [wbl, wbr]])

plt.pcolor(wtl)
plt.colorbar()
plt.show()
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
            for x in range(mx):
                if corners[x] > corners[x+1]:
                    return True
            for x in range(mx,3):
                if corners[x] < corners[x+1]:
                    return True
    return False # no bad corners

for i in range(20):
    num_corner_kinds = 5
    scalars = np.random.choice(np.linspace(0,1,num_corner_kinds), (ny+2,nx+2))
    while bad_corners(scalars):
        scalars = np.random.choice(np.linspace(0,1,num_corner_kinds), (ny+2,nx+2))
    for ix in range(nx+1):
        for iy in range(ny+1):
            square = scalar_to_rgb(weights[0,0]*scalars[iy  , ix  ] +
                                   weights[1,0]*scalars[iy+1, ix  ] +
                                   weights[0,1]*scalars[iy  , ix+1] +
                                   weights[1,1]*scalars[iy+1, ix+1])
            offy = iy*w+22-w
            offx = ix*w+22-w
            img[max(0,offy):offy+w, max(0,offx):offx+w,:] = square[max(0,-offy):max(0,min(w,size[0]-offy)),
                                                                   max(0,-offx):max(0,min(w,size[1]-offx)),
                                                                   :]

    # rgb = lab_to_rgb(img)
    rgb = img
    rgb[rgb<0] = 0
    rgb[rgb>1] = 1
    imageio.imwrite('card-{}[face].png'.format(i), rgb)
    imageio.imwrite('card-{}[back].png'.format(i), rgb[:,::-1,:])
#     plt.imshow(img)
# plt.show()
