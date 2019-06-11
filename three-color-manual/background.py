#!/usr/bin/python3

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
import imageio
import random, os

np.random.seed(0)

size = (int(11*150),int(8.5*150))

img = np.zeros((size[0],size[1],3))

x, y = np.meshgrid(np.linspace(0,1,size[1]), np.linspace(0,1,size[0]))

w = 33

img[:,:,0] = 1
img[:,:,1] = 1
img[:,:,2] = .9

specs = [
    (0.00, [1.000, 1.000, 1.000]),
    (0.10, [0.400, 0.000, 0.400]),
    (0.30, [0.000, 0.000, 1.000]),
    (0.40, [0.000, 0.800, 0.800]),
    (0.50, [0.000, 0.900, 0.000]),
    (0.60, [0.600, 1.000, 0.000]),
    (0.70, [1.000, 1.000, 0.000]),
    (0.80, [1.000, 0.451, 0.023]),
    (0.90, [1.000, 0.000, 0.000]),
    (1.00, [1.000, 1.000, 1.000]),
]
spectrum = np.array(list([x[1] for x in specs]))
spec_coords = np.array(list([x[0] for x in specs]))
def scalar_to_rgb(z):
    l = np.interp(z, spec_coords, spectrum[:,0])
    a = np.interp(z, spec_coords, spectrum[:,1])
    b = np.interp(z, spec_coords, spectrum[:,2])
    return np.moveaxis(np.array([l,a,b]), 0, -1)

nx = int(size[1]/w)-1
ny = int(size[0]/w)-1
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

weights = np.array([[wtl, wtr], [wbl, wbr]])

choices = np.linspace(0,1,4)
choices = np.append(choices, 0)
choices = np.append(choices, 1)

scalars = np.zeros((ny+2, nx+2))
scalars[:,:] = choices[0]

scalars[0,:] = choices[3]
scalars[:,0] = choices[3]
scalars[:,-1] = choices[3]
# scalars[1,:] = 1
# scalars[2,:] = 2
scalars[-1,:] = choices[3]

page2_scalars = scalars*1

scalars[0:6,:19] = choices[3]
scalars[1:6,18] = choices[1]
scalars[1:6,17] = choices[2]

scalars[1:10,1] = choices[0]
scalars[1:3,2] = choices[0]
scalars[4:10,2] = choices[0]

# scalars = choose([
#     [3  ,   3  , 3  , 3  , 3  , 3  ,   3  ],
#     [3  ,   0  , 0  , 0  , 0  , 0  ,   0  ],
#     [3  ,   0  , 0  , 3  , 0  , 0  ,   0  ],
#     [3  ,   3  , 3  , 3  , 3  , 3  ,   0  ],
#     [0  ,   3  , 3  , 3  , 3  , 3  ,   0  ],
#     [0  ,   3  , 3  , 3  , 3  , 3  ,   0  ],
#     [0  ,   3  , 3  , 3  , 3  , 3  ,   0  ],
#     [0  ,   3  , 3  , 3  , 3  , 3  ,   0  ],
#     [0  ,   3  , 3  , 3  , 3  , 3  ,   0  ],
#     [0  ,   3  , 3  , 3  , 3  , 3  ,   0  ],
#     [0  ,   3  , 3  , 3  , 3  , 3  ,   0  ],
#     [0  ,   3  , 3  , 3  , 3  , 3  ,   0  ],
#     [0  ,   3  , 3  , 2  , 3  , 3  ,   0  ],
#     [0  ,   3  , 3  , 3  , 3  , 3  ,   0  ],
#     [0  ,   0  , 0  , 0  , 0  , 0  ,   0  ],
#     [0  ,   0  , 3  , 3  , 3  , 3  ,   0  ],
#     [0  ,   0  , 3  , 3  , 3  , 3  ,   0  ],
#     [0  ,   0  , 3  , 3  , 3  , 3  ,   0  ],
# ])
for scalars in [scalars, page2_scalars]:
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
            offy = iy*w + (img.shape[0]%w)
            offx = ix*w + (img.shape[1]%w)
            img[max(0,offy):offy+w, max(0,offx):offx+w,:] = square[max(0,-offy):max(0,min(w,size[0]-offy)),
                                                               max(0,-offx):max(0,min(w,size[1]-offx)),
                                                                   :]

    # rgb = lab_to_rgb(img)
    rgb = img
    rgb[rgb<0] = 0
    rgb[rgb>1] = 1
    rgb = (rgb*255).astype(np.uint8)
    if (scalars == page2_scalars).all():
        imageio.imwrite('background.png', rgb)
    else:
        imageio.imwrite('title-background.png', rgb)
# plt.imshow(img)
# plt.show()
