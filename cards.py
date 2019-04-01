#!/usr/bin/python3

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
import imageio
import random
import colour

size = (575,1575)

img = np.zeros((size[0],size[1],3))

x, y = np.meshgrid(np.linspace(0,1,size[1]), np.linspace(0,1,size[0]))

w = int(size[0]/2 - 22)

img[:,:,0] = x*y
img[:,:,1] = x*(1-y)
img[:,:,2] = (1-x)*y

def rgb_to_lab(x):
    return colour.XYZ_to_Lab(colour.sRGB_to_XYZ(x))
def lab_to_rgb(x):
    return colour.XYZ_to_sRGB(colour.Lab_to_XYZ(x))

nx = 6
ny = 3
colors = np.zeros((3*ny+2,2*nx+2,3))
for i in range(colors.shape[0]):
    for j in range(colors.shape[1]):
        colors[i,j,:] = random.choice((rgb_to_lab([0,0,0]),
                                       rgb_to_lab([1,1,1]),
                                       rgb_to_lab([1,0,0]),
                                       rgb_to_lab([0,1,0]),
                                       rgb_to_lab([0,0,1]),
                                       rgb_to_lab([1,1,0]),
                                       rgb_to_lab([0.4,0,0.4]),
                                       rgb_to_lab([1,0.5,0.5]),
        ))

print(colour.sRGB_to_XYZ([1,0,0]))
print(rgb_to_lab([1,0,0]))

for ix in range(nx+1):
    for iy in range(ny+1):
        ctop = colors[2*iy    ,ix  ,:]
        cbot = colors[2*(iy+1),ix  ,:]
        clef = colors[2*iy+1  ,ix  ,:]
        crig = colors[2*iy+1  ,ix+1,:]
        for x in range(w):
            for y in range(w):
                idx = ix*w+x + 22 - w
                idy = iy*w+y + 22 - w
                if 0<=idx<size[1] and 0<=idy<size[0]:
                    xx = x/(w-1)
                    yy = y/(w-1)
                    mypow = 4
                    wtop = xx**mypow*(1-xx)**mypow*(1-yy)
                    wbot = xx**mypow*(1-xx)**mypow*yy
                    wlef =(1-xx)*yy**mypow*(1-yy)**mypow
                    wrig = xx*yy**mypow*(1-yy)**mypow
                    color = wtop*ctop
                    color += wbot*cbot
                    color += wlef*clef
                    color += wrig*crig
                    color /= wlef + wrig + wtop + wbot
                    img[idy,idx,:] = color

rgb = lab_to_rgb(img)
rgb[rgb<0] = 0
rgb[rgb>1] = 1
plt.imshow(rgb)
imageio.imwrite('card.png', rgb)

plt.show()
