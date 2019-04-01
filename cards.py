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
                                       rgb_to_lab([0.082,0.690,0.102]),
                                       rgb_to_lab([0,0,1]),
                                       rgb_to_lab([1,1,0]),
                                       rgb_to_lab([0.975,0.451,0.023]),
                                       rgb_to_lab([0.4,0,0.4]),
                                       rgb_to_lab([1,0.5,0.5]),
        ))

print(colour.sRGB_to_XYZ([1,0,0]))
print(rgb_to_lab([1,0,0]))

#print(xx)

# plt.contourf(xx)
# plt.colorbar()
# plt.figure()
# plt.contourf(yy)
# plt.colorbar()
# plt.figure()
#plt.show()

for ix in range(nx+1):
    for iy in range(ny+1):
        X, Y = np.meshgrid(np.linspace(0,1,w), np.linspace(0,1,w))
        mypow = 1+np.random.random()*4
        wtop = X**mypow*(1-X)**mypow*(1-Y)
        wbot = X**mypow*(1-X)**mypow*Y
        wrig = Y**mypow*(1-Y)**mypow*(1-X)
        wlef = Y**mypow*(1-Y)**mypow*X
        xx = (wrig - wlef)/(wrig+wlef+wtop+wbot)
        yy = (wtop - wbot)/(wrig+wlef+wtop+wbot)
        phi = np.arctan2(yy,xx)
        r = np.sqrt(xx**2 + yy**2)

        cbot = colors[2*iy    ,ix  ,:]
        ctop = colors[2*(iy+1),ix  ,:]
        crig = colors[2*iy+1  ,ix  ,:]
        clef = colors[2*iy+1  ,ix+1,:]
        twist = (1-r)**(1.5+np.random.random()*4)*(np.random.random()-0.5)*32
        XX = xx*np.cos(twist)+yy*np.sin(twist)
        YY = yy*np.cos(twist)-xx*np.sin(twist)
        dlef = np.sqrt((XX+1)**2 + YY**2)
        drig = np.sqrt((XX-1)**2 + YY**2)
        dtop = np.sqrt(XX**2 + (YY+1)**2)
        dbot = np.sqrt(XX**2 + (YY-1)**2)
        apow=4
        square = (np.tensordot(dlef**apow*dtop**apow*dbot**apow,crig,0)
                  + np.tensordot(drig**apow*dtop**apow*dbot**apow,clef,0)
                  + np.tensordot(dbot**apow*dlef**apow*drig**apow,ctop,0)
                  + np.tensordot(dtop**apow*dlef**apow*drig**apow,cbot,0))/(
                      np.tensordot(drig**apow*dlef**apow*dtop**apow
                                   +dlef**apow*drig**apow*dbot**apow
                                   +dtop**apow*dbot**apow*dlef**apow
                                   +dbot**apow*dtop**apow*drig**apow, [1,1,1], 0)
                  )
        offy = iy*w+22-w
        offx = ix*w+22-w
        img[max(0,offy):offy+w, max(0,offx):offx+w,:] = square[max(0,-offy):max(0,min(w,size[0]-offy)),
                                                               max(0,-offx):max(0,min(w,size[1]-offx)),
                                                               :]
        # for x in range(w):
        #     for y in range(w):
        #         idx = ix*w+x + 22 - w
        #         idy = iy*w+y + 22 - w
        #         if 0<=idx<size[1] and 0<=idy<size[0]:
        #             xx = x/(w-1)
        #             yy = y/(w-1)
        #             wtop = xx**mypow*(1-xx)**mypow*(1-yy)
        #             wbot = xx**mypow*(1-xx)**mypow*yy
        #             wlef =(1-xx)*yy**mypow*(1-yy)**mypow
        #             wrig = xx*yy**mypow*(1-yy)**mypow
        #             color = wtop*ctop
        #             color += wbot*cbot
        #             color += wlef*clef
        #             color += wrig*crig
        #             color /= wlef + wrig + wtop + wbot
        #             img[idy,idx,:] = color

rgb = lab_to_rgb(img)
rgb[rgb<0] = 0
rgb[rgb>1] = 1
plt.imshow(rgb)
imageio.imwrite('card.png', rgb)

plt.show()
