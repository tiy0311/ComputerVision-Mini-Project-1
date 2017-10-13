import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')

import cv2
import numpy as np
from matplotlib import pyplot as plt


def calcAndDrawHist(image, color):    
    hist= cv2.calcHist([image], [0], None, [256], [0.0,255.0])    
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(hist)    
    histImg = np.zeros([256,256,3], np.uint8)    
    hpt = int(0.9* 256);    
                    
    for h in range(256):    
        intensity = int(hist[h]*hpt/maxVal)    
        cv2.line(histImg,(h,256), (h,256-intensity), color)    
                                        
    return histImg;  


img = cv2.imread('mp1a.jpg')
b,g,r = cv2.split(img)

histImgB = calcAndDrawHist(b, [255,0,0])
cv2.imshow("histImgB", histImgB)
histImgG = calcAndDrawHist(g, [0,255,0])
cv2.imshow("histImgG", histImgG)
histImgR = calcAndDrawHist(r, [0,0,255])
cv2.imshow("histImgR", histImgR)

hist_b,bins_b = np.histogram(b.flatten(),256,[0,256])
hist_g,bins_g = np.histogram(g.flatten(),256,[0,256])
hist_r,bins_r = np.histogram(r.flatten(),256,[0,256])
 
cdf_b = np.cumsum(hist_b)
cdf_g = np.cumsum(hist_g)
cdf_r = np.cumsum(hist_r)

cdf_b = (cdf_b - cdf_b.min())*255/(cdf_b.max()-cdf_b.min())
cdf_b = np.ma.filled(cdf_b,0).astype('uint8')

cdf_g = (cdf_g - cdf_g.min())*255/(cdf_g.max()-cdf_g.min())
cdf_g = np.ma.filled(cdf_g,0).astype('uint8')

cdf_r = (cdf_r - cdf_r.min())*255/(cdf_r.max()-cdf_r.min())
cdf_r = np.ma.filled(cdf_r,0).astype('uint8')

b2 = cdf_b[b]
g2 = cdf_g[g]
r2 = cdf_r[r]

histImgB_after = calcAndDrawHist(r2, [255,0,0])
cv2.imshow("histImgB_after", histImgB_after)
histImgG_after = calcAndDrawHist(g2, [0,255,0])
cv2.imshow("histImgG_after", histImgG_after)
histImgR_after = calcAndDrawHist(b2, [0,0,255])
cv2.imshow("histImgR_after", histImgR_after)

rgb = cv2.merge((b2,g2,r2))

cv2.imshow("B4",img)
cv2.imshow("after",rgb)

cv2.waitKey(0)
cv2.destroyAllWindows()
