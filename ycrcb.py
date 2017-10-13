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
img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
y,cr,cb = cv2.split(img_ycrcb)


histImg = calcAndDrawHist(img, [255,255,255])
cv2.imshow("histImg-original", histImg)


hist,bins = np.histogram(y.flatten(),256,[0,256])
 
cdf = np.cumsum(hist)

cdf = (cdf - cdf.min())*255/(cdf.max()-cdf.min())
cdf = np.ma.filled(cdf,0).astype('uint8')

ycrcb = cdf[y]

histImg_after = calcAndDrawHist(ycrcb, [255,255,255])
cv2.imshow("histImg_after", histImg_after)

cv2.imshow("B4",img)
cv2.imshow("after",ycrcb)

cv2.waitKey(0)
cv2.destroyAllWindows()
