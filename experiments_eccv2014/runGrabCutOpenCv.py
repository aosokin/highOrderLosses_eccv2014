# -*- coding: utf-8 -*-
"""
Created on Tue May 06 22:05:32 2014

@author: AntonOsokin
"""

import numpy as np
import cv2
import os
import sys

def runGrabCutOnTheImage(image, seeds):
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    import copy    
    mask = copy.deepcopy( seeds )
    cv2.grabCut(image, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK )
    result = np.where( (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0)
    return result

def readSeedImage(fileName):
    seedImage = cv2.imread(fileName, 0)
    seedMask = np.zeros(seedImage.shape[:2], np.uint8) + cv2.GC_PR_BGD
    seedMask[seedImage == 255] = cv2.GC_FGD;    
    seedMask[ (seedImage < 255) & (seedImage > 0) ] = cv2.GC_BGD;        
    return seedMask

def getImageList(folderName):
    imageList = []
    for file in os.listdir(folderName):
        if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".bmp"):
            imageList.append(file)
    return imageList    

def saveSegmentationMask(mask, fileName):
    image = np.where( mask == 1, 255, 0).astype('uint8')
    cv2.imwrite(fileName, image) 
    return

# start the script
if len(sys.argv) != 4:
    print 'Argumant error: the three command line argumants have to be supplied!'
    sys.exit(1)

imageDir = sys.argv[1]
seedDir = sys.argv[2]
resultDir = sys.argv[3]

if not os.path.isdir( resultDir ):
    os.mkdir( resultDir )

imageList = getImageList( imageDir )
for imageFile in imageList:
     print 'Working with image ' + imageFile
     imageName = os.path.splitext(imageFile)[0]
     image = cv2.imread( imageDir + '/' + imageFile)
     seeds = readSeedImage( seedDir + '/' + imageName + '-anno.png' )
     result = runGrabCutOnTheImage(image, seeds)   
     saveSegmentationMask( result, resultDir + '/' + imageName + '.png' )    
