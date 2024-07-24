#%%
import numpy as np
import time
import matplotlib.pyplot as plt
from PIL import Image
import cv2 as cv
import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame, Series  # for convenience
import time
import trackpy as tp
import math
from scipy.ndimage import uniform_filter1d
from PIL import Image
import keras
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from PIL import Image
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, BatchNormalization
from sklearn.metrics import confusion_matrix, classification_report
import pathlib
from tensorflow.keras.optimizers import Adam
import ipyplot



def roi(frame): #get roi, this ins't being used at the moment
    r1 = cv.selectROI(frame) # top
    r2 = cv.selectROI(frame) # channel
    cv.destroyAllWindows()
    return r1, r2

def bleed(frame, BL):
    L430430=(BL[0])
    L410410=(BL[1])
    L630B=(BL[2])
    L430630=(BL[3])
    L410630=(BL[4])
    L630630=(BL[5])
    blr=(BL[6])
    B430= frame[0]-blr[0,:,:]
    B410= frame[1]-blr[0,:,:]
    R430= frame[2]-blr[1,:,:]
    R410= frame[3]-blr[1,:,:]

    B430=B430*L430430-B430*L630B
    B410=B410*L410410-R410*L630B
    R430=R430*L630630-B430*L430630
    R410=R410*L630630-B410*L410630

    frameBL=[B430,B410,R430,R410]
    return frameBL

def euler(f, thresh, x_old, y_old): #This function is used to check if a cell found in a frame is close to the cell in the previous frame, to avoid getting different cells
    pt=[0,0,0] 
    if len(f)==1:
        pt1=[f.x.values[0],f.y.values[0], f.signal.values[0]]
    elif len(f)==2:
        pt1=[f.x.values[0],f.y.values[0], f.signal.values[0]]
        pt2=[f.x.values[1],f.y.values[1], f.signal.values[1]]
        if pt1[2]>thresh and pt2[2]>thresh:
            dist1 = math.hypot(x_old - pt1[0], y_old - pt1[1])
            dist2 = math.hypot(x_old - pt2[0], y_old - pt2[1])
            if dist1<dist2:
                pt=pt1
            elif dist1>dist2:
                pt=pt2  
        elif pt1[2]>thresh and pt2[2]<thresh:
            pt=pt1
        elif pt1[2]<thresh and pt2[2]>thresh:
            pt=pt2
    # elif len(f)==3:
    #     point1=[f.x.values[0],f.y.values[0], f.signal.values[0]]
    #     point2=[f.x.values[1],f.y.values[1], f.signal.values[1]]
    #     point3=[f.x.values[2],f.y.values[2], f.signal.values[2]]
    #     input_list=[point1,point2,point3]
    #     output_list = sorted(input_list, key = lambda x: x[2])
    #     pt1=output_list[0]
    #     pt2=output_list[1]
    #     pt3=output_list[2]
    #     points=[pt1,pt2,pt3]
    #     if pt1[2]>thresh and pt2[2]>thresh and pt3[2]>thresh:
    #         dist1 = math.hypot(x_old - pt1[0], y_old - pt1[1])
    #         dist2 = math.hypot(x_old - pt2[0], y_old - pt2[1])
    #         dist3 = math.hypot(x_old - pt3[0], y_old - pt3[1])
    #         distances=[dist1,dist2,dist3]
    #         idx_sort = distances.argsort()
    #         pt=points[idx_sort[::-1]][0]
        if pt1[2]>thresh and pt2[2]>thresh:
            dist1 = math.hypot(x_old - pt1[0], y_old - pt1[1])
            dist2 = math.hypot(x_old - pt2[0], y_old - pt2[1])
            if dist1<dist2:
                pt=pt1
            elif dist1>dist2:
                pt=pt2 
        elif pt1[2]>thresh and pt2[2]<thresh:
            pt=pt1
        elif pt1[2]<thresh and pt2[2]>thresh:
            pt=pt2
    else:
        pt=[0,0,0]   
    return pt 

#normalize images
def rescale(img):
    norm_img = cv.normalize(img, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
    return norm_img
#noralize, invert, and mask blue images
def rescale_blue(img, mask):
    masked = (img)*mask
    masked=1-masked
    masked=rescale(masked)
    masked = (masked)*mask
    return masked

#nomalize and mask red images
def rescale_red(img, mask):
    masked = (img)*mask
    masked=rescale(masked)
    masked = (masked)*mask
    return masked

#calculate Hb mass
# def mass(b,pt):
#     x=pt[0]; y=pt[1]
#     parea=6.9
#     Hb=(b[int(y-10):int(y+10), int(x-10):int(x+10)])    
#     cell_mask = np.zeros_like(Hb)
#     cv.circle(cell_mask,(10,10), 6, (1,1,1), -1)
#     cell=Hb*cell_mask
#     masked=Hb
#     cv.circle(masked,(10,10), 6, (255,255,255), -1)
#     average = masked[np.nonzero(masked)].mean()
#     Hbnorm=(cell/average)
#     Hbnorm[Hbnorm <= 0] = 1
#     hbmass=((parea*(10**-8)*64500*np.sum(np.sum((-np.log10(Hbnorm))))))
#     return hbmass

def mass(b,pt):
    x=pt[0]; y=pt[1]
    parea=6.9
    Hb=(b[int(y-10):int(y+10), int(x-10):int(x+10)])
    base=Hb[10,10]*2
    level, cell_mask = cv.threshold(Hb, base, 255, cv.THRESH_BINARY)
    _, wall_mask = cv.threshold(Hb, 1.8*level, 255, cv.THRESH_BINARY)
    kernel = np.ones((5,5),np.uint8)
    dilation = cv.erode(cell_mask,kernel,iterations = 1)
    kernel_outer = np.ones((5,5),np.uint8)
    dilation_outer = cv.erode(cell_mask,kernel_outer,iterations = 1)
    cell_mask=cell_mask/255
    dilation=dilation/255
    masked = (Hb*dilation)
    Hb=Hb*(1-dilation_outer)
    average = masked[np.nonzero(masked)].mean()
    Hbnorm=Hb/average
    Hbnorm[Hbnorm <= 0] = 1
    hbmass=((parea*(10**-8)*64500*np.sum(np.sum((-np.log10(Hbnorm))))))
    return hbmass

#create a mask for the cell and normalize and mask the images
def masking(img, pt):
    x=pt[0]; y=pt[1]  
    img=(img[int(y-25):int(y+25), int(x-25):int(x+25)]).astype('uint8')
    cell_mask = np.zeros_like(img)
    masked = np.zeros_like(img)
    ret, thresh = cv.threshold(img, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU) 
    contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
            # Calculate area and remove small elements
            area = cv.contourArea(cnt)  
            # print(area)
            if area > 100 and area < 2000:
                cell_mask = np.zeros_like(img)
                masked = np.zeros_like(img)
                M = cv.moments(cnt)
                if M['m00'] != 0:
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                hull = cv.convexHull(cnt)
                cv.drawContours(cell_mask, [hull], -1,(1), -1)
                img=rescale(img)
                cell=(img[int(cy-20):int(cy+21), int(cx-20):int(cx+21)])
                cell_mask=(cell_mask[int(cy-20):int(cy+21), int(cx-20):int(cx+21)])
                masked=rescale_blue(cell,cell_mask)
    if masked.shape!=(41,41) or cell_mask.shape!=(41,41):
            masked=np.zeros((41,41))
            cell_mask=np.zeros((41,41))
    return masked, cell_mask

#calulcate the volume on the cell
def volume(img, pt, mask):
    channel=8
    x=pt[0]; y=pt[1]
    cell=(img[int(y-20):int(y+21), int(x-20):int(x+21)])
    if len(mask[0])<41:
        vol=np.nan
    else:
        kernel = np.ones((5,5),np.uint8)
        dilation = cv.dilate(mask,kernel,iterations = 1)
        kernel_outer = np.ones((5,5),np.uint8)
        dilation_outer = cv.dilate(mask,kernel_outer,iterations = 1)
        # print(cell.shape)
        # print(dilation.shape)
        masked = (cell*(mask))
        sur=np.mean(masked)
        top=np.mean(img[int(0):int(10), int(0):int(720)])
        parea=(6.9/20)**2
        nimg=cell*(dilation)/sur
        nimg[nimg <= 0] = 0.01
        transmission = sur/top
        alf = -np.log(transmission)/channel
        tcell=np.log(nimg)/alf
        vol=np.sum(tcell*mask)*parea
        # plt.imshow(mask)
    return vol
   
def saturation(frame, pt430, pt410): #Calcuate cell saturation
     #camera pixel area
    b=[frame[0],frame[1]]
    #Molecular absorbtion coefficints of something like that ~chemistry~
    w430_o = 2.1486*(10**8)
    w430_d = 5.2448*(10**8)
    w410_o = 4.6723*(10**8)
    w410_d = 3.1558*(10**8)

    mass410=mass(b[0], pt430)
    mass430=mass(b[1], pt410)
    
    e=mass410 #410
    f=mass430 #430
    #Set absorbtion values to equation constants
    a=w410_d
    b=w410_o
    c=w430_d
    d=w430_o
                
    #Calcuate mass of oxygenated and deoxygenated hemoglobin
    Mo=(a*f-e*c)/(a*d-b*c)
    Md=(e*d-b*f)/(a*d-b*c)

    saturation = Mo/(Mo+Md)
    hbmass=e+f
    # print(saturation)
    return saturation, hbmass

#resize the images, and process them for the neural net
def net(frame, pt430, pt410):
    B430= cv.resize(frame[0], dsize=(720, 540), interpolation=cv.INTER_LINEAR).astype("uint8")
    B410= cv.resize(frame[1], dsize=(720, 540), interpolation=cv.INTER_LINEAR).astype("uint8")
    R430= cv.resize(frame[2], dsize=(720, 540), interpolation=cv.INTER_LINEAR).astype("uint8")
    R410= cv.resize(frame[3], dsize=(720, 540), interpolation=cv.INTER_LINEAR).astype("uint8")
    pt430= np.asarray(pt430, dtype=np.float32)
    pt410= np.asarray(pt410, dtype=np.float32)
    ptres430=2*pt430; ptres410=2*pt410
    
    imgB430, mask430=masking(B430, ptres430)
    imgB410, mask410=masking(B410, ptres410)
    imgR430=(R430[int(ptres430[1]-20):int(ptres430[1]+21), int(ptres430[0]-20):int(ptres430[0]+21)])
    imgR410=(R410[int(ptres410[1]-20):int(ptres410[1]+21), int(ptres410[0]-20):int(ptres410[0]+21)])
    mask430[mask430<0] = 0
    mask410[mask410<0] = 0
    imgB410[imgB410<0] = 0
    imgB430[imgB430<0] = 0
    imgR410[imgR410<0] = 0
    imgR430[imgR430<0] = 0
    vol430=volume(R430, ptres430, mask430)
    vol410=volume(R410, ptres410, mask410)

    imgR410=rescale(imgR410)
    imgR410=rescale_red(imgR410,mask410)
    imgR430=rescale(imgR430)
    imgR430=rescale_red(imgR430,mask430)
    imgR=imgR430/2+imgR410/2

    imgB430=cv.resize(imgB430, dsize=(81, 81), interpolation=cv.INTER_LINEAR)
    imgB410=cv.resize(imgB410, dsize=(81, 81), interpolation=cv.INTER_LINEAR)
    imgR=cv.resize(imgR, dsize=(81, 81), interpolation=cv.INTER_LINEAR)
    img = np.zeros([81,81,3])
    img[:,:,0] = imgB430#cv.resize(imgB430, dsize=(81, 81), interpolation=cv.INTER_LINEAR).astype("uint8")
    img[:,:,1] = imgB410#cv.resize(imgB410, dsize=(81, 81), interpolation=cv.INTER_LINEAR).astype("uint8")
    img[:,:,2] = imgR#cv.resize(imgR, dsize=(81, 81), interpolation=cv.INTER_LINEAR).astype("uint8")
    
    vol=(vol430+vol410)/4
    return img, vol

# Seperate the frames and find the cell in each frame
def segment(frames,x_old, y_old,BL):
    thresh410=40
    thresh430=60
    img=frames
    if x_old<1 or y_old<1:   
        x_old=180; y_old=135
    L1=img[0][1::2, 1::2]
    L2=img[1][1::2, 1::2]
    R1=img[0][::2, ::2]
    R2=img[1][::2, ::2]
    if np.mean(L1)>np.mean(L2):
        b=[L1,L2]
        frame=[L1,L2,R1,R2]
    else:
        b=[L2,L1]
        frame=[L2,L1,R2,R1]
    f430 = tp.locate(b[0], 21, preprocess=True, percentile=99,  invert=True, max_iterations=1, characterize =True, topn=2)
    f410 = tp.locate(b[1], 21, preprocess=True, percentile=99,  invert=True, max_iterations=1, characterize =True, topn=2)
    # print(f410.head())
    pt430=euler(f430,thresh430, x_old, y_old)
    pt410=euler(f410,thresh410, x_old, y_old)
    frame=bleed(frame,BL)

    return pt430,pt410, frame



def getBL(): #Load the BL file
    with h5py.File('BL.h5', 'r') as BL:
        L430430=(BL['L430430'][:])
        L410410=(BL['L410410'][:])
        L630B=(BL['L630B'][:])
        L430630=(BL['L430630'][:])
        L410630=(BL['L410630'][:])
        L630630=(BL['L630630'][:])
        blr=(BL['blr'][:])
    BL=[L430430,L410410,L630B,L430630,L410630,L630630,blr]
    return BL

def main_run(video): #Run the anaylsis in the video
    BL=getBL()
    i=0
    x_old=0
    y_old=0
    saturations=[]
    imgs=[]
    volumes=[]
    hgb=[]
    x=[0,0,0]
    y=[0,0,0]
    while i<len(video):
        frame = video[i]
        x_old=np.mean(x[-3:])
        y_old=np.mean(y[-3:])
        pt430, pt410, frame = segment(frame,x_old, y_old,BL)
        # print(pt430)
        # print(pt410)
        if pt430[0]>11 and pt410[0]>11 and pt430[0]<349 and pt410[0]<349 and pt430[1]>11 and pt410[1]>11 and pt430[1]<239 and pt410[1]<239:
            sats, hbmass=saturation(frame,pt430,pt410)
            saturations.append(sats)
            hgb.append(hbmass)
            img, vol=net(frame,pt430, pt410)
            imgs.append(img)
            volumes.append(vol)
            x.append(pt410[0])
            y.append(pt410[1])
        else:
            saturations.append(np.nan)
            volumes.append(np.nan)
            hgb.append(np.nan)
            imgs.append(np.zeros([81,81,3]))
            x.append(0)
        i=i+1
    return imgs, saturations, volumes, x, hgb

def veiw(video):
    BL=getBL()
    i=0
    x_old=0
    y_old=0
    x=[0,0,0]
    y=[0,0,0]
    while i<len(video):
        frame = video[i]#np.maximum(data2[i][0],data2[i][1])
        x_old=np.mean(x[-3:])
        y_old=np.mean(y[-3:])
        pt430, pt410, frame = segment(frame,x_old, y_old,BL)
        gray=cv.resize(frame[0], dsize=(720, 540), interpolation=cv.INTER_CUBIC).astype("uint8")
        # gray = cv.cvtColor(frame[0], cv.COLOR_BayerBG2GRAY)
        if pt430[0]>21 and pt410[0]>21 and pt430[0]<349 and pt410[0]<349 and pt430[1]>11 and pt410[1]>11 and pt430[1]<239 and pt410[1]<239:
            sat=saturation(frame,pt430,pt410)
            pt430= np.asarray(pt430, dtype=np.float32)
            pt410= np.asarray(pt410, dtype=np.float32)
            ptres430=2*pt430; ptres410=2*pt410
            x.append(pt410[0]); y.append(pt410[1])
            locx=int(ptres410[0]); locy=int(ptres410[1])
            # img = gray[y-40:y+40,x-40:x+40]
            cv.rectangle(gray, (locx-20, locy-20), (locx+20, locy+20), (255,100,200),2)
            # img=cv.resize(img, dsize=(200, 200), interpolation=cv.INTER_CUBIC).astype("uint8")
            cv.putText(gray, str('%.2f' %sat), (locx-45,locy-45), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
        cv.imshow('Single Track', gray)
            # # # press 'q' to break loop and close window
        cv.waitKey(5)
        i=i+1
    # print(xf)
    cv.destroyAllWindows()

def CNN(imgs): #Run the CNN and display the resutls
    model=keras.models.load_model('CNN/model_ResNet50_A01.h5', safe_mode=False) #load the model
    cnn_imgs=np.stack( imgs, axis=0 )
    predictions = model.predict(cnn_imgs) #this is the line that calls the CNN to classify
    print(predictions[0:10])

    threshold=0.5 #threshold is set at 0.5 to start
    labels=[]
    preds=[]
    cell_imgs=[]
    for i in range(len(predictions)):
        preds.append(predictions[i][0])
        cell_imgs.append(cnn_imgs[i][:,:,1])
        if predictions[i][0] > threshold:
            labels.append(0)
        else:
            labels.append(1)
    preds=(predictions[:,0]) #get the first index of the predicitons, I believe this is what we've been using
    preds=np.round(preds, 3)
    pred=np.array(preds) 
    sorted = pred.argsort() #sort the predicitons by descending order
    sorted_preds = pred[sorted[::-1]]
    sorted_cells = cnn_imgs[sorted[::-1]]
    ipyplot.plot_images(sorted_cells, sorted_preds,max_images=500, img_width=50)
    return preds

def write(video, name):
    BL=getBL()
    i=0
    x_old=0
    y_old=0
    x=[0,0,0]
    y=[0,0,0]
    vid = cv.VideoWriter(name, cv.VideoWriter_fourcc('M','J','P','G'), 20, (720, 540), False)
    while i<len(video):
        frame = video[i]
        x_old=np.mean(x[-3:])
        y_old=np.mean(y[-3:])
        pt430, pt410, frame = segment(frame,x_old, y_old,BL)
        gray=cv.resize(frame[0], dsize=(720, 540), interpolation=cv.INTER_CUBIC).astype("uint8")
        # gray = cv.cvtColor(frame[0], cv.COLOR_BayerBG2GRAY)
        if pt430[0]>21 and pt410[0]>21 and pt430[0]<349 and pt410[0]<349 and pt430[1]>11 and pt410[1]>11 and pt430[1]<239 and pt410[1]<239:
            sat=saturation(frame,pt430,pt410)
            pt430= np.asarray(pt430, dtype=np.float32)
            pt410= np.asarray(pt410, dtype=np.float32)
            ptres430=2*pt430; ptres410=2*pt410
            x.append(pt410[0]); y.append(pt410[1])
            locx=int(ptres410[0]); locy=int(ptres410[1])
            # img = gray[y-40:y+40,x-40:x+40]
            cv.rectangle(gray, (locx-20, locy-20), (locx+20, locy+20), (255,100,200),2)
            # img=cv.resize(img, dsize=(200, 200), interpolation=cv.INTER_CUBIC).astype("uint8")
            cv.putText(gray, str('%.2f' %sat), (locx-45,locy-45), cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)
        vid.write(gray)
        i=i+1
    # print(xf)
    cv.destroyAllWindows()
    vid.release()
    
    
