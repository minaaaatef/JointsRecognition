import numpy as np
import ast
import math
import matplotlib.pyplot as plt





def clcangle(arr1,arr2):#############function return array of angles between points of two frames
    x1=np.array(np.reshape(arr1,(int(len(arr1)/3),3)))#reshape the first __KeyPoints__array
    x1=np.delete(x1,2,1)#delete the C colomn from first array
    x1=np.delete(x1,1,1)#delete the Y colomn from first array


    y1=np.array(np.reshape(arr1,(int(len(arr1)/3),3)))#reshape the first __KeyPoints__array
    y1=np.delete(y1,2,1)#delete the C colomn from first array
    y1=np.delete(y1,0,1)#delete the X colomn from first array


    x2=np.array(np.reshape(arr2,(int(len(arr2)/3),3)))#reshape the second __KeyPoints__array
    x2=np.delete(x2,2,1)#delete the C colomn from second array
    x2=np.delete(x2,1,1)#delete the Y colomn from second array

    y2=np.array(np.reshape(arr2,(int(len(arr2)/3),3)))#reshape the second __KeyPoints__array
    y2=np.delete(y2,2,1)#delete the C colomn from second array
    y2=np.delete(y2,0,1)#delete the X colomn from second array

    ang1=np.array(np.arctan2(y1,x1))*180/np.pi#calculate anggle between points of the first array and the horizontal line
    ang2=np.array(np.arctan2(y2,x2))*180/np.pi#calculate anggle between points of the second array and the horizontal line
    return np.transpose(ang1-ang2)


def clcspeed(arr1,arr2,time):#function return speed of all joints between two frames
    arr1=np.array(np.reshape(arr1,(int(len(arr1)/3),3)))#reshape the first __KeyPoints__array
    arr1=np.delete(arr1,2,1)#delete the C colomn from first array
    arr2=np.array(np.reshape(arr2,(int(len(arr2)/3),3)))#reshape the first __KeyPoints__array
    arr2=np.delete(arr2,2,1)#delete the C colomn from first array
    dist=(np.array(arr1)-np.array(arr2))**2
    dist=np.sum(dist,axis=1)
    dist=np.sqrt(dist)
    return dist/time


def average(input,no_frames):
    row, column = input.shape
    out = np.zeros((no_frames,column)) #initialization of output matrix
    for i in range(int(no_frames)): #fill the out array with rows equal desired no_frames by takig average of some frames to be one frame
        out[i] = np.mean(input[i*int(row/no_frames):i*int(row/no_frames)+int(row/no_frames)], axis=0,dtype=np.float64) #take the average of each int(row/no_frames) values
    return (out)


def weightaverage(beta,input):
    row,column=input.shape
    v = np.zeros((1,column)) #initialization of v which is the value we depend on it for each loop
    out =np.zeros((input.shape))  #initialization of out array
    for i in range(row):
        v =(( beta * v ) +((1 - beta) * input[i])) #weighting average equation
        out[i]=(v) #put each row in the out matrix
    return(out)


def InputProccessing(beta , path ,no_frames ): #beta for wightes average , path for jason file , no_frames for input frames

    with open(path) as f: #path of input file
        data = ast.literal_eval(f.read())#store data in list of dictionaries called data

    speed=np.matrix(clcspeed(data[0]['keypoints'],data[1]['keypoints'],0.01))#clc speed between the first two frames
    angle=np.matrix(clcangle(data[0]['keypoints'],data[1]['keypoints']))#clc direction between the first two frames

    for i in range(1,int(len(data)-1)):#for loop for all dictionaries in the list
        res2=np.matrix(clcspeed(data[i]['keypoints'],data[i+1]['keypoints'],0.01))##clc speed between each two frames
        speed=np.concatenate((speed,res2), axis=0)#concatenate the new array of speeds to the matrix
        angle2=np.matrix(clcangle(data[i]['keypoints'],data[i+1]['keypoints']))#clc direction between each two frames
        angle=np.concatenate((angle,angle2), axis=0)#concatenate the new array of directions to the matrix
    #print("speed",speed.shape)
    #print("dir",angle.shape)
    # speed=np.matrix(average(speed,no_frames))
    # angle=np.matrix(average(angle,no_frames))
    #print("avg_speed",speed.shape)
    #print("avg_dir",angle.shape)

    speed=np.matrix(weightaverage(beta,speed))
    angle=np.matrix(weightaverage(beta,angle))
    out = np.concatenate((speed, angle), axis=1)
    out = np.array(out.reshape((1, out.shape[0], out.shape[1])))






    return out.reshape((1,out.shape[0],out.shape[1]))
