import ast
import math
#function return output after apply weighting average equation
def weightaverage(beta,theta):#theta is the input array
    out=[]
    v = 0
    for i in range(len(theta)):
        v =( beta * v ) +((1 - beta) * theta[i])
        out.append(v)
    return(out)


#function return array after decreasing number of frames
def average(speed,no_frames):#speed is the input array and no_frames is the desired number of output
    avg = []
    sum=0
    x=len(speed)/no_frames #int of x is the number of frames in the input array which we will take their average and represent them by one frame
    index =0 #variable which represent the index in input array each loop increase by int(x)
    for j in range(no_frames):#outer loop which append value to the output array
        if j==no_frames-1: #the last frame in the output frame may not take the average of x frames but the reminder in the input array
            for i in range(len(speed)-index):#reminder in input array
                sum += speed[i + index]
            avg.append(sum/(len(speed)-index))
        else:
            for i in range(int(x)):#take the sum of int(x) frames
                sum += speed[i+index]
            index += int(x) #increase the index for the next loop
            avg.append(sum/int(x)) #take the average and append them in the output array
        sum=0
    return avg

def speedfun(x1,x2,y1,y2,timePerFrame):#function return speed between each two frames for each joint
    dist=math.sqrt(pow(x1-x2,2)+pow(y1-y2,2))
    return(dist/timePerFrame)
def direction(x1,x2,y1,y2):#function return direction between each two frames for each joint
    f2=math.degrees(math.atan2(y2,x2))
    f1=math.degrees(math.atan2(y1,x1))
    return f2-f1
########################################################################################################################
#function return two dimentional array for speed and direction between each two frames for each joint
def speedanddir(data,timePerFrame):
    speed=[]
    dir=[]
    dirforall=[]
    speedforall=[]
    dir=[]
    for j in range(int(len(data[0]['keypoints'])/3)):#for loop for each joint
        for i in range(int(len(data)-1)):#for loop for each frame
            s=speedfun(data[i]['keypoints'][j*3],data[i+1]['keypoints'][j*3],data[i]['keypoints'][j*3+1],data[i+1]['keypoints'][j*3+1],timePerFrame)#calculate speed from speed function
            dir.append(direction(data[i]['keypoints'][j*3],data[i+1]['keypoints'][j*3],data[i]['keypoints'][j*3+1],data[i+1]['keypoints'][j*3+1]))#calculate diretion from speed function
            speed.append(s)
        speedforall.append(speed)
        speed=[]
        dirforall.append(dir)
        dir=[]
    return speedforall , dirforall
#########################################################################################################################
#function return two dimentional array for waighting speed and direction between each two frames for each joint
def wait_speedanddir(speed,dir,beta):
    for i in range(len(speed)):
        speed[i]=weightaverage(beta,speed[i])
        dir[i]=weightaverage(beta,dir[i])
    return speed,dir
########################################################################################################################
#function return two dimentional array for average speed and direction between each two frames for each joint
def avg_speedanddir(speed,dir,no_frames):
    for i in range(len(speed)):
        speed[i]=average(speed[i],no_frames)
        dir[i]=average(dir[i],no_frames)
    return speed,dir
def main():
    beta=0.9#input from user
    timePerFrame=0.01#gets from number of frames per second
    no_frames=20#number of needs frames --input from user--
    speed=[]#two dimentional array [speed between every two frames][for each joint]
    dir=[]#two dimentional array [direcion between every two frames][for each joint]
    with open('C://Users//future//PycharmProjects//pro1test//out.json') as f:#path of input file
        data = ast.literal_eval(f.read())#store data in list of dictionaries called data
    speed,dir=speedanddir(data,timePerFrame)#get speed and direction between each two frames for each joint
    speed,dir=avg_speedanddir(speed,dir,no_frames)# get average for the previous speed and direction related to spesific number of frames
    speed,dir=wait_speedanddir(speed,dir,beta)# get weighting average for the previous speed and frame
    print(speed)
    print(dir)
if __name__=='__main__':
    main()