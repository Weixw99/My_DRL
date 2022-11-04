import numpy as np
import math  as mh
 
#class Vehicle(object):

#    def __init__(self, x):
def Vehicle(x1,x2,x3,x4,x5,x6,x7,x8,x9,step_num):
#def  Vehicle(x)
#        x=np.array([x],float)
        m1 = 25.8
        m2 = 0
        m3 = 0
        m4 = 0
        m5 = 33.8
        m6 = 0
        m7 = 0
        m8 = 0
        m9= 2.760
        t = step_num
#        x1=  x[0][0]#x[:1]
#        x2 = x[0][1]#x[1:2]
#        x3 = x[0][2]#x[2:3]
#        x4 = x[0][3]#x[3:4]
#        x5 = x[0][4]#x[4:5]
#        x6 = x[0][5]#x[5:6]
#        x7 = x[0][6]#x[6:7]
#        x8 = x[0][7]#x[7:8]
#        x9 = x[0][8]#x[8:9]
#        x1=x1
#        x2=x2
#        x3=x3
#        x4=x4
#        x5=x5
#        x6=x6
#        x7=x7
#        x8=x8
#        x9=x9
        dt = 0.1
       
#    def update(self,ob):
        M = [[m1, m2, m3], [m4, m5, m6], [m7, m8, m9]]
        M = np.array(M, float)
        M_inv = np.linalg.inv(M) ## qiu ni
        ##self.M_inv = np.mat(M).I ## qiu ni
        B = M_inv
        J = np.array([[mh.cos(x3) , -mh.sin(x3), 0], [mh.sin(x3), mh.cos(x3), 0], [0, 0, 1.0]], float)
        C = np.array([[0, 0, -33.8 * x5 - 1.0948 * x6], [0, 0, 25.8 * x4], [33.8 * x5 + 1.0948 * x6, -25.8 * x4,  0]], float)
        D = np.array([[0.7225 + 1.3274 * abs(x4) + 5.8664 * x4 ** 2, 0, 0],[0, 0.88965 + 36.47287 * abs(x5) + 0.805 * abs(x6), 0],[0, 0, 1.90 - 0.08 * abs(x5) + 0.75 * abs(x6)]], float)
################# round(x, 5)
        eta =  np.array([[x1], [x2], [x3]])
        nu  =  np.array([[x4], [x5], [x6]])
        T   =  np.array([[x7], [0], [x9]])
#####################zhuan matrix######
        B = np.mat(B)
        J = np.mat(J)
        C = np.mat(C)
        D = np.mat(D)
        eta = np.mat(eta)
        nu = np.mat(nu)
        T = np.mat(T)
        disturbance =  0.5*np.sin(0.8*t)+0.08*np.cos(0.5*t)
#        disturbance = 0
        eta_dot=[]
        nu_dot=[]
        eta_dot = eta + dt * J * nu
        nu_dot  = nu + dt * B  *(( -D - C ) * nu + T + disturbance)
#        sin_dot = mh.sin(x3)
#        cos_dot = mh.cos(x3)
       
        return eta_dot ,nu_dot #,sin_dot ,cos_dot
        
      


