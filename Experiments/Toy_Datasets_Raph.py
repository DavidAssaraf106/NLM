import autograd.numpy as np
import autograd.numpy as np
from sklearn import cluster, datasets
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')



def create_two_circular_classes(n=1500, noise_input=0.05, plot=False):
    """
    INPUT:
    n is the target number of points in each circle (note that this is a target)
    noise is the noise used in sklearn make_circles function
    plot: if TRUE, will return a plot of the two classes of points (in red and yellow)
    as well as the boundary class (in blue)
    
    OUPUT: boundary, class1, class2
    the two classes of points
    the boundary class
    
    """
    # Generating the data using sklearn built-in function
    noisy_circles_1 = datasets.make_circles(n_samples=n, factor=.0, noise=noise_input)
    noisy_circles_2 = datasets.make_circles(n_samples=n, factor=.3, noise=noise_input)
    noisy_circles_3 = datasets.make_circles(n_samples=n, factor=.5, noise=noise_input)
    noisy_circles_4 = datasets.make_circles(n_samples=n, factor=.7, noise=noise_input)

    X_1 = []
    Y_1 = []
    for i in range(len(noisy_circles_1[0])):
        # make_circles creates two circles, we only want to create one
        if (noisy_circles_1[0][i][0]) ** 2 + noisy_circles_1[0][i][1] ** 2 < .7:
            X_1.append(noisy_circles_1[0][i][0])
            Y_1.append(noisy_circles_1[0][i][1])

    X_2 = []
    Y_2 = []
    for i in range(len(noisy_circles_2[0])):
        # make_circles creates two circles, we only want to create one
        if (noisy_circles_2[0][i][0]) ** 2 + noisy_circles_2[0][i][1] ** 2 < .7:
            X_2.append(noisy_circles_2[0][i][0])
            Y_2.append(noisy_circles_2[0][i][1])
    X_3 = []
    Y_3 = []
    for i in range(len(noisy_circles_3[0])):
        # make_circles creates two circles, we only want to create one
        if (noisy_circles_3[0][i][0]) ** 2 + noisy_circles_3[0][i][1] ** 2 < .7:
            X_3.append(noisy_circles_3[0][i][0])
            Y_3.append(noisy_circles_3[0][i][1])

    X_4 = []
    Y_4 = []
    for i in range(len(noisy_circles_4[0])):
        # make_circles creates two circles, we only want to create one
        if (noisy_circles_4[0][i][0]) ** 2 + noisy_circles_4[0][i][1] ** 2 < .7:
            X_4.append(noisy_circles_4[0][i][0])
            Y_4.append(noisy_circles_4[0][i][1])

    if plot:
        plt.plot(X_1, Y_1, 'x', c='r')
        plt.plot(X_2, Y_2, 'x', c='b')
        plt.plot(X_3, Y_3, 'x', c='y')
        plt.plot(X_4, Y_4, 'x', c='b')
        plt.show()

    boundary = [X_2 + X_4, Y_2 + Y_4]
    class1 = [X_1, Y_1]
    class2 = [X_3, Y_3]

    return boundary, class1, class2



def create_three_circular_classes(k=2, n=5000, noise_input=0.02, plot=False):
    """
    INPUT:
    n/2 is the target number of points in each circle (note that this is a target)
    noise is the noise used in sklearn make_circles function
    plot: if TRUE, will return a plot of the two classes of points (in red, yellow, green)
    as well as the boundary class (in blue)
    
    OUPUT: boundary, class1, class2, class3
    the three classes of points
    the boundary class
    
    """
    # Generating the data using sklearn built-in function
    noisy_circles_1 = datasets.make_circles(n_samples=n, factor=.0, noise=noise_input)
    noisy_circles_2 = datasets.make_circles(n_samples=n, factor=.3, noise=noise_input)
    noisy_circles_3 = datasets.make_circles(n_samples=n, factor=.5, noise=noise_input)
    noisy_circles_4 = datasets.make_circles(n_samples=n, factor=.7, noise=noise_input)
    
    
    X_1=[]
    Y_1=[]
    for i in range(len(noisy_circles_1[0])):
        # make_circles creates two circles, we only want to create one
        if (noisy_circles_1[0][i][0])**2+noisy_circles_1[0][i][1]**2<.7:
            X_1.append(noisy_circles_1[0][i][0])
            Y_1.append(noisy_circles_1[0][i][1])
            
    X_2=[]
    Y_2=[]
    for i in range(len(noisy_circles_2[0])):
        # make_circles creates two circles, we only want to create one
        if (noisy_circles_2[0][i][0])**2+noisy_circles_2[0][i][1]**2<.7:
            X_2.append(noisy_circles_2[0][i][0])
            Y_2.append(noisy_circles_2[0][i][1])
    X_3=[]
    Y_3=[]
    for i in range(len(noisy_circles_3[0])):
        # make_circles creates two circles, we only want to create one
        if (noisy_circles_3[0][i][0])**2+noisy_circles_3[0][i][1]**2<.7:
            X_3.append(noisy_circles_3[0][i][0])
            Y_3.append(noisy_circles_3[0][i][1])
            
    X_4=[]
    Y_4=[]
    for i in range(len(noisy_circles_4[0])):
        # make_circles creates two circles, we only want to create one
        if (noisy_circles_4[0][i][0])**2+noisy_circles_4[0][i][1]**2<.7:
            X_4.append(noisy_circles_4[0][i][0])
            Y_4.append(noisy_circles_4[0][i][1])
                
    X_5=[]
    Y_5=[]
    for i in range(len(X_3)):
        X_5.append(X_3[i]*1.9)
        Y_5.append(Y_3[i]*1.9)
        
    X_6=[]
    Y_6=[]
    for i in range(len(X_3)):
        X_6.append(X_3[i]*2.5)
        Y_6.append(Y_3[i]*2.5)
            
    if plot:
        plt.plot(X_1,Y_1,'x',c='r')
        plt.plot(X_2,Y_2,'x',c='b')
        plt.plot(X_3,Y_3,'x',c='y')
        plt.plot(X_4,Y_4,'x',c='b')
        plt.plot(X_5,Y_5,'x',c='g')
        plt.plot(X_6,Y_6,'x',c='b')
        plt.show()
        
    boundary=[X_2+X_4+X_6,Y_2+Y_4+Y_6]
    class1=[X_1,Y_1]
    class2=[X_3,Y_3]
    class3=[X_5,Y_5]
    
    return boundary, class1, class2, class3



def create_two_classes(n=1500, noise_input=0.05, plot=False, factor_1=0.3, position=[1,1]):
    """
    INPUT:
    n/2 is the target number of points in each circle (note that this is a target)
    noise is the noise used in sklearn make_circles function
    plot: if TRUE, will return a plot of the two classes of points (in red and yellow)
    as well as the boundary class (in blue)
    factor_1: allows us to vary the average distance between the boundary class and the original two classes
    position: can specify the position of one cluster (the other one is a the origin)
    
    OUPUT: boundary, class1, class2
    the two classes of points
    the boundary class
    
    """
    # Generating the data using sklearn built-in function
    noisy_circles_1 = datasets.make_circles(n_samples=n, factor=.0, noise=noise_input)
    noisy_circles_2 = datasets.make_circles(n_samples=n, factor=factor_1, noise=noise_input)

    
    X_1=[]
    Y_1=[]
    for i in range(len(noisy_circles_1[0])):
        # make_circles creates two circles, we only want to create one
        if (noisy_circles_1[0][i][0])**2+noisy_circles_1[0][i][1]**2<.7:
            X_1.append(noisy_circles_1[0][i][0])
            Y_1.append(noisy_circles_1[0][i][1])
            
    X_2=[]
    Y_2=[]
    for i in range(len(noisy_circles_2[0])):
        # make_circles creates two circles, we only want to create one
        if (noisy_circles_2[0][i][0])**2+noisy_circles_2[0][i][1]**2<.7:
            X_2.append(noisy_circles_2[0][i][0])
            Y_2.append(noisy_circles_2[0][i][1])
    X_3=[]
    Y_3=[]
    for i in range(len(X_1)):
        # make_circles creates two circles, we only want to create one
        X_3.append(X_1[i]+position[0])
        Y_3.append(Y_1[i]+position[1])
            
    X_4=[]
    Y_4=[]
    for i in range(len(X_2)):
        X_4.append(X_2[i]+position[0])
        Y_4.append(Y_2[i]+position[1])
   
            
    if plot:
        plt.plot(X_1,Y_1,'x',c='r')
        plt.plot(X_2,Y_2,'x',c='b')
        plt.plot(X_3,Y_3,'x',c='y')
        plt.plot(X_4,Y_4,'x',c='b')
        plt.show()
        
    boundary=[X_2+X_4,Y_2+Y_4]
    class1=[X_1,Y_1]
    class2=[X_3,Y_3]
    
    return boundary, class1, class2


def create_three_classes(n=1500, noise_input=0.05, plot=False, factor_1=0.3, position=[[1,1],[-1,-1]]):
    """
    INPUT:
    n/2 is the target number of points in each circle (note that this is a target)
    noise is the noise used in sklearn make_circles function
    plot: if TRUE, will return a plot of the two classes of points (in red, yellow, green)
    as well as the boundary class (in blue)
    position: can specify the position of two clusters (the other one is a the origin)
    
    OUPUT: boundary, class1, class2, class3
    the three classes of points
    the boundary class
    
    """
    
    # Generating the data using sklearn built-in function
    noisy_circles_1 = datasets.make_circles(n_samples=n, factor=.0, noise=noise_input)
    noisy_circles_2 = datasets.make_circles(n_samples=n, factor=factor_1, noise=noise_input)

    
    X_1=[]
    Y_1=[]
    for i in range(len(noisy_circles_1[0])):
        # make_circles creates two circles, we only want to create one
        if (noisy_circles_1[0][i][0])**2+noisy_circles_1[0][i][1]**2<.7:
            X_1.append(noisy_circles_1[0][i][0])
            Y_1.append(noisy_circles_1[0][i][1])
            
    X_2=[]
    Y_2=[]
    for i in range(len(noisy_circles_2[0])):
        # make_circles creates two circles, we only want to create one
        if (noisy_circles_2[0][i][0])**2+noisy_circles_2[0][i][1]**2<.7:
            X_2.append(noisy_circles_2[0][i][0])
            Y_2.append(noisy_circles_2[0][i][1])
    X_3=[]
    Y_3=[]
    for i in range(len(X_1)):
        # make_circles creates two circles, we only want to create one
        X_3.append(X_1[i]+position[0][0])
        Y_3.append(Y_1[i]+position[0][1])
            
    X_4=[]
    Y_4=[]
    for i in range(len(X_2)):
        X_4.append(X_2[i]+position[0][0])
        Y_4.append(Y_2[i]+position[0][1])
        
    X_5=[]
    Y_5=[]
    for i in range(len(X_1)):
        # make_circles creates two circles, we only want to create one
        X_5.append(X_1[i]+position[1][0])
        Y_5.append(Y_1[i]+position[1][1])
            
    X_6=[]
    Y_6=[]
    for i in range(len(X_2)):
        X_6.append(X_2[i]+position[1][0])
        Y_6.append(Y_2[i]++position[1][1])
   
            
    if plot:
        plt.plot(X_1,Y_1,'x',c='r')
        plt.plot(X_2,Y_2,'x',c='b')
        plt.plot(X_3,Y_3,'x',c='y')
        plt.plot(X_4,Y_4,'x',c='b')
        plt.plot(X_5,Y_5,'x',c='g')
        plt.plot(X_6,Y_6,'x',c='b')
        plt.show()
        
    boundary=[X_2+X_4+X_6,Y_2+Y_4+Y_6]
    class1=[X_1,Y_1]
    class2=[X_3,Y_3]
    class3=[X_5,Y_5]
    
    return boundary, class1, class2, class3


def create_four_classes(n=1500, noise_input=0.05, plot=False, factor_1=0.3, position=[[1,1],[-1,-1],[2,2]]):
    """
    INPUT:
    n/2 is the target number of points in each circle (note that this is a target)
    noise is the noise used in sklearn make_circles function
    plot: if TRUE, will return a plot of the two classes of points (in red, yellow, green, black)
    as well as the boundary class (in blue)
    position: can specify the position of three clusters (the other one is a the origin)
    
    OUPUT: boundary, class1, class2, class3, class4
    the four classes of points
    the boundary class
    
    """
    
    # Generating the data using sklearn built-in function
    noisy_circles_1 = datasets.make_circles(n_samples=n, factor=.0, noise=noise_input)
    noisy_circles_2 = datasets.make_circles(n_samples=n, factor=factor_1, noise=noise_input)

    
    X_1=[]
    Y_1=[]
    for i in range(len(noisy_circles_1[0])):
        # make_circles creates two circles, we only want to create one
        if (noisy_circles_1[0][i][0])**2+noisy_circles_1[0][i][1]**2<.7:
            X_1.append(noisy_circles_1[0][i][0])
            Y_1.append(noisy_circles_1[0][i][1])
            
    X_2=[]
    Y_2=[]
    for i in range(len(noisy_circles_2[0])):
        # make_circles creates two circles, we only want to create one
        if (noisy_circles_2[0][i][0])**2+noisy_circles_2[0][i][1]**2<.7:
            X_2.append(noisy_circles_2[0][i][0])
            Y_2.append(noisy_circles_2[0][i][1])
    X_3=[]
    Y_3=[]
    for i in range(len(X_1)):
        # make_circles creates two circles, we only want to create one
        X_3.append(X_1[i]+position[0][0])
        Y_3.append(Y_1[i]+position[0][1])
            
    X_4=[]
    Y_4=[]
    for i in range(len(X_2)):
        X_4.append(X_2[i]+position[0][0])
        Y_4.append(Y_2[i]+position[0][1])
        
    X_5=[]
    Y_5=[]
    for i in range(len(X_1)):
        # make_circles creates two circles, we only want to create one
        X_5.append(X_1[i]+position[1][0])
        Y_5.append(Y_1[i]+position[1][1])
            
    X_6=[]
    Y_6=[]
    for i in range(len(X_2)):
        X_6.append(X_2[i]+position[1][0])
        Y_6.append(Y_2[i]+position[1][1])
        
    X_7=[]
    Y_7=[]
    for i in range(len(X_1)):
        # make_circles creates two circles, we only want to create one
        X_7.append(X_1[i]+position[2][0])
        Y_7.append(Y_1[i]+position[2][1])
            
    X_8=[]
    Y_8=[]
    for i in range(len(X_2)):
        X_8.append(X_2[i]+position[2][0])
        Y_8.append(Y_2[i]+position[2][1])
   
   
            
    if plot:
        plt.plot(X_1,Y_1,'x',c='r')
        plt.plot(X_2,Y_2,'x',c='b')
        plt.plot(X_3,Y_3,'x',c='y')
        plt.plot(X_4,Y_4,'x',c='b')
        plt.plot(X_5,Y_5,'x',c='g')
        plt.plot(X_6,Y_6,'x',c='b')
        plt.plot(X_7,Y_7,'x',c='k')
        plt.plot(X_8,Y_8,'x',c='b')
        plt.show()
        
    boundary=[X_2+X_4+X_6+X_8,Y_2+Y_4+Y_6+Y_8]
    class1=[X_1,Y_1]
    class2=[X_3,Y_3]
    class3=[X_5,Y_5]
    class4=[X_7,Y_7]
    
    return boundary, class1, class2, class3, class4
