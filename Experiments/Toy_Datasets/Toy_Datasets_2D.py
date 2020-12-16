import autograd.numpy as np
import autograd.numpy as np
from sklearn import cluster, datasets
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')



######## Circular classes: concentric circles, with the boundary class as concentri cirlces ############


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
        plt.figure(figsize=(20,10))
        plt.plot(X_1, Y_1, 'x', c='r')
        plt.plot(X_2, Y_2, 'x', c='b')
        plt.plot(X_3, Y_3, 'x', c='y')
        plt.plot(X_4, Y_4, 'x', c='b')
        plt.show()

    boundary = [X_2 + X_4, Y_2 + Y_4]
    class1 = [X_1, Y_1]
    class2 = [X_3, Y_3]

    return boundary, class1, class2



def create_three_circular_classes(n=5000, noise_input=0.02, plot=False):
    """
    INPUT:
    n/2 is the target number of points in each circle (note that this is a target)
    noise is the noise used in sklearn make_circles function
    plot: if TRUE, will return a plot of the three classes of points (in red, yellow, green)
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
        plt.figure(figsize=(20,10))        
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



def create_four_circular_classes(n=5000, noise_input=0.02, plot=False):
    """
    INPUT:
    n/2 is the target number of points in each circle (note that this is a target)
    noise is the noise used in sklearn make_circles function
    plot: if TRUE, will return a plot of the four classes of points (in red, yellow, green, black)
    as well as the boundary class (in blue)
    
    OUPUT: boundary, class1, class2, class3, class4
    the four classes of points
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
        
    X_7=[]
    Y_7=[]
    for i in range(len(X_3)):
        X_7.append(X_3[i]*3.2)
        Y_7.append(Y_3[i]*3.2)
        
    X_8=[]
    Y_8=[]
    for i in range(len(X_3)):
        X_8.append(X_3[i]*3.9)
        Y_8.append(Y_3[i]*3.9)
            
    if plot:
        plt.figure(figsize=(20,10))
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




######## Circular classes with only the outer boundary: concentric circles, with the boundary class as the outer circle ############


def create_two_circular_classes_outer(n=1500, noise_input=0.05, plot=False, distance=1):
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
            X_4.append(distance*noisy_circles_4[0][i][0])
            Y_4.append(distance*noisy_circles_4[0][i][1])

    if plot:
        plt.figure(figsize=(20,10))
        plt.plot(X_1, Y_1, 'x', c='r')
        plt.plot(X_3, Y_3, 'x', c='y')
        plt.plot(X_4, Y_4, 'x', c='b')
        plt.show()

    boundary = [X_4, Y_4]
    class1 = [X_1, Y_1]
    class2 = [X_3, Y_3]

    return boundary, class1, class2



def create_three_circular_classes_outer(n=5000, noise_input=0.02, plot=False, distance=1):
    """
    INPUT:
    n/2 is the target number of points in each circle (note that this is a target)
    noise is the noise used in sklearn make_circles function
    plot: if TRUE, will return a plot of the three classes of points (in red, yellow, green)
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
            
    X_3=[]
    Y_3=[]
    for i in range(len(noisy_circles_3[0])):
        # make_circles creates two circles, we only want to create one
        if (noisy_circles_3[0][i][0])**2+noisy_circles_3[0][i][1]**2<.7:
            X_3.append(noisy_circles_3[0][i][0])
            Y_3.append(noisy_circles_3[0][i][1])
            
                
    X_5=[]
    Y_5=[]
    for i in range(len(X_3)):
        X_5.append(X_3[i]*1.9)
        Y_5.append(Y_3[i]*1.9)
        
    X_6=[]
    Y_6=[]
    for i in range(len(X_3)):
        X_6.append(distance*X_3[i]*2.5)
        Y_6.append(distance*Y_3[i]*2.5)
            
    if plot:
        plt.figure(figsize=(20,10))
        plt.plot(X_1,Y_1,'x',c='r')
        plt.plot(X_3,Y_3,'x',c='y')
        plt.plot(X_5,Y_5,'x',c='g')
        plt.plot(X_6,Y_6,'x',c='b')
        plt.show()
        
    boundary=[X_6,Y_6]
    class1=[X_1,Y_1]
    class2=[X_3,Y_3]
    class3=[X_5,Y_5]
    
    return boundary, class1, class2, class3



def create_four_circular_classes_outer(n=5000, noise_input=0.02, plot=False, distance=1):
    """
    INPUT:
    n/2 is the target number of points in each circle (note that this is a target)
    noise is the noise used in sklearn make_circles function
    plot: if TRUE, will return a plot of the four classes of points (in red, yellow, green, black)
    as well as the boundary class (in blue)
    
    OUPUT: boundary, class1, class2, class3, class4
    the four classes of points
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
            
    X_3=[]
    Y_3=[]
    for i in range(len(noisy_circles_3[0])):
        # make_circles creates two circles, we only want to create one
        if (noisy_circles_3[0][i][0])**2+noisy_circles_3[0][i][1]**2<.7:
            X_3.append(noisy_circles_3[0][i][0])
            Y_3.append(noisy_circles_3[0][i][1])
            
                
    X_5=[]
    Y_5=[]
    for i in range(len(X_3)):
        X_5.append(X_3[i]*1.9)
        Y_5.append(Y_3[i]*1.9)
            
        
    X_7=[]
    Y_7=[]
    for i in range(len(X_3)):
        X_7.append(X_3[i]*3.2)
        Y_7.append(Y_3[i]*3.2)
        
    X_8=[]
    Y_8=[]
    for i in range(len(X_3)):
        X_8.append(distance*X_3[i]*3.9)
        Y_8.append(distance*Y_3[i]*3.9)
            
    if plot:
        plt.figure(figsize=(20,10))
        plt.plot(X_1,Y_1,'x',c='r')
        plt.plot(X_3,Y_3,'x',c='y')
        plt.plot(X_5,Y_5,'x',c='g')
        plt.plot(X_7,Y_7,'x',c='k')
        plt.plot(X_8,Y_8,'x',c='b')
        plt.show()
        
    boundary=[X_8,Y_8]
    class1=[X_1,Y_1]
    class2=[X_3,Y_3]
    class3=[X_5,Y_5]
    class4=[X_7,Y_7]
    
    return boundary, class1, class2, class3, class4




######## Circular classes with only an imperfect outer boundary: concentric circles, with the boundary class as the outer circle ############


def create_two_circular_classes_outerimperfect(n=1500, noise_input=0.05, plot=False, gap=0.2, distance=1):
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
            if np.abs(noisy_circles_4[0][i][0])>gap:
                X_4.append(distance*noisy_circles_4[0][i][0])
                Y_4.append(distance*noisy_circles_4[0][i][1])

    if plot:
        plt.figure(figsize=(20,10))
        plt.plot(X_1, Y_1, 'x', c='r')
        plt.plot(X_3, Y_3, 'x', c='y')
        plt.plot(X_4, Y_4, 'x', c='b')
        plt.show()

    boundary = [X_4, Y_4]
    class1 = [X_1, Y_1]
    class2 = [X_3, Y_3]

    return boundary, class1, class2



def create_three_circular_classes_outerimperfect(n=5000, noise_input=0.02, plot=False, gap=0.2, distance=1):
    """
    INPUT:
    n/2 is the target number of points in each circle (note that this is a target)
    noise is the noise used in sklearn make_circles function
    plot: if TRUE, will return a plot of the three classes of points (in red, yellow, green)
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
            
    X_3=[]
    Y_3=[]
    for i in range(len(noisy_circles_3[0])):
        # make_circles creates two circles, we only want to create one
        if (noisy_circles_3[0][i][0])**2+noisy_circles_3[0][i][1]**2<.7:
            X_3.append(noisy_circles_3[0][i][0])
            Y_3.append(noisy_circles_3[0][i][1])
            
                
    X_5=[]
    Y_5=[]
    for i in range(len(X_3)):
        X_5.append(X_3[i]*1.9)
        Y_5.append(Y_3[i]*1.9)
        
    X_6=[]
    Y_6=[]
    for i in range(len(X_3)):
        if np.abs(X_3[i])>gap:
            X_6.append(distance*X_3[i]*2.5)
            Y_6.append(distance*Y_3[i]*2.5)
            
    if plot:
        plt.figure(figsize=(20,10))
        plt.plot(X_1,Y_1,'x',c='r')
        plt.plot(X_3,Y_3,'x',c='y')
        plt.plot(X_5,Y_5,'x',c='g')
        plt.plot(X_6,Y_6,'x',c='b')
        plt.show()
        
    boundary=[X_6,Y_6]
    class1=[X_1,Y_1]
    class2=[X_3,Y_3]
    class3=[X_5,Y_5]
    
    return boundary, class1, class2, class3



def create_four_circular_classes_outerimperfect(n=5000, noise_input=0.02, plot=False, gap=0.2, distance=1):
    """
    INPUT:
    n/2 is the target number of points in each circle (note that this is a target)
    noise is the noise used in sklearn make_circles function
    plot: if TRUE, will return a plot of the four classes of points (in red, yellow, green, black)
    as well as the boundary class (in blue)
    
    OUPUT: boundary, class1, class2, class3, class4
    the four classes of points
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
            
    X_3=[]
    Y_3=[]
    for i in range(len(noisy_circles_3[0])):
        # make_circles creates two circles, we only want to create one
        if (noisy_circles_3[0][i][0])**2+noisy_circles_3[0][i][1]**2<.7:
            X_3.append(noisy_circles_3[0][i][0])
            Y_3.append(noisy_circles_3[0][i][1])
            
                
    X_5=[]
    Y_5=[]
    for i in range(len(X_3)):
        X_5.append(X_3[i]*1.9)
        Y_5.append(Y_3[i]*1.9)
            
        
    X_7=[]
    Y_7=[]
    for i in range(len(X_3)):
        X_7.append(X_3[i]*3.2)
        Y_7.append(Y_3[i]*3.2)
        
    X_8=[]
    Y_8=[]
    for i in range(len(X_3)):
        if np.abs(X_3[i])>gap:
            X_8.append(distance*X_3[i]*3.9)
            Y_8.append(distance*Y_3[i]*3.9)
            
    if plot:
        plt.figure(figsize=(20,10))
        plt.plot(X_1,Y_1,'x',c='r')
        plt.plot(X_3,Y_3,'x',c='y')
        plt.plot(X_5,Y_5,'x',c='g')
        plt.plot(X_7,Y_7,'x',c='k')
        plt.plot(X_8,Y_8,'x',c='b')
        plt.show()
        
    boundary=[X_8,Y_8]
    class1=[X_1,Y_1]
    class2=[X_3,Y_3]
    class3=[X_5,Y_5]
    class4=[X_7,Y_7]
    
    return boundary, class1, class2, class3, class4



############# Disks with boundary classes as circles around the disks ###################



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
        plt.figure(figsize=(20,10))
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
        plt.figure(figsize=(20,10))
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
        plt.figure(figsize=(20,10))
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


########## Moons with boundary around each moon ##########

from scipy.interpolate import lagrange
import numpy as np

def two_moon(n = 1500, noise_input=.01, plot=False):
    """
    INPUTS: 
    n: the number of points in the two half-moons (n/2 for each half moon)
    noise_input: allow to have more noisy half-moons (the boundary won't be more noisy)
    plot: if set to True, will return the plot
    
    OUTPUTS:
    a plot (if plot=True)
    boundary class
    class1
    class2 
    
    """
    
    # create the half-moons using sklearn function
    noisy_moons = datasets.make_moons(n_samples=n, noise=noise_input)
    
    # get the two classes
    X_1=[]
    Y_1=[]
    X_2=[]
    Y_2=[]
    for i in range(len(noisy_moons[0])):
        if noisy_moons[1][i]==0:
            X_1.append(noisy_moons[0][i][0])
            Y_1.append(noisy_moons[0][i][1])
        else:
            X_2.append(noisy_moons[0][i][0])
            Y_2.append(noisy_moons[0][i][1])
            
    #######################
    # create the boundaries
    #######################
    x11=[-1.1 , -0.9, -0.75, 0, 0.75, 0.9, 1.1]
    y11=[-0.1 , 0.6,  0.8, 1.05, 0.8, 0.6, -0.1]
    poly11 = lagrange(x11, y11)

    x12=[ -0.9, -0.75, 0, 0.75, 0.9]
    y12=[ 0.1,  0.5, 0.9, 0.5, 0.1]
    poly12 = lagrange(x12, y12)

    x13=[ -1.1,-1,-0.9]
    y13=[ -0.1, -0.08, 0.1]
    poly13 = lagrange(x13, y13)

    x14=[ 0.9,1,1.1]
    y14=[ 0.1, -0.08, -0.1]
    poly14 = lagrange(x14, y14)

    x21=[-0.1 , 0.1, 0.25, 1, 1.75, 1.9, 2.1]
    y21=[0.6 , -0.1,  -0.3, -0.55, -0.3, -0.1, 0.6]
    poly21 = lagrange(x21, y21)

    x22=[ 0.1, 0.25, 1, 1.75, 1.9]
    y22=[ 0.4,  0, -0.4, 0, 0.4]
    poly22 = lagrange(x22, y22)

    x23=[-0.1, 0, 0.1]
    y23=[0.6, 0.58, 0.4]
    poly23 = lagrange(x23, y23)

    x24=[ 1.9,2,2.1]
    y24=[0.4, 0.58, 0.6]
    poly24 = lagrange(x24, y24)

    x_ax11=np.linspace(-1.1,1.1,1000)
    y_ax11=poly11(x_ax11)
    x_ax12=np.linspace(-0.9,0.9,1000)
    y_ax12=poly12(x_ax12)
    x_ax13=np.linspace(-1.1,-0.9,100)
    y_ax13=poly13(x_ax13)
    x_ax14=np.linspace(0.9,1.1,100)
    y_ax14=poly14(x_ax14)
    x_ax21=np.linspace(-0.1,2.1,1000)
    y_ax21=poly21(x_ax21)
    x_ax22=np.linspace(0.1,1.9,1000)
    y_ax22=poly22(x_ax22)
    x_ax23=np.linspace(-0.1,0.1,100)
    y_ax23=poly23(x_ax23)
    x_ax24=np.linspace(1.9,2.1,100)
    y_ax24=poly24(x_ax24)
    
    #######################
    # finished the boundaries
    #######################

    
    # plot if necessary
    # boundary class in blue
    # classes in red and yellow
    if plot==True:
        f,ax=plt.subplots(1,1,figsize=(12,12))
        ax.plot(X_1,Y_1,'x', c='r')
        ax.plot(X_2,Y_2,'x', c='y')
        ax.plot(x_ax11,y_ax11,'x', c='b')
        ax.plot(x_ax12,y_ax12,'x', c='b')
        ax.plot(x_ax13,y_ax13,'x', c='b')
        ax.plot(x_ax14,y_ax14,'x', c='b')
        ax.plot(x_ax21,y_ax21,'x', c='b')
        ax.plot(x_ax22,y_ax22,'x', c='b')
        ax.plot(x_ax23,y_ax23,'x', c='b')
        ax.plot(x_ax24,y_ax24,'x', c='b')
        plt.show()
        
    # Outputs  
    class1=[X_1,Y_1]
    class2=[X_2,Y_2]
    bx=list(x_ax11)+list(x_ax12)+list(x_ax13)+list(x_ax14)+list(x_ax21)+list(x_ax22)+list(x_ax23)+list(x_ax24)
    by=list(y_ax11)+list(y_ax12)+list(y_ax13)+list(y_ax14)+list(y_ax21)+list(y_ax22)+list(y_ax23)+list(y_ax24)
    boundary=[bx,by]
    
    return boundary, class1, class2


def four_moon(n = 1500, noise_input=.01, plot=False, matrix=[[1,0],[0,1]],translate=[2,2]):
    """
    INPUTS: 
    n: the number of points in the fours half-moons is 2*n (n/2 for each half moon)
    noise_input: allow to have more noisy half-moons (the boundary won't be more noisy)
    plot: if set to True, will return the plot
    matrix: we have two copy of the same two half-moons: the second copy is a linear transformation Ax+b
    where A=matrix, b=translate
    
    OUTPUTS:
    a plot (if plot=True)
    boundary class
    class1
    class2 
    class3
    class4
    
    """
    n_call=n
    noise_call=noise_input
    boundary_i,class1_i,class2_i=two_moon(n=n_call, noise_input= noise_call)
    for i in range(len(boundary_i[0])):
        x=boundary_i[0][i]
        y=boundary_i[1][i]
        trans=np.matmul(matrix,[x,y])+translate
        boundary_i[0].append(trans[0])
        boundary_i[1].append(trans[1])
    
    class3=[[],[]]
    class4=[[],[]]
    for i in range(len(class1_i[0])):
        x1=class1_i[0][i]
        y1=class1_i[1][i]
        x2=class2_i[0][i]
        y2=class2_i[1][i]
        trans1=np.matmul(matrix,[x1,y1])+translate
        trans2=np.matmul(matrix,[x2,y2])+translate
        class3[0].append(trans1[0])
        class3[1].append(trans1[1])
        class4[0].append(trans2[0])
        class4[1].append(trans2[1])
    
    boundary=boundary_i
    class1=class1_i
    class2=class2_i
    
    # plot if necessary
    # boundary class in blue
    # classes in red and yellow
    if plot==True:
        f,ax=plt.subplots(1,1,figsize=(12,12))
        ax.plot(boundary[0],boundary[1],'x',c='b')
        ax.plot(class1[0],class1[1],'x',c='y')
        ax.plot(class2[0],class2[1],'x',c='r')
        ax.plot(class3[0],class3[1],'x',c='k')
        ax.plot(class4[0],class4[1],'x',c='g')
        plt.show()
        
    return boundary, class1, class2, class3, class4


def six_moon(n = 1500, noise_input=.01, plot=False, matrix1=[[1,0],[0,1]],translate1=[2,2], matrix2=[[-1,2],[0,1]], translate2=[-2,-2]):
    """
    INPUTS: 
    n: the number of points in the six half-moons is 3*n (n/2 for each half moon)
    noise_input: allow to have more noisy half-moons (the boundary won't be more noisy)
    plot: if set to True, will return the plot
    matrix: we have three copy of the same two half-moons: the second,third copies are linear transformation Ax+b
    where A=matrix1 or matrix2, b=translate1 or translate2
    
    OUTPUTS:
    a plot (if plot=True)
    boundary class
    class1
    class2 
    class3
    class4
    class5
    class6
    
    """
    n_call=n
    noise_call=noise_input
    boundary_i,class1_i,class2_i=two_moon(n=n_call, noise_input= noise_call)
    for i in range(len(boundary_i[0])):
        x=boundary_i[0][i]
        y=boundary_i[1][i]
        trans01=np.matmul(matrix1,[x,y])+translate1
        trans02=np.matmul(matrix2,[x,y])+translate2
        boundary_i[0].append(trans01[0])
        boundary_i[1].append(trans01[1])
        boundary_i[0].append(trans02[0])
        boundary_i[1].append(trans02[1])
    
    class3=[[],[]]
    class4=[[],[]]
    class5=[[],[]]
    class6=[[],[]]
    for i in range(len(class1_i[0])):
        x1=class1_i[0][i]
        y1=class1_i[1][i]
        x2=class2_i[0][i]
        y2=class2_i[1][i]
        trans1=np.matmul(matrix1,[x1,y1])+translate1
        trans2=np.matmul(matrix1,[x2,y2])+translate1
        trans3=np.matmul(matrix2,[x1,y1])+translate2
        trans4=np.matmul(matrix2,[x2,y2])+translate2
        class3[0].append(trans1[0])
        class3[1].append(trans1[1])
        class4[0].append(trans2[0])
        class4[1].append(trans2[1])
        class5[0].append(trans3[0])
        class5[1].append(trans3[1])
        class6[0].append(trans4[0])
        class6[1].append(trans4[1])
    
    boundary=boundary_i
    class1=class1_i
    class2=class2_i
    
    # plot if necessary
    # boundary class in blue
    # classes in red and yellow
    if plot==True:
        f,ax=plt.subplots(1,1,figsize=(12,12))
        ax.plot(boundary[0],boundary[1],'x',c='b')
        ax.plot(class1[0],class1[1],'x',c='y')
        ax.plot(class2[0],class2[1],'x',c='r')
        ax.plot(class3[0],class3[1],'x',c='k')
        ax.plot(class4[0],class4[1],'x',c='g')
        ax.plot(class5[0],class5[1],'x',c='m')
        ax.plot(class6[0],class6[1],'x',c='c')
        plt.show()
        
    return boundary, class1, class2, class3, class4, class5, class6


########## Moons with one boundary ##########

def two_moon_circular_boundary(n = 1500, noise_input=.01, plot=False, distance=1, n_boundary=1500, noise_input_boundary=0.01):
    """
    INPUTS: 
    n: the number of points in the two half-moons (n/2 for each half moon)
    noise_input: allow to have more noisy half-moons (the boundary won't be more noisy)
    plot: if set to True, will return the plot
    
    OUTPUTS:
    a plot (if plot=True)
    boundary class
    class1
    class2 
    
    """
    
    # create the half-moons using sklearn function
    noisy_moons = datasets.make_moons(n_samples=n, noise=noise_input)
    
    
    # get the two classes
    X_1=[]
    Y_1=[]
    X_2=[]
    Y_2=[]
    for i in range(len(noisy_moons[0])):
        if noisy_moons[1][i]==0:
            X_1.append(noisy_moons[0][i][0])
            Y_1.append(noisy_moons[0][i][1])
        else:
            X_2.append(noisy_moons[0][i][0])
            Y_2.append(noisy_moons[0][i][1])
            
    #######################
    # create the boundaries
    #######################
    noisy_circles_2 = datasets.make_circles(n_samples=n_boundary, factor=.3, noise=noise_input_boundary)
    
    X_3 = []
    Y_3 = []
    for i in range(len(noisy_circles_2[0])):
        # make_circles creates two circles, we only want to create one
        if (noisy_circles_2[0][i][0]) ** 2 + noisy_circles_2[0][i][1] ** 2 < .7:
            X_3.append(distance*10*(noisy_circles_2[0][i][0]+.02))
            Y_3.append(distance*10*noisy_circles_2[0][i][1])
    
    #######################
    # finished the boundaries
    #######################

    
    # plot if necessary
    # boundary class in blue
    # classes in red and yellow
    if plot==True:
        f,ax=plt.subplots(1,1,figsize=(12,12))
        ax.plot(X_1,Y_1,'x', c='r')
        ax.plot(X_2,Y_2,'x', c='y')
        ax.plot(X_3,Y_3,'x', c='b')
        plt.show()
        
    # Outputs  
    class1=[X_1,Y_1]
    class2=[X_2,Y_2]
    boundary=[X_3,Y_3]
    
    return boundary, class1, class2






####### TO FINISH: RAPH
############# Disks with boundary classes as circles around the disks -> just one disk ###################
########## Moons -> just one boundary ##########




