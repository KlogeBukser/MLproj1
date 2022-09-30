from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import os


ERR_NOT_ENUF_COLORS = 'Colors should be the same length as or longer plot_count'
ERR_TOO_MANY_PLOTS = 'Too many plots on a single figure'
ERR_NOT_ENUF_LABELS = 'Label count should match plot count'
COLORS = 'cbrgmykw' # all default colours


def simple_plot(x,y,title = "Temporary title",xlab = "x",ylab = "y"):
    # Simple plot of 2 vectors
    title_axlabs(title,xlab,ylab)
    plt.plot(x,y)


def multi_plot(x_vecs,y_vecs,labels,title = "Temporary title",xlab = "x",ylab = "y"):
    # Plots several graphs that share x-values
    # Takes lists (of the same length) of input vectors for x and y
    title_axlabs(title,xlab,ylab)
    for i,vals in enumerate(zip(x_vecs,y_vecs)):
        x = vals[0]
        y = vals[1]
        plt.plot(x,y, label = labels[i])
    plt.legend()

def multi_yplot(x,y_vecs,labels,title = "Temporary title",xlab = "x",ylab = "y"):
    # Plots several graphs that share x-values
    # Takes list of y-vectors, but one x-vector as input
    title_axlabs(title,xlab,ylab)
    for i,y in enumerate(y_vecs):
        plt.plot(x,y, label = labels[i])
    plt.legend()

def title_axlabs(title,xlab,ylab):
    # Helper function for making title and axis labels
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)

def set_paras(x_title,y_title,title=None,filename=None,file_dir='plots',has_label=False):

    '''set all the parameters in the figure and save files'''
    if has_label:
        plt.legend()
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.title(title)

    if filename:
        full_path = os.path.join(file_dir, filename)
        plt.savefig(full_path)
        # plt.close()
        plt.show() #for testing
    else:
        plt.show()


def is_valid(plot_count,color, label):
    '''check input validity'''
    assert plot_count <= 8, ERR_TOO_MANY_PLOTS
    if color:
        assert len(color) >= plot_count, ERR_NOT_ENUF_COLORS
    if label:
        assert len(label) == plot_count, ERR_TOO_MANY_PLOTS

def make_dir(file_dir):
    '''checks if the directory exists if not make one'''
    if file_dir:
        if not os.path.exists(file_dir):
            os.mkdir(file_dir)

def plot_2D(x, y, plot_count=1,title=None,x_title=None,y_title=None,label=False,filename=None,
        file_dir='plots',color=COLORS, multi_x=True):

    '''plots inputs: x:array like of array like, y:array like of array likes,
    plot_count:int(number of plots),title:string, file_dir:string,colour:string'''

    is_valid(plot_count, color, label)

    make_dir(file_dir)

    for i in range(plot_count):
        if multi_x:
            if label:
                plt.plot(x[i],y[i],label=label[i],color=color[i])
            else:
                plt.plot(x[i],y[i],color=color[i])
        else:
            if label:
                plt.plot(x,y[i],label=label[i],color=color[i])
            else:
                plt.plot(x,y[i],color=color[i])



    set_paras(x_title, y_title, title, filename, file_dir, label)


def plot_surface(x,y,z,title=None,x_title=None,y_title=None,z_title=None,label=None,filename=None,
        file_dir=''):

    # some of the code from this function is given in the project

    make_dir(file_dir)

    fig = plt.figure()
    ax = plt.axes(projection ='3d')

    surf = ax.plot_surface(x, y, z,cmap ='viridis',
                       linewidth=0, antialiased=False)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    set_paras(x_title, y_title, title, filename, file_dir)
