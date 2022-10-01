from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import os


ERR_NOT_ENUF_COLORS = 'Colors should be the same length as or longer plot_count'
ERR_TOO_MANY_PLOTS = 'Too many plots on a single figure'
ERR_NOT_ENUF_LABELS = 'Label count should match plot count'
# COLORS = 'cbrgmykw' # all default colours



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


def is_valid(plot_count, label):
    '''check input validity'''

    if label:
        assert len(label) == plot_count, ERR_TOO_MANY_PLOTS

def make_dir(file_dir):
    '''checks if the directory exists if not make one'''
    if file_dir:
        if not os.path.exists(file_dir):
            os.mkdir(file_dir)

def plot_2D(x, y, plot_count=1,title=None,x_title=None,y_title=None,label=False,filename=None,
        file_dir='plots', multi_x=True):

    '''plots inputs: x:array like of array like, y:array like of array likes,
    plot_count:int(number of plots),title:string, file_dir:string,colour:string'''

    is_valid(plot_count, label)

    make_dir(file_dir)

    if plot_count == 1:
        x = [x]
        y = [y]

    for i in range(plot_count):
        if multi_x:
            if label:
                plt.plot(x[i],y[i],label=label[i])
            else:
                plt.plot(x[i],y[i])
        else:
            if label:
                plt.plot(x,y[i],label=label[i])
            else:
                plt.plot(x,y[i])



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
