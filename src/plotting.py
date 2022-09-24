import matplotlib.pyplot as plt

def simple_plot(x,y,title = "Temporary title",xlab = "x",ylab = "y"):
    # Simple plot of 2 vectors
    title_axlabs(title,xlab,ylab)
    plt.plot(x,y)
    plt.show()


def multi_plot(x_vecs,y_vecs,labels,title = "Temporary title",xlab = "x",ylab = "y"):
    # Plots several graphs that share x-values
    # Takes lists (of the same length) of input vectors for x and y
    title_axlabs(title,xlab,ylab)
    for i,vals in enumerate(zip(x_vecs,y_vecs)):
        x = vals[0]
        y = vals[1]
        plt.plot(x,y, label = labels[i])
    plt.legend()
    plt.show()

def multi_yplot(x,y_vecs,labels,title = "Temporary title",xlab = "x",ylab = "y"):
    # Plots several graphs that share x-values
    # Takes list of y-vectors, but one x-vector as input
    title_axlabs(title,xlab,ylab)
    for i,y in enumerate(y_vecs):
        plt.plot(x,y, label = labels[i])
    plt.legend()
    plt.show()

def title_axlabs(title,xlab,ylab):
    # Helper function for making title and axis labels
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
