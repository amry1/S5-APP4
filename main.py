import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
from scipy import signal
import matplotlib
from matplotlib import patches
from matplotlib.figure import Figure
from matplotlib import rcParams

def abberation():
    image = np.load('images/goldhill_aberrations.npy')
    a = np.poly([0.9*np.exp((1j*(np.pi/2))),0.9*np.exp((-1j*(np.pi/2))), 0.95*np.exp((1j*(np.pi/8))), 0.95*np.exp((-1j*(np.pi/8)))])
    b = np.poly([0, -0.99, -0.99, 0.8])
    zplane(a,b, "Abberation Poles et Zero")
    image_filtrer = signal.lfilter(b, a, image)
    plt.show()
    plt.imshow(image, cmap='gray')
    plt.show()
    plt.imshow(image_filtrer, cmap='gray')
    plt.show()


import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def rotation_mat():
    matplotlib.pyplot.gray()
    img_couleur = matplotlib.image.imread('images/goldhill_rotate.png')
    image = np.mean(img_couleur, -1)

    plt.imshow(image, cmap='gray')
    plt.show()

    hauteur, largeur = image.shape

    image_rot = np.zeros((largeur, hauteur))

    for x in range(largeur):
        for y in range(hauteur):
            dep_x = -y + hauteur - 1
            dep_y = x

            image_rot[dep_y, dep_x] = image[y, x]

    plt.imshow(image_rot, cmap='gray')
    plt.show()


def zplane(b, a, filename=None):
    """Plot the complex z-plane given a transfer function.
    """

    # get a figure/plot
    ax = plt.subplot(111)

    # create the unit circle
    uc = patches.Circle((0, 0), radius=1, fill=False,
                        color='black', ls='dashed')
    ax.add_patch(uc)

    # The coefficients are less than 1, normalize the coeficients
    if np.max(b) > 1:
        kn = np.max(b)
        b = b / float(kn)
    else:
        kn = 1

    if np.max(a) > 1:
        kd = np.max(a)
        a = a / float(kd)
    else:
        kd = 1

    # Get the poles and zeros
    p = np.roots(a)
    z = np.roots(b)
    k = kn / float(kd)

    # Plot the zeros and set marker properties
    t1 = plt.plot(z.real, z.imag, 'go', ms=10)
    plt.setp(t1, markersize=10.0, markeredgewidth=1.0,
             markeredgecolor='k', markerfacecolor='g')

    # Plot the poles and set marker properties
    t2 = plt.plot(p.real, p.imag, 'rx', ms=10)
    plt.setp(t2, markersize=12.0, markeredgewidth=3.0,
             markeredgecolor='r', markerfacecolor='r')

    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # set the ticks
    r = 1.5;
    plt.axis('scaled');
    plt.axis([-r, r, -r, r])
    ticks = [-1, -.5, .5, 1];
    plt.xticks(ticks);
    plt.yticks(ticks)

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)

    return z, p, k


abberation()
rotation_mat()