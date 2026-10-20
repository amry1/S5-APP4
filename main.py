import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
from scipy import signal
import matplotlib
from matplotlib import patches
from matplotlib.figure import Figure
from matplotlib import rcParams

def aberration():
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
    return image_filtrer

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

def rotation_mat2():
    matplotlib.pyplot.gray()
    img_couleur = matplotlib.image.imread('images/goldhill_rotate.png')
    image = np.mean(img_couleur, -1)

    plt.imshow(image, cmap='gray')

    hauteur, largeur = image.shape

    image_rot = np.zeros([largeur, hauteur])

    for x in range(largeur):
            for y in range(hauteur):
                dep_x = hauteur-1 -y
                dep_y = x
    
                image_rot[dep_y, dep_x] = image[y, x]
    
    plt.imshow(image_rot, cmap='gray')
    plt.show()


def RII():
    w_a = 4789.14
    f_e = 1600

    b = [w_a**2, 2*w_a**2, w_a**2]
    a = [w_a**2 + 4*f_e**2 + np.sqrt(2)*2*f_e*w_a, 
         -8*f_e**2 + 2*w_a**2, 
         4*f_e**2 - np.sqrt(2)*2*f_e*w_a + w_a**2]

    a_0 = a[0]
    b_norm = [coef / a_0 for coef in b]
    a_norm = [coef / a_0 for coef in a]

    plt.figure()
    zplane(b_norm, a_norm)
    plt.show()

    freq, h = signal.freqz(b, a)

    plt.figure()
    plt.semilogy()
    plt.plot(freq, np.abs(h))
    plt.show()

    print("--- Méthode Manuelle Normalisée ---")
    print("b_norm: ", b_norm)
    print("a_norm: ", a_norm)

    print("\n--- Méthode Python ---")
    f_c = 500
    nyq = 0.5 * f_e
    w_n = f_c / nyq
    gpass = 0.2
    gstop = 60

    N_but, w_b = signal.buttord(500/nyq, 750/nyq, 0.2, 60)
    N_c1, w_c1 = signal.cheb1ord(500/nyq, 750/nyq, 0.2, 60)
    N_c2, w_c2 = signal.cheb2ord(500/nyq, 750/nyq, 0.2, 60)
    N_e, w_e = signal.ellipord(500/nyq, 750/nyq, 0.2, 60)



    filters = {
        "Butterworth": (N_but, w_b, signal.butter),
        "Chebyshev Type I": (N_c1, w_c1, signal.cheby1),
        "Chebyshev Type II": (N_c2, w_c2, signal.cheby2),
        "Elliptic": (N_e, w_e, signal.ellip)
    }

    chosen_name, (order, w_n, filter_func) = min(
        filters.items(), key=lambda item: item[1][0]
    )

    
    if chosen_name == "Chebyshev Type I":
        b_1, a_1 = filter_func(order, gpass, w_n, btype='low')
    elif chosen_name == "Chebyshev Type II":
        b_1, a_1 = filter_func(order, gstop, w_n, btype='low')
    elif chosen_name == "Elliptic":
        b_1, a_1 = filter_func(order, gpass, gstop, w_n, btype='low')
    else:
        b_1, a_1 = filter_func(order, w_n, btype='low')


    print(f"Chosen filter: {chosen_name}")
    print(f"Filter order: {order}")
    print(f"Normalized cutoff frequency: {w_n}")
    print(f"Numerator coefficients (b_1):\n{b_1}")
    print(f"Denominator coefficients (a_1):\n{a_1}")

    plt.figure()
    zplane(b_1, a_1)
    plt.show()

    freq, h = signal.freqz(b_1, a_1)

    plt.figure()
    plt.semilogy()
    plt.plot(freq, np.abs(h))
    plt.show()

    image = np.load('images/goldhill_bruit.npy')
    image_filtrer1 = signal.lfilter(b, a, image)
    image_filtrer = signal.lfilter(b_1, a_1, image)
    plt.imshow(image, cmap='gray')
    plt.show()
    plt.imshow(image_filtrer1, cmap='gray')
    plt.show()
    plt.imshow(image_filtrer, cmap='gray')
    plt.show()


def compression():

    I_o = aberration()

    C_xx = np.cov(I_o, rowvar=True)


    valeurs_propres, vecteurs_propres = np.linalg.eigh(C_xx)
    indices_tries = np.argsort(valeurs_propres)[::-1]
    vecteurs_propres_tries = vecteurs_propres[:, indices_tries]


    P_vo = vecteurs_propres_tries.T

    I_v = P_vo @ I_o
    
    hauteur = I_v.shape[0]
    ratio_compr = 0.7
    retrait_ligne = int(hauteur * ratio_compr)

    I_v_comp = I_v.copy()
    I_v_comp[-retrait_ligne:, :] = 0    

    P_ov = np.linalg.inv(P_vo)
    img_reconstruite = P_ov @ I_v_comp
    
    plt.imshow(img_reconstruite, cmap='gray')
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


aberration()
rotation_mat()
rotation_mat2()
RII()
compression()