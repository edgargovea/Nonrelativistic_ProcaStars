# Proyecto proca v.1.0

####################
# LOADING MODULES
####################
import matplotlib.pyplot as plt 
import numpy as np
import sys

from matplotlib.colors import LinearSegmentedColormap

from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.integrate import solve_ivp, quad
from scipy.linalg import eig
from matplotlib.patches import FancyArrowPatch

from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d import proj3d
from matplotlib.text import Annotation
from matplotlib.patches import ArrowStyle
from matplotlib import cm 


####################
# TOOLS
####################
def progressbar(current_value, total_value, bar_lengh, progress_char): 
    """
    Barra de progreso
    """
    percentage = int((current_value/total_value)*100)                                                # Percent Completed Calculation 
    progress = int((bar_lengh * current_value ) / total_value)                                       # Progress Done Calculation 
    loadbar = "Progress: [{:{len}}]{}%".format(progress*progress_char,percentage, len = bar_lengh)    # Progress Bar String
    print(loadbar, end='\r')

def find_nearest(array, value):
    """
    Encontrando el valor más cercano
    """
    n = [abs(i-value) for i in array]
    idx = n.index(min(n))
    #print(idx)
    return (array[idx], idx)

# https://stackoverflow.com/questions/67362634/regular-distribution-of-points-in-the-volume-of-a-sphere
def randomEsfera(n, r):
    """ 
    n -> número de puntos
    r -> radio de la esfera
    """
    x, y, z = [], [], []
    
    alpha = 4.0*np.pi*r*r/n 
    d = np.sqrt(alpha) 
    m_nu = int(np.round(np.pi/d))
    d_nu = np.pi/m_nu
    d_phi = alpha/d_nu
    count = 0
    
    for m in range(0, m_nu):
        nu = np.pi*(m+0.5)/m_nu
        m_phi = int(np.round(2*np.pi*np.sin(nu)/d_phi))
        for n in range(0, m_phi):
            phi = 2*np.pi*n/m_phi
            xp = r*np.sin(nu)*np.cos(phi)
            yp = r*np.sin(nu)*np.sin(phi)
            zp = r*np.cos(nu)
            x.append(xp)
            y.append(yp)
            z.append(zp)
            count = count +1
    return x, y, z

def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return list(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def rgb_to_hex(rgb):
    return '#%02x%02x%02x'%rgb

def plt_sphere(ax, list_center, list_radius, color='#04617d'):
  for c, r in zip(list_center, list_radius):    
    # draw sphere
    #u, v = np.mgrid[0:2*np.pi:5j, 0:np.pi:5j]  # 20
    #x = r*np.cos(u)*np.sin(v)
    #y = r*np.sin(u)*np.sin(v)
    #z = r*np.cos(v)
    #ax.plot_surface(x-c[0], y-c[1], z-c[2],
    #                color='#04617d', alpha=0.5)
    
    u, v = np.mgrid[0:2*np.pi:15j, 0:np.pi:15j]
    x = r*np.cos(u)*np.sin(v)
    y = r*np.sin(u)*np.sin(v)
    z = r*np.cos(v)
    ax.plot_wireframe(x-c[0], y-c[1], z-c[2], color=color, lw=.2)
    
    ax.set_aspect("equal")

# plot círculo difuminado
# https://stackoverflow.com/questions/70003173/how-to-plot-a-hollow-circle-with-gradient-fadeout-glow-in-matplotlib
def circlef(ax, xy0, func, rad, inrad=0,
            halo_color='#04617d', colf='#FFFFFF00',
            center_color='white', npt=500, orig=False,
            colorbar=False, fig=None, pad=0.02, label=r'$n(r)$',
            location='left', aspect=20, fraction=0.047, zorder=3, fxy=False): 
    
    x0, y0 = xy0
    # Creando la malla                 
    xcoord = [x0 - rad, x0 + rad]  # [xmin, xmax]
    ycoord = [y0 - rad, y0 + rad]  # [ymin, ymax]
    
    x, y = np.meshgrid(
        np.linspace(xcoord[0], xcoord[1], npt),
        np.linspace(ycoord[0], ycoord[1], npt))
    r = np.sqrt((x-x0)**2 + (y-y0)**2)
    
    if orig:
        z = np.where(r<inrad, np.nan, np.clip(rad-r, 0, np.inf))
        cmap = LinearSegmentedColormap.from_list('', [colf, halo_color]) 
        cmap.set_bad(center_color)
    elif fxy:
        z = func(x, y)/np.max(func(x, y))
        cmap = LinearSegmentedColormap.from_list('', [colf, halo_color])
    else:
        z = func(r)/np.max(func(r))
        cmap = LinearSegmentedColormap.from_list('', [colf, halo_color])
    
    im = ax.imshow(z, cmap=cmap, extent=[xcoord[0], xcoord[1], ycoord[0], ycoord[1]], origin='lower', zorder=zorder)
    
    if colorbar:
        cbar = fig.colorbar(cm.ScalarMappable(cmap=cmap), fraction=fraction, pad=pad,
             ax=ax, aspect=aspect, location=location)  # label=label
        cbar.ax.set_title(label, fontsize=14)
    
    return im

# plot circular arrow
# https://stackoverflow.com/questions/37512502/how-to-make-arrow-that-loops-in-matplotlib
def circarrowdraw(ax, x0, y0, z0=None, radius=1, aspect=1, direction=270, closingangle=-330,
                  arrowheadrelativesize=0.3, arrowheadopenangle=30, 
                  head_width=0.5, head_length=0.5,
                  cArr='k', cCir='k', Dim3=False, zorder=100):
    """
    Circular arrow drawing. x0 and y0 are the anchor points.
    direction gives the angle of the circle center relative to the anchor
    in degrees. closingangle indicates how much of the circle is drawn
    in degrees with positive being counterclockwise and negative being
    clockwise. aspect is important to make the aspect of the arrow 
    fit the current figure.
    """

    xc = x0 + radius * np.cos(direction*np.pi/180)
    yc = y0 + aspect * radius*np.sin(direction*np.pi/180)

    headcorrectionangle = 5

    if closingangle < 0:
        step = -1
    else:
        step = 1
    x = [xc + radius * np.cos((ang + 180 + direction) * np.pi / 180)
         for ang in np.arange(0, closingangle, step)]
    y = [yc + aspect * radius * np.sin((ang + 180 + direction) * np.pi / 180)
         for ang in np.arange(0, closingangle, step)]

    if Dim3:
        style = ArrowStyle('->', head_length=0.12, head_width=0.04, widthA=1.0, widthB=1.0, lengthA=0.2,
                       lengthB=0.2, angleA=0, angleB=0, scaleA=None, scaleB=None)
    
        ax.plot(x, y, [z0]*len(x), c=cArr, lw=1, zorder=zorder)  # graficando circulo
        xt, yt, zt = [xc, x[-1]], [yc, y0], [z0, z0]
        a = Arrow3D(xt, yt, zt, mutation_scale=20, 
                lw=0.8,
                arrowstyle=style,
                color='k',  # #04617d
                fc='#04617d')
        ax.add_artist(a)
        ax.plot([xc], [yc], [z0], "ok", markersize=3, zorder=zorder)
    else:
        ax.plot(x, y, c=cArr, lw=1, zorder=zorder)  # graficando circulo
        xt, yt = [xc, x[-1]], [yc, y0]
        ax.arrow(xc, yc, dx=xc-x0, dy=yc-y0,
                lw=0.8,
                color='k',  # #04617d
                fc='#04617d',
                length_includes_head=True,
                head_width=head_width,
                head_length=head_length,
                zorder=zorder)
        ax.plot([xc], [yc], "ok", markersize=3, zorder=zorder)
       

    xlast = x[-1]
    ylast = y[-1]

    l = radius * arrowheadrelativesize

    headangle = (direction + closingangle + (90 - headcorrectionangle) *
                 np.sign(closingangle))

    x = [xlast +
         l*np.cos((headangle + arrowheadopenangle)*np.pi/180),
         xlast,
         xlast +
         l*np.cos((headangle - arrowheadopenangle)*np.pi/180)]
    y = [ylast +
         aspect*l*np.sin((headangle + arrowheadopenangle)*np.pi/180),
         ylast,
         ylast +
         aspect*l*np.sin((headangle - arrowheadopenangle)*np.pi/180)]

    if Dim3:
        ax.plot(x, y, [z0]*len(x), c=cCir, zorder=zorder)  # graficando arrow
    else:
        ax.plot(x, y, c=cCir, zorder=zorder)  # graficando arrow
    
    return

# https://gist.github.com/WetHat/1d6cd0f7309535311a539b42cccca89c 
# https://github.com/matplotlib/matplotlib/issues/21688
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)

####################
# FONDO
####################
def system(r, yV, arg):
    """
    Sistema de ecuaciones
    [phi, phi', u, u'] -> [p0, p1, u0, u1]
    
    Linealmente Polarizada: gamma=lambda2=0 -> se traduce a que LambT=lambda1
    Circularmente Polarizada: gamma=0 -> se traduce a que LambT=lambda1+lambda2
    Radialmente Polarizada: lambda2=0 -> se traduce a que LambT=lambda1
    """
    p0, p1, u0, u1 = yV
    LambT, gamma = arg

    if np.abs(p0)>80:
        #print('datos = ', yV)
        #print('lambda, gamma = ', arg)
        #sys.exit('El perfil se indeterminó')
        f0, f1, f2, f3 = 0, 0, 0, 0
    elif r > 0:
        f0 = p1
        f1 = LambT*p0**3*r**(2*gamma)-2*(1+gamma)*p1/r-u0*p0
        f2 = u1
        f3 = -r**(2*gamma)*p0**2-2*u1/r
    else:
        f0 = p1
        f1 = (LambT*p0**3*r**(2*gamma)-u0*p0)/(2*gamma+3)
        f2 = u1
        f3 = -r**(2*gamma)*p0**2/3
        
    return [f0, f1, f2, f3]

def energ(r, sig, gamma, V0):
    """
    Energia
    """
    sigF = interp1d(r, sig, kind='quadratic')
    Af = lambda r: r**(2*gamma+1)*sigF(r)**2
    Bf = lambda r: r**(2*(gamma+1))*sigF(r)**2

    rmin = r[0]
    rfin = r[-1]

    En = V0 - quad(Af, rmin, rfin)[0]  # energía: (2c^2 m)/Lambda  -> Lambda=4pi m^3/Mp^2
    Mas = quad(Bf, rmin, rfin)[0]  # masa: c*hb/(G*m*Lambda^(1/2))
    return En, Mas

def Freq_solveG(f_max, f_min, LambT, gamma, rmax_, rmin_, nodos, u0=1.0, df0=0, du0=0,
                met='RK45', Rtol=1e-09, Atol=1e-10):
    """
    SHOOTING PARA ENCONTRAR N nodos
    Orden de las variables U = w, dw, phi, dphi
    """
    print('Finding a profile with ', nodos, 'nodes')
    # IMPORTANT: it is not possible to find two event at same time
    # Events
    arg = [LambT, gamma]
    def Sig(r, U, arg): return U[0]
    def dSig(r, U, arg): return U[1]
    Sig.direction = 0
    dSig.direction = 0
    while True:
        f0_ = (f_max+f_min)/2
        U0 = [f0_, df0, u0, du0]
        sol_ = solve_ivp(system, [rmin_, rmax_], U0, events=(Sig, dSig),
                         args=(arg,), method=met,  rtol=Rtol, atol=Atol)
                          # 'DOP853''LSODA'
        #print(f0_)
        if sol_.t_events[1].size == nodos+1 and sol_.t_events[0].size == nodos:
            print('Found', f0_)
            return f0_, rmax_, sol_.t_events[0]
        elif sol_.t_events[1].size > nodos+1:  # una vez por nodo
            if sol_.t_events[0].size > nodos:  # dos veces por nodo
                f_min = f0_
                rTemp_ = sol_.t_events[0][-1]
            else:  # si pasa por cero más veces que 2*nodos se aumenta la w, sino se disminuye
                f_max = f0_
                rTemp_ = sol_.t_events[1][-1]
        elif sol_.t_events[1].size <= nodos+1:
            if sol_.t_events[0].size > nodos:  # dos veces por nodo
                f_min = f0_
                rTemp_ = sol_.t_events[0][-1]
            else:
                f_max = f0_
                rTemp_ = sol_.t_events[1][-1]

        # checking the lim freq.
        if abs((f_max-f_min)/2) <= 1e-15:
            print('Maxima precisión alcanzada', f0_, 'radio', rTemp_)
            return f0_, rTemp_, sol_.t_events[0]

def Freq_solveG2(f0, u_max, u_min, LambT, gamma, rmax_, rmin_, nodos, df0=0, du0=0,
                met='RK45', Rtol=1e-09, Atol=1e-10):
    """
    Orden de las variables U = w, dw, phi, dphi
    """
    
    print('Finding a profile with ', nodos, 'nodes')
    # IMPORTANT: it is not possible to find two event at same time
    # Events
    arg = [LambT, gamma]
    def Sig(r, U, arg): return U[0]
    def dSig(r, U, arg): return U[1]
    Sig.direction = 0
    dSig.direction = 0
    while True:
        u0_ = (u_max+u_min)/2
        U0 = [f0, df0, u0_, du0]
        #print('Test con U0 = ', u0_)
        sol_ = solve_ivp(system, [rmin_, rmax_], U0, events=(Sig, dSig),
                         args=(arg,), method=met,  rtol=Rtol, atol=Atol)
                          # 'DOP853''LSODA'
        #print(u0_, abs((u_max-u_min)/2))
        if sol_.t_events[1].size == nodos+1 and sol_.t_events[0].size == nodos:
            print('Found', u0_)
            return u0_, rmax_, sol_.t_events[0]
        elif sol_.t_events[1].size > nodos+1:  # una vez por nodo
            if sol_.t_events[0].size > nodos:  # dos veces por nodo
                u_max = u0_
                rTemp_ = sol_.t_events[0][-1]
            else:  # si pasa por cero más veces que 2*nodos se aumenta la w, sino se disminuye
                u_min = u0_
                rTemp_ = sol_.t_events[1][-1]
        elif sol_.t_events[1].size <= nodos+1:
            if sol_.t_events[0].size > nodos:  # dos veces por nodo
                u_max = u0_
                rTemp_ = sol_.t_events[0][-1]
            else:
                u_min = u0_
                rTemp_ = sol_.t_events[1][-1]

        # checking the lim freq.
        if abs((u_max-u_min)/2) <= 1e-14: #1e-14
            print('Maxima precisión alcanzada', u0_, 'radio', rTemp_)
            return u0_, rTemp_, sol_.t_events[0]


def profilesFromSolut(datos, rmin=0, Nptos=2000, inf=False):
    """
    Usando una solución
    """
    
    f0, rTemp, gamma, LambT, nodos, posNodos, met, Rtol, Atol, U0 = datos
    
    # boundary conditions
    V0 = [f0, 0., U0, 0.]  # sigma, dsigma, u, du
    rspan = np.linspace(rmin, rTemp, Nptos)
    arg = [LambT, gamma]

    sol2 = solve_ivp(system, [rmin, rTemp], V0, t_eval=rspan,
                     args=(arg,), method=met, rtol=Rtol, atol=Atol)

    Ec = sol2.y[2][-1]  # energía u = E - Uf
    #masa = -(sol2.y[2][-1]-Ec)*sol2.t[-1]  # M = -Uf*r
    
    # calculando energía y masa por la integral
    En, Mas = energ(sol2.t, sol2.y[0], gamma, U0) 

    if inf:
        print(r'masa=', Mas)
        print('')
        print(r'energía= ', Ec, r'energíaInt= ', En)
        print('')
    
    return En, Mas, sol2.t, sol2.y[0], sol2.y[1], sol2.y[2], sol2.y[3],\
           posNodos, LambT, gamma
           
           
def fondo(soluciones_Fondo, gamma, rtake=-160):
    """
    Obteniendo la configuración del fondo
    """
    # Fondo
    s0 = soluciones_Fondo[0]
    r0M = soluciones_Fondo[1]
    Ext = (s0*r0M)+7000
    Np = int(Ext/2)

    # Resolviendo
    en, Mas, rD, sD, dsD, uD, duD, cer0, LamV = profilesFromSolut(soluciones_Fondo) 
    
    # Extendiendo
    rDnew, sDnew, dsDnew, uDnew, duDnew, datosEquiv = extend(gamma, rD[:rtake], sD[:rtake], dsD[:rtake], uD[:rtake], duD[:rtake],
                                                                Ext, Np, inf=False)

    # interpolación de los datos
    fsN = interp1d(rDnew, sDnew, kind='quadratic') # quadratic
    fdsN = interp1d(rDnew, dsDnew, kind='linear')
    fuN = interp1d(rDnew, uDnew, kind='quadratic')
    fduN = interp1d(rDnew, duDnew, kind='quadratic')
    return  fsN, fdsN, fuN, fduN, rDnew[-1]

def extend(gamma, rD, sD, dsD, uD, duD, Ext, Np=1000, inf=False, ptos=400):
    """
    Extendiendo solución del fondo
    """
    # Parámetros
    def parametrosS(r, S):
        yr1, yr2 = S[-2], S[-1]
        r1, r2 = r[-2], r[-1]

        k = np.real(np.log(np.abs(yr1*r1/(yr2*(r2)))))
        s = np.exp(-k*r1)/r1
        C = yr1/s
        return C, k

    #def parametrosS2(r, S, En, M, ptos):
    #    def expDec(x, c1):
    #        k = np.sqrt(-En)
    #        sig = c1*np.exp(-k*x)/x**(1-M/(2*k))
    #        return sig

    #    popt, pcov = curve_fit(expDec, r[-ptos:], S[-ptos:])
    #    return popt

    # funciones asíntóticas
    def sigm(r, C, k):
        y = C*np.exp(-k*r)/r
        dy = -(C*np.exp(-k*r)*(1+k*r))/r**2
        return y, dy

    #def sigm2(r, C, En, M):
    #    k = np.sqrt(-En)
    #    y = C*np.exp(-k*r)/r**(1-M/(2*k))
    #    dy = C*np.exp(-k*r)*r**(-2+M/(2*k))*(M-2*k*(1+k*r))/(2*k)
    #    return y, dy

    def Up(r, A, B):
        y = A+B/r
        dy = -B/r**2
        return y, dy

    rad = np.linspace(rD[-1], rD[-1]+Ext, Np)

    # calculando parámetros
    En, Mas = energ(rD, sD, gamma, uD[0])
    Ap, k = parametrosS(rD, sD)
    #Ap = parametrosS2(rD, sD, En, Mas, ptos=ptos)
    
    # uniendo datos
    sExt, dsExt = sigm(rad, Ap, k)
    #sExt, dsExt = sigm2(rad, Ap, En, Mas)
    uExt, duExt = Up(rad, En, Mas)

    rDnew = np.concatenate((rD[:-1], rad), axis=None)
    sDnew = np.concatenate((sD[:-1], sExt), axis=None)
    dsDnew = np.concatenate((dsD[:-1], dsExt), axis=None)
    uDnew = np.concatenate((uD[:-1], uExt), axis=None)
    duDnew = np.concatenate((duD[:-1], duExt), axis=None)

    fsN = interp1d(rDnew, sDnew, kind='quadratic') # quadratic
    fprof = lambda x: x**2*fsN(x)**2
    masa = quad(fprof, rDnew[0], rDnew[-1])[0]
    
    # checking
    if inf:
        print('checking ')
        print('Energia: ', En, ' ', uExt[-1]) #, ' ', k**2)
        print('Masa: ', Mas,  ' ', masa)

    return rDnew, sDnew, dsDnew, uDnew, duDnew, [masa, En, sD[0]]