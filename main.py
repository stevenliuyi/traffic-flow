import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# set computational mesh
def set_mesh():
    dx = (xmax-xmin) / (nx-1.)
    x  = np.linspace(xmin, xmax, nx)
    return x, dx

# -----------------------------------------------------------------------------
# define initial condition
def ic():
    u = np.ones((lmax, nx))
    if (lmax == 1):
        u[0,:] *= rho0
    elif (lmax == 2):
        u[0,:] *= rho0
        u[1,:] *= rho0 * (1 - .9*rho0)
    return u

# -----------------------------------------------------------------------------
# compute step size
def step():
    dt = cfl * dx / maxlam(u)
    return dt

# -----------------------------------------------------------------------------
# solver
def solver():
    global u, res
    if (method == 'maccormack'):
        for stage in range(0,2):
            e   = flux(stage)
            res = residual(e)
            if (stage == 0):
                u_old = np.copy(u)
                u += dt * res
            elif (stage == 1):
                u = .5 * (u + u_old + dt * res)
    elif (method == 'rk4'):
        alpha = [1./4, 1./3, 1./2, 1.]
        u_old = np.copy(u)
        for stage in range(0,4):
            e   = flux()
            res = residual(e)
            u   = u_old + alpha[stage] * dt * res
    else:
        e   = flux()
        res = residual(e)
        u  += dt * res
        #print u[:,1:10]
    return

# -----------------------------------------------------------------------------
# compute maximum eigenvalue of Jacobi matrix
def maxlam(u):
    lam = 0.
    for i in range(0, nx):
        if (lmax == 1):
            lam = max(lam, abs(vel(u[0,i])))
        elif (lmax == 2):
            lam = max(lam, abs(u[1,i]/u[0,i])+c0)
    return lam

# -----------------------------------------------------------------------------
# model for velocity
def vel(rho):
    if (model == 'lwr-greenshield'):
        k = 0.9
        v = 1 - k * rho
    elif (model == 'lwr-greenberg'):
        vmax = 10.
        if (rho < 1/np.exp(vmax)):
            v = vmax
        else:
            v = min(vmax, np.log(1/rho))
    elif (model == 'lwr-underwood'):
        v = np.exp(-rho)
    return v

# -----------------------------------------------------------------------------
# flux vector at a single grid point
def ee(ui):
    if (lmax == 1):
        vi = vel(ui)
        ei = ui * vi
    elif (lmax == 2):
        rhoi = ui[0]
        vi   = ui[1] / ui[0]
        ei = np.array([rhoi*vi, rhoi*vi**2 + c0**2*rhoi])
    return ei
    
# -----------------------------------------------------------------------------
# Jacobi matrix at a single grid point
def aa(ui):
    vi = vel(ui)
    ai = vi
    return ai

# -----------------------------------------------------------------------------
# flux vector
def flux(stage=0):
    e = np.zeros((lmax, nx-1))
    for i in range(0,nx-1):
        # Lax method
        if (method == 'lax'):
            e1 = ee(u[:,i])
            e2 = ee(u[:,i+1])
            e[:,i] = .5 * (e1 + e2) - .5 * dx / dt * (u[:,i+1] - u[:,i])
        # Lax-Wendroff method
        elif (method == 'lax-wendroff'):
            e1 = ee(u[:,i])
            e2 = ee(u[:,i+1])
            a1 = aa(u[:,i])
            a2 = aa(u[:,i+1])
            a  = .5 * (a1 + a2)
            e[:,i] = .5 * (e1 + e2) - .5 * dt / dx * a * (e2 - e1)
        # MacCormack method
        elif (method == 'maccormack'):
            if (stage == 0):
                e[:,i] = ee(u[:,i+1])
            elif (stage == 1):
                e[:,i] = ee(u[:,i])
        # Jameson 4-stage Runga-Kutta
        elif (method == 'rk4'):
            e1 = ee(u[:,i])
            e2 = ee(u[:,i+1])
            e[:,i] = .5 * (e1 + e2)

    # artificial viscosity
    if (avmodel): e = av(e)
    return e

# -----------------------------------------------------------------------------
# source vector
def source(u):
    s = np.zeros((lmax,nx))
    for i in range(0, nx):
        rhoi   = u[0,i]
        vi     = u[1,i] / u[0,i]
        s[0,i] = 0.
        s[1,i] = rhoi * ((1 - .9 * rhoi) - vi) / 2.
    return s

# -----------------------------------------------------------------------------
# residual
def residual(e):
    res = np.zeros((lmax,nx))
    for i in range(1, nx-1):
        res[:,i] = -(e[:,i] - e[:,i-1]) / dx
    #if (lmax == 2): res += source(u)
    return res

# -----------------------------------------------------------------------------
# artificial viscosity
def av(e):
    # Von-Neumann & Ritchmyer
    lam0 = maxlam(u)
    u0   = .5
    for i in range(1,nx-2):
        du   = u[:,i+1] - u[:,i]
        d3u  = u[:,i+2] - 3*u[:,i+1] + 3*u[:,i] - u[:,i-1]
        eps2 = kappa2 * abs(du) / u0
        eps4 = kappa4
        e[:,i] -= (eps2 * du - eps4 * d3u) * lam0
    return e

# -----------------------------------------------------------------------------
# parameters
xmin = 0
xmax = 50
nx   = 101     # number of grid points

rho0 = 0.8
fr   = 0.2
cfl  = 0.5
imax = 1000
eps  = 1e-5
tmax = 20

# -----------------------------------------------------------------------------
# traffic flow model
# acceptable values: lwr-greenshield, lwr-greenberg, lwr-underwood, pw
model = 'lwr-greenshield'
model = 'pw'
c0 = .5
lmax  = 2 if (model == 'pw') else 1

# -----------------------------------------------------------------------------
# numerical methods
# acceptable values: lax, lax-wendroff, maccormack, rk4
method  = 'lax'

avmodel = False
kappa2  = 2.
kappa4  = 0.2

# grid points
(x, dx) = set_mesh()
# initial condition
u = ic()

fig = plt.figure()
ax  = fig.add_subplot(111)

time = 0
for i in range(0, imax):
    # step size
    dt = step()

    solver()

    maxres = max(abs(res[0,]))
    # if (maxres < 1e-5): break
    time  += dt
    if (time < tmax * fr):
        color = 'r'
        if (lmax == 1):
            u[0,0] = 0.
        elif (lmax == 2):
            u[0,0] = 0.01
            if (u[1,1]/u[0,1]<c0):
                u[1,0] = u[1,1]
            else:
                u[1,0] = u[0,0]*(1-.9*u[0,0])
    else:
        color = 'g'
        if (lmax == 1):
            u[0,0] = rho0
        elif (lmax == 2):
            u[0,0] = rho0
            u[1,0] = u[1,1]
    u[:,-1] = u[:,-2]

    if (time > tmax): time = 0
    
    if (i == 0):
        line1, = ax.plot(x, u[0,],'-o')
        line1.set_color(color)
        ax.set_ylim(0,1)
        fig.show()
    else:
        line1.set_ydata(u[0,])
        line1.set_color(color)
        fig.canvas.draw()
