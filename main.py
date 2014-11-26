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
    u = np.ones(nx) * rho0
    return u

# -----------------------------------------------------------------------------
# compute step size
def step():
    dt = cfl * dx / max(abs(1-k*u))
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
    return

# -----------------------------------------------------------------------------
# flux vector at a single grid point
def ee(ui):
    vi = 1 - k * ui
    ei = ui * vi
    return ei
    
# -----------------------------------------------------------------------------
# Jacobi matrix at a single grid point
def aa(ui):
    vi = 1 - k * ui
    ai = vi
    return ai

# -----------------------------------------------------------------------------
# flux vector
def flux(stage=0):
    e = np.zeros(nx-1)
    for i in range(0,nx-1):
        # Lax method
        if (method == 'lax'):
            e1 = ee(u[i])
            e2 = ee(u[i+1])
            e[i] = .5 * (e1 + e2) - .5 * dx / dt * (u[i+1] - u[i])
        # Lax-Wendroff method
        elif (method == 'lax-wendroff'):
            e1 = ee(u[i])
            e2 = ee(u[i+1])
            a1 = aa(u[i])
            a2 = aa(u[i+1])
            a  = .5 * (a1 + a2)
            e[i] = .5 * (e1 + e2) - .5 * dt / dx * a * (e2 - e1)
        # MacCormack method
        elif (method == 'maccormack'):
            if (stage == 0):
                e[i] = ee(u[i+1])
            elif (stage == 1):
                e[i] = ee(u[i])
        # Jameson 4-stage Runga-Kutta
        elif (method == 'rk4'):
            e1 = ee(u[i])
            e2 = ee(u[i+1])
            e[i] = .5 * (e1 + e2)
    return e

# -----------------------------------------------------------------------------
# residual
def residual(e):
    res = np.zeros(nx)
    for i in range(1, nx-1):
        res[i] = -(e[i] - e[i-1]) / dx
    return res

xmin = 0
xmax = 50
nx   = 201     # number of grid points

rho0 = 0.2
k    = 0.9
fr   = 0.2
cfl  = 0.5
imax = 1000
eps  = 1e-5
tmax = 20

method = 'maccormack'

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
    maxres = max(abs(res))
    # if (maxres < 1e-5): break
    time  += dt
    if (time < tmax * fr):
        color = 'r'
        u[0] = 0.
    else:
        color = 'g'
        u[0] = rho0
    u[-1] = u[-2]

    if (time > tmax): time = 0
    
    if (i == 0):
        line1, = ax.plot(x, u,'-o')
        line1.set_color(color)
        ax.set_ylim(0,1)
        fig.show()
    else:
        line1.set_ydata(u)
        line1.set_color(color)
        fig.canvas.draw()
