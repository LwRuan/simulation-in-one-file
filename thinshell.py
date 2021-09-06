import taichi as ti
import math

ti.init(arch=ti.gpu)

nParticle = 20
Xt = ti.Vector.field(2, float, nParticle)
Ft = ti.Vector.field(2, float, nParticle)
Vt = ti.Vector.field(2, float, nParticle)

k = 1e5
tau = 1e-2
mass = 1 / nParticle
dt = 1e-4
gravity = ti.Vector([0, -10])

L0 = 0.8 # total length
delta_xi = 1 / (nParticle - 1)
a0 = L0 * ti.Vector([1, 0])
d = 3 * delta_xi

nSample = 100
delta_xi_sample = 1/(nSample - 1)
M_inv = ti.Matrix.field(3, 3, float, shape=nSample)
c = ti.Matrix.field(3, 2, float, shape=nSample) # [x a 0.5*a']^T
Xs = ti.Vector.field(2, float, shape=nSample)
L0_smaple = L0 / nSample

@ti.func
def W(r):
    r = abs(r) / d
    w = 0.0
    if r < 1/2:
        w = 2/3 - 4*r**2 + 4*r**3
    elif r < 1:
        w = 4/3 - 4*r + 4*r**2 - 4/3*r**3
    return w

@ti.func
def P(xi):
    return ti.Vector([1, xi, xi**2])

# compute M_inv and c for all samples
@ti.kernel
def compute_M_inv_c():
    for i in range(nSample):
        xi_i = i/(nSample-1)
        M = ti.Matrix.zero(float, 3, 3)
        b = ti.Matrix.zero(float, 3, 2)
        for p in range(nParticle):
            xi_p = p/(nParticle-1)
            w = W(xi_p - xi_i)
            M += w*P(xi_p - xi_i).outer_product(P(xi_p - xi_i))
            b += w*P(xi_p - xi_i).outer_product(Xt[p])
        M_inv[i] = M.inverse()
        c[i] = M_inv[i] @ b
        Xs[i] = c[i].transpose() @ P(0)
    return

@ti.kernel
def compute_c():
    for i in range(nSample):
        xi_i = i/(nSample-1)
        b = ti.Matrix.zero(float, 3, 2)
        for p in range(nParticle):
            xi_p = p/(nParticle-1)
            w = W(xi_p - xi_i)
            b += w*P(xi_p - xi_i).outer_product(Xt[p])
        c[i] = M_inv[i] @ b
        Xs[i] = c[i].transpose() @ P(0)
    return

@ti.kernel
def compute_force():
    for p in range(1, nParticle-1):
        Ft[p] = ti.Vector([0.0, 0.0])
        xi_p = p/(nParticle - 1)
        min_sample_index = max(int((xi_p - d) / delta_xi_sample) - 1, 0)
        max_sample_index = min(int((xi_p + d) / delta_xi_sample) + 1, nSample - 1)
        for i in range(min_sample_index, max_sample_index+1):
            xi_i = i/(nSample - 1)
            phi_and_grad = W(xi_p - xi_i) * M_inv[i] @ P(xi_p - xi_i) # [phi phi' 0.5*phi'']^T
            a = c[i].transpose() @ ti.Vector([0, 1, 0])
            a_grad = c[i].transpose() @ ti.Vector([0, 0, 2])
            n = ti.Matrix.rotation2d(math.pi / 2) @ a.normalized()
            alpha = (a.dot(a) - a0.dot(a0)) / 2
            beta = - a_grad.dot(n)
            alpha_grad = phi_and_grad[1]*a
            beta_grad = -2 * phi_and_grad[2] * n + phi_and_grad[1] * a_grad.dot(n) * a / (a.norm()**2) \
                - phi_and_grad[1] * ti.Matrix.rotation2d(- math.pi / 2) @ a_grad / a.norm()
            Ft[p] -= k / ((a0.dot(a0))**2) * L0_smaple * (tau * alpha * alpha_grad + tau**3 / 12 * beta * beta_grad)
    return

@ti.kernel
def advance():
    for i in range(1, nParticle-1):
        Vt[i] += dt * (Ft[i] / mass + gravity)
        Xt[i] += dt * Vt[i]
    return

@ti.kernel
def reset_particle():
    delta = L0 / (nParticle - 1)
    for i in range(nParticle):
        x = 0.1+i*delta
        #Xt[i] = [x, 0.7+1.5*(x-0.5)**2-0.3+0.02*ti.random()]
        Xt[i] = [x, 0.5]
        Ft[i] = [0.0, 0.0]
        Vt[i] = [0.0, 0.0]
    return

if __name__ == '__main__':
    gui = ti.GUI('mls2d',(2*800,2*600),0x112F41)
    reset_particle()
    compute_M_inv_c()
    pause = False
    while gui.running:
        for substep in range(30):
            if gui.get_event(ti.GUI.PRESS):
                if gui.event.key == 'r':
                    reset_particle()
                    compute_M_inv_c()
                elif gui.event.key == ti.GUI.LMB: pause = not pause
            if not pause:
                compute_force()
                advance()
                compute_c()
        gui.circles(pos=Xt.to_numpy(),radius=6,color=0xEEEEF0)
        gui.circles(pos=Xs.to_numpy(),radius=2,color=0xED553B)
        #gui.lines(begin=Xs.to_numpy()[:-1],end=Xs.to_numpy()[1:],radius=2,color=0xED553B)
        gui.show()

'''
@ti.func
def X(xi, order=0):
    M = ti.Matrix.zero(float, 3, 3)
    b = ti.Matrix.zero(float, 3, 2)
    for i in range(nParticle):
        xi_ = i / (nParticle-1)
        w = W(xi - xi_)
        M += w*P(xi_-xi).outer_product(P(xi_-xi))
        b += w*P(xi_-xi).outer_product(Xt[i])
    c = M.inverse() @ b
    res = ti.Vector.zero(float, 2)
    if order == 0:
        res = c.transpose() @ ti.Vector([1, 0, 0])
    elif order == 1:
        res = c.transpose() @ ti.Vector([0, 1, 0])
    elif order == 2:
        res = c.transpose() @ ti.Vector([0, 0, 1])
    return res

@ti.func
def N(xi):
    return ti.Matrix.rotation2d(math.pi / 2) @ X(xi).normalized()
'''