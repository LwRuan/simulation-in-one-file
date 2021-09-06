import taichi as ti

ti.init(arch=ti.gpu)
gui = ti.GUI('mls2d',(800,600),0x112F41)

nParticle = 20
delta = 0.8/nParticle
X = ti.Vector.field(2,dtype=float,shape=nParticle)

nSample = 1000
XS = ti.Vector.field(2,dtype=float,shape=nSample)

d = 3*delta # two neighbors
@ti.func
def W(r):
    r = abs(r)/d
    w = 0.0
    if r < 1/2:
        w = 2/3-4*r**2+4*r**3
    elif r < 1:
        w = 4/3 - 4*r + 4*r**2 - 4/3*r**3
    return w

@ti.func
def P(x):
    return ti.Vector([1, x, x**2])

@ti.kernel
def mls():
    for i in range(nSample):
        x = i/(nSample-1) # material space coord
        M = ti.Matrix.zero(float, 3, 3)
        b = ti.Matrix.zero(float, 3, 2)
        for j in range(nParticle):
            z = j/(nParticle-1) # material space coord
            w = W(z-x)
            M += w*P(z-x).outer_product(P(z-x))
            b += w*P(z-x).outer_product(X[j])
        c = M.inverse() @ b
        XS[i] = c.transpose() @ P(0)
    return

@ti.kernel
def init():
    for i in range(nParticle):
        x = 0.1+i*delta
        X[i] = [x, 0.7+1.5*(x-0.5)**2-0.3+0.4*ti.random()]
    return

if __name__ == '__main__':
    init()
    mls()
    while gui.running:
        gui.circles(pos=X.to_numpy(),radius=6,color=0xEEEEF0)
        gui.lines(begin=XS.to_numpy()[:-1],end=XS.to_numpy()[1:],radius=2,color=0xED553B)
        gui.show()
