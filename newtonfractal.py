import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)
WIDTH, HEIGHT = 1280, 1280
image = ti.Vector.field(n=3, shape=(WIDTH, HEIGHT), dtype=ti.f32)
roots = [ti.Vector([1, 0]), ti.Vector([-0.5, 0.86602]),
         ti.Vector([-0.5, -0.86602])]
colors = [[0.067, 0.184, 0.255], [0, 0.5, 0.5], [0.93, 0.33, 0.23]]


@ti.func
def complex_sqr(z):
    return ti.Vector([z[0]**2 - z[1]**2, z[1] * z[0] * 2])


@ti.func
def cmulti(z1, z2):
    return ti.Vector([z1[0]*z2[0]-z1[1]*z2[1], z1[0]*z2[1]+z1[1]*z2[0]])


@ti.func
def cinv(z):
    return ti.Vector([z[0] / (z[0]**2+z[1]**2), -z[1] / (z[0]**2+z[1]**2)])


@ti.func
def csq(z):
    return z[0]*z[0] + z[1]*z[1]


@ti.kernel
def draw():
    for w, h in image:
        x, y = w/WIDTH-0.5, h/HEIGHT-0.5
        c = ti.Vector([x, y])
        for i in range(200):
            p = cmulti(c-roots[0], cmulti(c-roots[1], c-roots[2]))
            dp = cmulti(c-roots[0], c-roots[1]) + \
                cmulti(c-roots[1], c-roots[2]) + \
                cmulti(c-roots[0], c-roots[2])
            c -= cmulti(p, cinv(dp))

        min_dist = csq(c - roots[0])
        image[w, h] = colors[0]
        if csq(c-roots[1]) < min_dist:
            image[w, h] = colors[1]
            min_dist = csq(c-roots[1])
        if csq(c-roots[2]) < min_dist:
            image[w, h] = colors[2]


def main():
    window = ti.ui.Window('Newton Fractal', (WIDTH, HEIGHT))
    canvas = window.get_canvas()
    while window.running:
        canvas.set_image(image)
        draw()
        window.show()


if __name__ == '__main__':
    main()
