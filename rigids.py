import pygame as pg
import numpy as np
import random
import math

pg.init()

screen_width = 1080
screen_height = 640
screen = pg.display.set_mode([1080, 640])


n_rigid = 3
density = np.ones(n_rigid)
M = np.ones(n_rigid)
I = np.ones(n_rigid)
X = np.zeros([n_rigid, 2])
V = np.zeros([n_rigid, 2])
F = np.zeros([n_rigid, 2])
W = np.zeros(n_rigid)
angle = np.zeros(n_rigid)
tau = np.zeros(n_rigid)

rigid_shape = [[0.1, 0.3], [0.1, 0.1], [0.1, 0.2]]
rigid_type = [0, 1, 2] # 0=rect, 1=circle, 2=triangle

grav = np.array([0, -20.0])
dt = 1e-4
enable_force = False
force_center = np.zeros(2)
force_coef = 100

# helper functions
def rotation2d(_angle):
    c, s = math.cos(_angle), math.sin(_angle)
    return np.array([[c, -s],[s, c]])

def coord_to_pixel(pos):
    ratio = min(screen_width, screen_height)
    return (screen_width // 2 + int(pos[0]*ratio), screen_height // 2 - int(pos[1]*ratio))

def pixel_to_coord(pixel):
    ratio = min(screen_width, screen_height) 
    return np.array([(pixel[0] - screen_width // 2) / ratio, (screen_height // 2 - pixel[1]) / ratio])

def to_pixel(x):
    ratio = min(screen_width, screen_height)
    return ratio * x


# core function
def init(case=0):
    global n_rigid, rigid_shape, rigid_type, density, M, I, X, V, F, W, angle, tau
    if case == 0:
        n_rigid = 10
        rigid_shape = [[0.1, 0.3], [0.05, 0.05], [0.05, 0.2], [0.05, 0.15], [0.05, 0.2], [0.05, 0.15], [0.05, 0.2], [0.05, 0.15], [0.05, 0.2], [0.05, 0.15]]
        rigid_type = [0, 1, 0, 2, 0, 2, 0, 2, 0, 2]
        density = np.ones(n_rigid)
        M = np.ones(n_rigid)
        I = np.ones(n_rigid)
        X = np.zeros([n_rigid, 2])
        V = np.zeros([n_rigid, 2])
        F = np.zeros([n_rigid, 2])
        W = np.zeros(n_rigid)
        angle = np.zeros(n_rigid)
        tau = np.zeros(n_rigid)
        for i in range(n_rigid):
            X[i] = 0.8*np.array([random.random() - 0.5, random.random() - 0.5])
            angle[i] = math.pi * random.random()
    elif case == 1:
        n_rigid = 3
        rigid_shape = [[0.1, 0.3], [0.1, 0.1], [0.1, 0.2]]
        rigid_type = [0, 1, 2]
        density = np.ones(n_rigid)
        M = np.ones(n_rigid)
        I = np.ones(n_rigid)
        X = np.zeros([n_rigid, 2])
        V = np.zeros([n_rigid, 2])
        F = np.zeros([n_rigid, 2])
        W = np.zeros(n_rigid)
        angle = np.zeros(n_rigid)
        tau = np.zeros(n_rigid)
        for i in range(n_rigid):
            X[i] = 0.8*np.array([random.random() - 0.5, random.random() - 0.5])
            angle[i] = math.pi * random.random()
    
    for i in range(n_rigid):
        if rigid_type[i] == 0:
            a, b = rigid_shape[i][0], rigid_shape[i][1]
            M[i] = density[i] * a * b
            I[i] = 1 / 12 * M[i] * (a**2 + b**2)
        elif rigid_type[i] == 1:
            r = rigid_shape[i][0]
            M[i] = density[i] * math.pi * (r**2)
            I[i] = 1 / 2 * M[i] * (r**2)
        elif rigid_type[i] == 2:
            l, h = rigid_shape[i][0], rigid_shape[i][1]
            M[i] = density[i] * l * h / 2
            I[i] = 1 / 24 * M[i] * (l**2) + 1 / 18 * M[i] * (h**2)

def apply_force(f, pos, i):
    F[i] += f
    dr = pos - X[i]
    tau[i] += dr[0]*f[1] - dr[1]*f[0]


def soft_boundary_collision():
    coef = 1e4
    beta = 1e1
    for i in range(n_rigid):
        if rigid_type[i] == 0: # rect
            for dir in [[-1, -1], [1, -1], [-1, 1], [1, 1]]:
                dr = 0.5 * rotation2d(angle[i]) @ np.array([dir[0] * rigid_shape[i][0], dir[1] * rigid_shape[i][1]])
                corner = X[i] + dr
                vel_corner = V[i] + W[i] * np.array([-dr[1], dr[0]])
                if corner[0] < -0.5:
                    apply_force([coef * (-0.5 - corner[0]) - beta * vel_corner[0], 0], corner, i)
                if corner[0] > 0.5:
                    apply_force([coef * (0.5 - corner[0]) - beta * vel_corner[0], 0], corner, i)
                if corner[1] < -0.5:
                    apply_force([0, coef * (-0.5 - corner[1]) - beta * vel_corner[1]], corner, i)
                if corner[1] > 0.5:
                    apply_force([0, coef * (0.5 - corner[1]) - beta * vel_corner[1]], corner, i)
        if rigid_type[i] == 1:
            r = rigid_shape[i][0]
            if X[i][0] + 0.5 < r:
                apply_force( [coef * (-0.5 + r - X[i][0]) - beta * V[i][0], 0], X[i], i)
            if 0.5 - X[i][0] < r:
                apply_force([coef * (0.5 - X[i][0] - r) - beta * V[i][0], 0], X[i], i)
            if X[i][1] + 0.5 < r:
                apply_force([0, coef * (-0.5 + r - X[i][1]) - beta * V[i][1]], X[i], i)
            if 0.5 - X[i][1] < r:
                apply_force([0, coef * (0.5 - X[i][1] - r) - beta * V[i][1]], X[i], i)
        if rigid_type[i] == 2:
            dh = 1 / 3 * rotation2d(angle[i]) @ np.array([0, rigid_shape[i][1]])
            dl = 0.5 * rotation2d(angle[i]) @ np.array([rigid_shape[i][0], 0])
            corners = [X[i] + 2 * dh, X[i] - dh - dl, X[i] - dh + dl]
            vel_corners = [V[i] + W[i] * 2 * np.array([-dh[1], dh[0]]), \
                V[i] + W[i] * np.array([dh[1]+dl[1], -dh[0]-dl[0]]), V[i] + W[i] * np.array([dh[1]-dl[1], -dh[0]+dl[0]])]
            for c in range(3):
                corner = corners[c]
                vel_corner = vel_corners[c]
                if corner[0] < -0.5:
                    apply_force([coef * (-0.5 - corner[0]) - beta * vel_corner[0], 0], corner, i)
                if corner[0] > 0.5:
                    apply_force([coef * (0.5 - corner[0]) - beta * vel_corner[0], 0], corner, i)
                if corner[1] < -0.5:
                    apply_force([0, coef * (-0.5 - corner[1]) - beta * vel_corner[1]], corner, i)
                if corner[1] > 0.5:
                    apply_force([0, coef * (0.5 - corner[1]) - beta * vel_corner[1]], corner, i)



def advance():
    global F, tau
    F = np.zeros([n_rigid, 2])
    tau = np.zeros(n_rigid)
    soft_boundary_collision()
    for i in range(n_rigid):
        V[i] += (F[i] / M[i] + grav + enable_force * force_coef * (force_center - X[i])) * dt
        W[i] += tau[i] / I[i] * dt
        X[i] += V[i] * dt
        angle[i] += W[i] * dt 

def draw():
    pg.draw.rect(screen, 0x458985, [coord_to_pixel([-0.5,0.5]), (to_pixel(1), to_pixel(1))], 2)
    for i in range(n_rigid):
        if rigid_type[i] == 0:
            dtl = 0.5 * rotation2d(angle[i]) @ np.array([-rigid_shape[i][0], rigid_shape[i][1]])
            dtr = 0.5 * rotation2d(angle[i]) @ np.array(rigid_shape[i])
            pg.draw.polygon(screen, 0xDBA67B, \
                [coord_to_pixel(X[i] + dtl), coord_to_pixel(X[i] - dtr), coord_to_pixel(X[i] - dtl), coord_to_pixel(X[i] + dtr)])
        elif rigid_type[i] == 1:
            pg.draw.circle(screen, 0xDFDFC9, coord_to_pixel(X[i]), to_pixel(rigid_shape[i][0]))
        elif rigid_type[i] == 2:
            dh = 1 / 3 * rotation2d(angle[i]) @ np.array([0, rigid_shape[i][1]])
            dl = 0.5 * rotation2d(angle[i]) @ np.array([rigid_shape[i][0], 0])
            pg.draw.polygon(screen, 0xA55C55, \
                [coord_to_pixel(X[i] + 2 * dh), coord_to_pixel(X[i] - dh - dl), coord_to_pixel(X[i] - dh + dl)])



# main loop
running = True
init(case=1)
while running:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False
        if event.type == pg.MOUSEMOTION:
            mouse_pos = pg.mouse.get_pos()
            force_center = pixel_to_coord(mouse_pos)
        if event.type == pg.KEYDOWN:
            if event.key == pg.K_RETURN:
                enable_force = not enable_force
    advance()
    screen.fill(0x063647)
    draw()
    pg.display.flip()
pg.quit()