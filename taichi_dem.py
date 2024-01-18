import taichi as ti
import math
import os
import numpy as np
ti.init(arch=ti.gpu)
vec = ti.math.vec2

SAVE_FRAMES = False
SHOW_GUI = True
window_size = 1024  # Number of pixels of the window
dim_x = 64
dim_y = 64
gravity = -9.81
frame_dt = 1.0 / 60
frame_count = 400
substeps = 64
dt = frame_dt/substeps #1e-4  # Larger dt might lead to unstable results.

bc_right = ti.field(ti.f32, shape=()) 
domain_size = 1.0
bc_right[None] = 1.0 * domain_size
k_n = 8e3
k_d = 2.0
k_f = 1.0
k_mu = 0.5 # for cohesive materials

@ti.dataclass
class Grain:
    p: vec  # Position
    m: ti.f32  # Mass
    r: ti.f32  # Radius
    v: vec  # Velocity
    a: vec  # Acceleration
    f: vec  # Force

gf = Grain.field(shape=(dim_x*dim_y, ))
grain_r_min = 0.0015 * domain_size
grain_r_max = 0.003 * domain_size
grid_n = 128
grid_size = domain_size / grid_n  # Simulation domain of size [0, 1]
print(f"Grid size: {grid_n}x{grid_n}")

assert grain_r_max * 2 < grid_size

# create a 2D particle grid with random size
def particle_random_size(dim_x, dim_y, lower, max_radius, min_radius, jitter=0.5):
        points = np.meshgrid(np.linspace(0, dim_x, dim_x), np.linspace(0, dim_y, dim_y))
        points_t = np.array((points[0], points[1])).T * max_radius * 2.0 + np.array(lower)
        radius = np.random.rand(dim_x*dim_y) * (max_radius - min_radius) + min_radius
        noise =  np.multiply(max_radius-np.vstack((radius,radius)).T, np.random.rand(len(radius),2))
        points_t = points_t.reshape(-1,2) + noise
        return points_t.astype(np.float32), radius.astype(np.float32)

pts, rs = particle_random_size(dim_x, dim_y, [grain_r_max,5./33.3*domain_size], grain_r_max, grain_r_min, jitter=0.5)
points = ti.field(float, shape=(pts.shape[0], pts.shape[1]))
points.from_numpy(pts)
radius = ti.field(float, shape=(rs.shape[0]))
radius.from_numpy(rs)

@ti.kernel
def init():
    for i in gf:
        # Spread grains in a restricted area.
        '''l = i * grid_size
        padding = 0.001
        region_width = 1.0 - bc_right[None] - 0.003 #padding * 2
        pos = vec(l % region_width + padding + grid_size * ti.random() * 0.2,
                  l // region_width * grid_size + 0.01)
        gf[i].p = pos
        gf[i].r = ti.random() * (grain_r_max - grain_r_min) + grain_r_min'''
        gf[i].p = vec(points[i,0], points[i,1])
        gf[i].r = radius[i]
        gf[i].m = 1./64. #density * math.pi * gf[i].r**2
        gf[i].v = [0,0]
        gf[i].a = [0,0]
        gf[i].f = [0,0]

@ti.kernel
def update():
    for i in gf:
        '''a = gf[i].f / gf[i].m
        gf[i].v += (gf[i].a + a) * dt / 2.0
        gf[i].p += gf[i].v * dt + 0.5 * a * dt**2
        gf[i].a = a'''
        #gf[i].f += vec(0., gravity * gf[i].m)  # Apply gravity.
        gf[i].v += gf[i].f / gf[i].m * dt
        gf[i].p += gf[i].v * dt

@ti.func
def contact_force(n, v, c, k_n, k_d, k_f, k_mu):
    vn = ti.math.dot(n, v)
    jn = c * k_n
    jd = min(vn, 0.0) * k_d

    # contact force
    fn = jn + jd

    # friction force
    vt = v - n * vn
    vs = ti.math.length(vt)

    if vs > 0.0:
        vt = vt / vs

    # Coulomb condition
    ft = ti.min(vs * k_f, k_mu * ti.abs(fn))

    # total force
    return -n * fn - vt * ft

list_head = ti.field(dtype=ti.i32, shape=grid_n * grid_n)
list_cur = ti.field(dtype=ti.i32, shape=grid_n * grid_n)
list_tail = ti.field(dtype=ti.i32, shape=grid_n * grid_n)

grain_count = ti.field(dtype=ti.i32,
                       shape=(grid_n, grid_n),
                       name="grain_count")
column_sum = ti.field(dtype=ti.i32, shape=grid_n, name="column_sum")
prefix_sum = ti.field(dtype=ti.i32, shape=(grid_n, grid_n), name="prefix_sum")
particle_id = ti.field(dtype=ti.i32, shape=dim_x*dim_y, name="particle_id")

@ti.kernel
def contact(gf: ti.template()):
    '''
    Handle the collision between grains.
    '''
    for i in gf:
        gf[i].f = vec(0., gravity * gf[i].m)  # Apply gravity.
        # bc condition
        normal = ti.math.vec2(0.0, 1.0)
        c = ti.math.dot(normal, gf[i].p) - gf[i].r
        if c < 0: #bottom
            gf[i].f = gf[i].f + contact_force(normal, gf[i].v, c, k_n, k_d, 100.0, 0.5)
        normal = ti.math.vec2(1.0, 0.0)
        c = ti.math.dot(normal, gf[i].p) - gf[i].r
        if c < 0: #left
            gf[i].f = gf[i].f + contact_force(normal, gf[i].v, c, k_n, k_d, 100.0, 0.5)
        normal = ti.math.vec2(-1.0, 0.0)
        c = bc_right[None] - ti.math.dot(-normal, gf[i].p) - gf[i].r
        if c < 0: #right
            gf[i].f = gf[i].f + contact_force(normal, gf[i].v, c, k_n, k_d, 100.0, 0.5)
    
    grain_count.fill(0)

    for i in range(dim_x*dim_y):
        grid_idx = ti.floor(gf[i].p * grid_n, int)
        grain_count[grid_idx] += 1

    for i in range(grid_n):
        sum = 0
        for j in range(grid_n):
            sum += grain_count[i, j]
        column_sum[i] = sum

    prefix_sum[0, 0] = 0

    ti.loop_config(serialize=True)
    for i in range(1, grid_n):
        prefix_sum[i, 0] = prefix_sum[i - 1, 0] + column_sum[i - 1]

    for i in range(grid_n):
        for j in range(grid_n):
            if j == 0:
                prefix_sum[i, j] += grain_count[i, j]
            else:
                prefix_sum[i, j] = prefix_sum[i, j - 1] + grain_count[i, j]

            linear_idx = i * grid_n + j

            list_head[linear_idx] = prefix_sum[i, j] - grain_count[i, j]
            list_cur[linear_idx] = list_head[linear_idx]
            list_tail[linear_idx] = prefix_sum[i, j]

    for i in range(dim_x*dim_y):
        grid_idx = ti.floor(gf[i].p * grid_n, int)
        linear_idx = grid_idx[0] * grid_n + grid_idx[1]
        grain_location = ti.atomic_add(list_cur[linear_idx], 1)
        particle_id[grain_location] = i
    
    # Brute-force collision detection
    '''
    for i in range(dim_x*dim_y):
        for j in range(i + 1, n):
            resolve(i, j)
    '''

    # Fast collision detection
    for i in range(dim_x*dim_y):
        grid_idx = ti.floor(gf[i].p * grid_n, int)
        x_begin = max(grid_idx[0] - 1, 0)
        x_end = min(grid_idx[0] + 2, grid_n)

        y_begin = max(grid_idx[1] - 1, 0)
        y_end = min(grid_idx[1] + 2, grid_n)

        for neigh_i in range(x_begin, x_end):
            for neigh_j in range(y_begin, y_end):
                neigh_linear_idx = neigh_i * grid_n + neigh_j
                for p_idx in range(list_head[neigh_linear_idx],
                                   list_tail[neigh_linear_idx]):
                    j = particle_id[p_idx]
                    if i != j:
                        #resolve(i, j)
                        #contact_force(i, j, k_n, k_d, k_f, k_mu)
                        # compute distance to point
                        normal = gf[i].p - gf[j].p
                        d = ti.math.length(normal)
                        #err = d - r * 2.0
                        err = d - gf[i].r - gf[j].r

                        if err < 0: # in contact
                            normal = normal / d
                            vrel = gf[i].v - gf[j].v

                            gf[i].f = gf[i].f + contact_force(normal, vrel, err, k_n, k_d, k_f, k_mu)
                            #gf[j].f = gf[j].f - bc_contact(normal, vrel, err, k_n, k_d, k_f, k_mu)


init()
if SHOW_GUI:
    gui = ti.GUI('Taichi DEM', (window_size, window_size), show_gui=True)
    if SAVE_FRAMES:
        os.makedirs('output', exist_ok=True)

#while gui.running:
positions = []
for step in range(frame_count):
    for s in range(substeps):
        #apply_bc()
        contact(gf)
        update()
    if SHOW_GUI:
        if gui.get_event(ti.GUI.PRESS): 
            if gui.event.key == "r": # reset button
                init()
            
            elif gui.event.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
                break
        pos = gf.p.to_numpy()
        positions.append(pos)
        r = gf.r.to_numpy() * window_size
        #v = gf.v.to_numpy()
        #v_mean = np.mean(v,axis=0)
        #print(f"step: {step}, time: {(step+1)*frame_dt:.4f}, v: ({v_mean[1]:.3f}) v_real: {gravity*(step+1)*frame_dt:.3f}")
        gui.circles(pos, radius=r)
        if SAVE_FRAMES:
            gui.show(f'output/{step:06d}.png')
        else:
            gui.show()
    '''# Convert positions and r to numpy arrays
    positions_np = np.array(positions)
    r_np = gf.r.to_numpy()

    # Save positions and r as numpy files
    np.save('positions.npy', positions_np)
    np.save('r.npy', r_np)'''
    #step += 1
