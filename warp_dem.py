import os
import numpy as np
import argparse
import warp as wp
import warp.render

wp.init()


@wp.func
def contact_force(n: wp.vec3, v: wp.vec3, c: float, k_n: float, k_d: float, k_f: float, k_mu: float):
    vn = wp.dot(n, v)
    jn = c * k_n
    jd = min(vn, 0.0) * k_d

    # contact force
    fn = jn + jd

    # friction force
    vt = v - n * vn
    vs = wp.length(vt)

    if vs > 0.0:
        vt = vt / vs

    # Coulomb condition
    ft = wp.min(vs * k_f, k_mu * wp.abs(fn))

    # total force
    return -n * fn - vt * ft

    

@wp.kernel
def apply_forces(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    particle_f: wp.array(dtype=wp.vec3),
    particle_r: wp.array(dtype=wp.float32),
    #radius: float,
    k_contact: float,
    k_damp: float,
    k_friction: float,
    k_mu: float,
    bc_right: float,
):
    tid = wp.tid()

    # order threads by cell
    i = wp.hash_grid_point_id(grid, tid)

    x = particle_x[i]
    v = particle_v[i]
    r = particle_r[i]
    f = wp.vec3()

    # bc condition
    n = wp.vec3(0.0, 1.0, 0.0)
    c = wp.dot(n, x) - r
    if c < 0: #bottom
        f = f + contact_force(n, v, c, k_contact, k_damp, 100.0, 0.5)
    n = wp.vec3(1.0, 0.0, 0.0)
    c = wp.dot(n, x) - r
    if c < 0: #left
        f = f + contact_force(n, v, c, k_contact, k_damp, 100.0, 0.5)
    n = wp.vec3(-1.0, 0.0, 0.0)
    c = bc_right - wp.dot(-n, x) - r
    if c < 0: #right
        f = f + contact_force(n, v, c, k_contact, k_damp, 100.0, 0.5)

    # particle contact
    neighbors = wp.hash_grid_query(grid, x, r * 5.0)

    for index in neighbors:
        if index != i:
            # compute distance to point
            n = x - particle_x[index]
            d = wp.length(n)
            #err = d - r * 2.0
            err = d - r - particle_r[index]

            if err < 0: # in contact
                n = n / d
                vrel = v - particle_v[index]

                f = f + contact_force(n, vrel, err, k_contact, k_damp, k_friction, k_mu)

    particle_f[i] = f


@wp.kernel
def integrate(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    f: wp.array(dtype=wp.vec3),
    gravity: wp.vec3,
    dt: float,
    inv_mass: float,
):
    tid = wp.tid()

    v_new = v[tid] + f[tid] * inv_mass * dt + gravity * dt
    x_new = x[tid] + v_new * dt

    v[tid] = v_new
    x[tid] = x_new


class Example:
    def __init__(self, stage):
        self.scale = 0.1/0.003
        self.frame_dt = 1.0 / 60
        self.frame_count = 400

        self.sim_substeps = 64
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_steps = self.frame_count * self.sim_substeps
        self.sim_time = 0.0

        self.max_radius = 0.1/self.scale
        self.min_radius = 0.05/self.scale

        self.bc_right = 1.0
        self.k_contact = 8000.0
        self.k_damp = 2.0
        self.k_friction = 1.0
        self.k_mu = 0.5  # for cohesive materials
        
        self.inv_mass = 64.0
        if args.render:
            self.renderer = wp.render.UsdRenderer(stage)
            self.renderer.render_ground()

        self.grid = wp.HashGrid(128, 128, 128)
        self.grid_cell_size = self.max_radius * 5.0
        # loading from files
        '''position_data = np.load('positions.npy')[0]  # first frame
        self.points = np.zeros((4096, 3))
        self.points[:,:2] = position_data
        radius = np.load('r.npy')'''
        
        self.points, radius = self.particle_random_size(64, 64, 1, (0.1/self.scale, 5.0/self.scale, 0.0), self.max_radius, self.min_radius, jitter=0.5)
        self.radius = wp.array(radius, dtype=wp.float32)
        #self.points = self.particle_grid(64, 64, 1, (0.1, 5.0, 0.0), 0.1, 0.01)
        self.x = wp.array(self.points, dtype=wp.vec3)
        self.v = wp.array(np.ones([len(self.x), 3]) * np.array([0.0, 0.0, 0.0]), dtype=wp.vec3)
        self.f = wp.zeros_like(self.v)

        self.use_graph = wp.get_device().is_cuda

        if self.use_graph:
            wp.capture_begin()

            for _ in range(self.sim_substeps):
                with wp.ScopedTimer("forces", active=False):
                    wp.launch(
                        kernel=apply_forces,
                        dim=len(self.x),
                        inputs=[
                            self.grid.id,
                            self.x,
                            self.v,
                            self.f,
                            self.radius,
                            self.k_contact,
                            self.k_damp,
                            self.k_friction,
                            self.k_mu,
                            self.bc_right,
                        ],
                    )
                    wp.launch(
                        kernel=integrate,
                        dim=len(self.x),
                        inputs=[self.x, self.v, self.f, (0.0, -9.8, 0.0), self.sim_dt, self.inv_mass],
                    )

            self.graph = wp.capture_end()

    def update(self):
        with wp.ScopedTimer("simulate", active=False):
            if self.use_graph:
                with wp.ScopedTimer("grid build", active=False):
                    self.grid.build(self.x, self.grid_cell_size)

                with wp.ScopedTimer("solve", active=False):
                    wp.capture_launch(self.graph)

                self.sim_time += self.frame_dt

            else:
                with wp.ScopedTimer("grid build", active=False):
                    self.grid.build(self.x, self.grid_cell_size)

                with wp.ScopedTimer("solve", active=False):
                    for _ in range(self.sim_substeps):
                        wp.launch(
                            kernel=apply_forces,
                            dim=len(self.x),
                            inputs=[
                                self.grid.id,
                                self.x,
                                self.v,
                                self.f,
                                self.radius,
                                self.k_contact,
                                self.k_damp,
                                self.k_friction,
                                self.k_mu,
                                self.bc_right,
                            ],
                        )
                        wp.launch(
                            kernel=integrate,
                            dim=len(self.x),
                            inputs=[self.x, self.v, self.f, (0.0, -9.8, 0.0), self.sim_dt, self.inv_mass],
                        )
                        self.sim_time += self.sim_dt

    def render(self, is_live=False):
        with wp.ScopedTimer("render", active=True):
            time = 0.0 if is_live else self.sim_time

            self.renderer.begin_frame(time)
            self.renderer.render_points(points=self.x.numpy(), radius=self.radius.numpy(), name="points")
            self.renderer.end_frame()

    # creates a grid of particles
    def particle_grid(self, dim_x, dim_y, dim_z, lower, radius, jitter):
        points = np.meshgrid(np.linspace(0, dim_x, dim_x), np.linspace(0, dim_y, dim_y), np.linspace(0, dim_z, dim_z))
        points_t = np.array((points[0], points[1], points[2])).T * radius * 2.0 + np.array(lower)
        # apply noise on x and y coordinate
        noise = np.random.rand(*points_t.shape) * radius * jitter
        noise[:,:,:,2] = 0
        points_t = points_t + noise

        return points_t.reshape((-1, 3))
    # create a grid of particles with random size
    def particle_random_size(self, dim_x, dim_y, dim_z, lower, max_radius, min_radius, jitter=0.5):
        points = np.meshgrid(np.linspace(0, dim_x, dim_x), np.linspace(0, dim_y, dim_y), np.linspace(0, dim_z, dim_z))
        points_t = np.array((points[0], points[1], points[2])).T * max_radius * 2.0 + np.array(lower)
        radius = np.random.rand(dim_x*dim_y*dim_z) * (max_radius - min_radius) + min_radius
        noise =  np.multiply(max_radius-np.vstack((radius,radius,radius)).T, np.random.rand(len(radius),3))
        noise[:,2] = 0
        points_t = points_t.reshape(-1,3) + noise
        return points_t, radius

if __name__ == "__main__":
    positions = [] # for saving positions
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true", help="Render the simulation")
    args = parser.parse_args()
    stage_path = os.path.join(os.path.dirname('./'), "outputs/example_dem.usd")

    example = Example(stage_path)
    if args.render:
        for i in range(example.frame_count):
            example.update()
            example.render()
            positions.append(example.x.numpy()) # append position to list
        example.renderer.save()
        positions_np = np.array(positions) # list to numpy array
        np.save('positions_warp.npy', positions_np) # save positions
    else:
        for i in range(example.frame_count):
            example.update()
    