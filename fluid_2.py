import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, cm
from scipy.ndimage import gaussian_filter
from scipy.ndimage import map_coordinates
import cmocean
from matplotlib.cm import twilight
import time

"""
Impose vel at 0,0 - in this coord system we center at 0
Rotatiom, radiat velocity is 0, tangential velocity is velocity of the glass
Impose these conditions, solve navier stokes in spherical polars 
"""

class OceanBloodSimulation:
    def __init__(self, grid_size=100, domain_size=20, diffusion_coef=0.002,
                 viscosity=0.02, dt=0.1, pressure_iterations=75,
                 mode='stirring', # 'stirring', 'shaking', 'random_eddies', 'combined_shake'
                 stir_strength=1.5,
                 shake_accel_amplitude=50.0, shake_frequency=2.0, # For 'shaking' and 'combined_shake'
                 random_amplitude=1.0,       # random_eddies and combined_shake
                 random_correlation_sigma=8.0): # random_eddies and combined_shake

        self.grid_size = grid_size
        self.domain_size = domain_size
        self.dx = domain_size / grid_size; self.dy = domain_size / grid_size
        self.dt = dt; self.diffusion_coef = diffusion_coef; self.viscosity = viscosity
        self.pressure_iterations = pressure_iterations; self.time = 0
        self.x = np.linspace(-domain_size/2, domain_size/2, grid_size)
        self.y = np.linspace(-domain_size/2, domain_size/2, grid_size)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.u_vel = np.zeros((grid_size, grid_size)); self.v_vel = np.zeros((grid_size, grid_size))
        self.concentration = np.zeros((grid_size, grid_size))

        # mode and parameters
        self.mode = mode
        self.stir_strength = stir_strength
        self.shake_accel_amplitude = shake_accel_amplitude
        self.shake_frequency = shake_frequency
        self.random_amplitude = random_amplitude
        self.random_correlation_sigma = random_correlation_sigma
        
        # mass tracker for conservation
        self.initial_mass = 0.0

        print(f"Simulation initialized in '{self.mode}' mode.")
        # ... (print parameters for the selected mode)

    def apply_vortex(self, strength): 
        #only applied if stirring switched on
        #strength depends on time
        #mask sets radius of vortex
        #factor sets strendth drop off
        #circular movement by making distance to center vector perpendicular
    
        u_vortex_temp = np.zeros((self.grid_size, self.grid_size)); v_vortex_temp = np.zeros((self.grid_size, self.grid_size))
        if self.mode != 'stirring': return u_vortex_temp, v_vortex_temp
        if self.time < 10: current_strength = strength
        elif self.time >= self.total_runtime/6 and self.time < 2*self.total_runtime/6:
            current_strength = strength * (1 - (self.time - 10) / 10) * np.exp(-(self.time - 2*self.total_runtime/6)/ self.total_runtime)
        else: current_strength = 0
        if current_strength > 1e-9:
            center_x, center_y = 0, 0
            dX = self.X - center_x
            dY = self.Y - center_y
            r = np.sqrt(dX**2 + dY**2) + 1e-10
            mask = r < self.domain_size / 2.2
            factor = np.exp(-r[mask]/4)
            u_vortex_temp[mask] = -current_strength * dY[mask] / r[mask] * factor
            v_vortex_temp[mask] = current_strength * dX[mask] / r[mask] * factor
        return u_vortex_temp, v_vortex_temp

    def source(self, x_pos=0, y_pos=0, volumeOfBlob=15.0, std=5):
        #translate positional input into indices of the grid
        #ensures source is within grid
        #makes grid of indices
        #finds distance of each matrices point to the source
        #creates a gaussian distribution of the source
        i = int((y_pos + self.domain_size/2) / self.domain_size * self.grid_size)
        j = int((x_pos + self.domain_size/2) / self.domain_size * self.grid_size)
        i = np.clip(i, 0, self.grid_size-1); j = np.clip(j, 0, self.grid_size-1)
        y_idx, x_idx = np.ogrid[:self.grid_size, :self.grid_size]; distancesq = (y_idx-i)**2 + (x_idx-j)**2
        std_grid = std / self.dx if self.dx > 0 else std
        self.concentration += volumeOfBlob * np.exp(-distancesq / (2 * max(std_grid, 1e-6)**2))
        
        #calculate the initial total mass
        self.initial_mass = np.sum(self.concentration) * self.dx * self.dy
        print(f"Added source: peak={volumeOfBlob:.2f}, std={std:.2f} (grid std {std_grid:.2f}).")
        print(f"Initial total mass: {self.initial_mass:.4f}")

    #sets velocity at edges to 0, no-slip boundary condition
    def apply_boundary_conditions(self, u, v):
        u[0,:] = 0; u[-1,:] = 0; u[:,0] = 0; u[:,-1] = 0
        v[0,:] = 0; v[-1,:] = 0; v[:,0] = 0; v[:,-1] = 0

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x, y = self.x[j], self.y[i]

                if np.sqrt(x**2 + y**2) >= self.domain_size / 2.2:
                    u[i, j] = 0
                    v[i, j] = 0
                    self.concentration[i,j] = 0
        
        return u, v

    def step_advection(self, field, u, v):
        jgrid, igrid = np.meshgrid(np.arange(self.grid_size), np.arange(self.grid_size))
        x_start = jgrid - u * self.dt / self.dx; y_start = igrid - v * self.dt / self.dy
        coords = np.vstack((y_start.ravel(), x_start.ravel()))
        advected_field = map_coordinates(field, coords, order=1, mode='reflect').reshape(self.grid_size, self.grid_size)
        return advected_field

    def step_diffusion(self, field, diffusion_coef):
        # if diffusion_coef > 0 and self.dt > 0:
        #     sigma = np.sqrt(2 * diffusion_coef * self.dt) / self.dx
        #     if sigma > 1e-9: return gaussian_filter(field, sigma=sigma, mode='reflect')

        rolled_up = np.roll(field, 1, axis=0)
        rolled_down = np.roll(field, -1, axis=0)
        rolled_left = np.roll(field, 1, axis=1)
        rolled_right = np.roll(field, -1, axis=1)

        # Apply boundary conditions to prevent looping
        rolled_up[0, :] = 0
        rolled_down[-1, :] = 0
        rolled_left[:, 0] = 0
        rolled_right[:, -1] = 0

        laplacian = (
            rolled_up + rolled_down +
            rolled_left + rolled_right -
            4 * field
        ) / self.dx**2
        
        return field + diffusion_coef * self.dt * laplacian

    def _calculate_divergence(self, u, v):
        #central difference for divergence
        #Neumann boundary conditions for no flow across boundary
        dudx=np.zeros_like(u); dvdy=np.zeros_like(v)
        #finite differences, looping over all rows and columns except boundary
        #for loop done with square brackets
        #u[1:-1,2:] means take all interior points but look one to the right
        #here is a whole sub-grid, Subtract this other sub-grid. assign the result into another sub-grid.
        dudx[1:-1, 1:-1]=(u[1:-1, 2:] - u[1:-1, :-2])/(2*self.dx)
        dvdy[1:-1, 1:-1]=(v[2:, 1:-1] - v[:-2, 1:-1])/(2*self.dy)
        #total divergence
        div=dudx + dvdy
        #Neumann boundary conditions for no flow across boundary
        div[0,:]=div[1,:]; div[-1,:]=div[-2,:]; div[:,0]=div[:,1]; div[:,-1]=div[:,-2]; return div

    def _solve_pressure_poisson(self, rhs):
        #pressure field of 0s
        p=np.zeros_like(rhs) #rhs is right hand side of poission eq, lap p = div/dt
        #update pressure field
        p_new=np.zeros_like(rhs)

        dx2=self.dx**2; dy2=self.dy**2 #squared of grid cell dx
        denom=2*(dx2 + dy2) #denominator for finite differences
        if denom==0: return p
        #calculate pressure field and adjusts it to cancel out divergence
        # we want div u = 0
        #currently not = 0
        #lap p = div u / dt
        #use this to correct the velocity using a gaus-seidel method 
        #unew = u - tgradp
        for _ in range(self.pressure_iterations):
            p_new[1:-1,1:-1]=((p[1:-1,2:]+p[1:-1,:-2])*dy2+(p[2:,1:-1]+p[:-2,1:-1])*dx2-rhs[1:-1,1:-1]*dx2*dy2)/denom
            #Neumann boundary conditions for no flow across boundary
            p_new[0,:]=p_new[1,:]; p_new[-1,:]=p_new[-2,:]; p_new[:,0]=p_new[:,1]; p_new[:,-1]=p_new[:,-2]; p=p_new.copy()
        return p

    def _calculate_gradient(self, p):
        #gradient matrix
        grad_p_x=np.zeros_like(p); grad_p_y=np.zeros_like(p)
        #finding pressure gradient using finite differences
        grad_p_x[1:-1,1:-1]=(p[1:-1,2:]-p[1:-1,:-2])/(2*self.dx); grad_p_y[1:-1,1:-1]=(p[2:,1:-1]-p[:-2,1:-1])/(2*self.dy)
        #copies last row and column from the one next to it
        #sets gradient = 0 at the boundary because of neumann conditions
        grad_p_x[0,:]=grad_p_x[1,:]; grad_p_x[-1,:]=grad_p_x[-2,:]; grad_p_x[:,0]=grad_p_x[:,1]; grad_p_x[:,-1]=grad_p_x[:,-2]
        grad_p_y[0,:]=grad_p_y[1,:]; grad_p_y[-1,:]=grad_p_y[-2,:]; grad_p_y[:,0]=grad_p_y[:,1]; grad_p_y[:,-1]=grad_p_y[:,-2]; return grad_p_x, grad_p_y

    def project_velocity(self, u, v):
        #unew = u -tgradp
        div=self._calculate_divergence(u,v)
        rhs=div/self.dt if self.dt!=0 else div
        pressure=self._solve_pressure_poisson(rhs)
        grad_p_x,grad_p_y=self._calculate_gradient(pressure); 
        u_corrected=u-self.dt*grad_p_x; 
        v_corrected=v-self.dt*grad_p_y
        u_final,v_final=self.apply_boundary_conditions(u_corrected,v_corrected); return u_final,v_final
    
    def step_function_decay(self):

        if self.time < 30:
            return 1
        else: # self.time >= 30 and self.time < 55:
            return np.exp(-(self.time)/20)
        #else:
            #return 0

    def simulate_step(self):
        #velocity from the previous step
        u_old = self.u_vel.copy()
        v_old = self.v_vel.copy()

        #advect
        u_advected = self.step_advection(u_old, u_old, v_old)
        v_advected = self.step_advection(v_old, u_old, v_old)

        #Diffusion
        u_diffused = self.step_diffusion(u_advected, self.viscosity)
        v_diffused = self.step_diffusion(v_advected, self.viscosity)

        #Force
        u_forced = u_diffused # Start with diffused velocity
        v_forced = v_diffused

        # forces based on mode 
        a_shake = 0.0 # Oscillating vertical acceleration component
        u_rand = 0.0  # Random horizontal velocity component
        v_rand = 0.0  # Random vertical velocity component

        if self.mode == 'stirring':
            u_vortex, v_vortex = self.apply_vortex(strength=self.stir_strength)
            u_forced += u_vortex
            v_forced += v_vortex

        elif self.mode == 'shaking' or self.mode == 'combined_shake':
            # Calculate oscillating acceleration (used in both modes)
            omega = 2 * np.pi * self.shake_frequency
            a_shake_base = self.shake_accel_amplitude * np.sin(omega * self.time) #alternating up and down
            damping_cells = 5 # Damp near boundaries
            #create a matrix and imposing that on our other matrix, to dampen the shake near the boundaries
            if damping_cells > 0 and 2 * damping_cells < self.grid_size:
                damping_profile = np.ones(self.grid_size)
                damping_profile[:damping_cells] = np.linspace(0, 1, damping_cells)
                damping_profile[-damping_cells:] = np.linspace(1, 0, damping_cells)
                a_shake = a_shake_base * damping_profile[:, np.newaxis] #this just creates an array of equal amp horizontally, and dampended vertically
            else: a_shake = a_shake_base

        if self.mode == 'random_eddies' or self.mode == 'combined_shake':
             # Calculate random velocity field (used in both modes)
            noise_u = np.random.randn(self.grid_size, self.grid_size)
            noise_v = np.random.randn(self.grid_size, self.grid_size)
            sigma = self.random_correlation_sigma
            if sigma > 0:
                filtered_u = gaussian_filter(noise_u, sigma=sigma, mode='wrap')
                filtered_v = gaussian_filter(noise_v, sigma=sigma, mode='wrap')
            else: filtered_u = noise_u; filtered_v = noise_v
            std_u = np.std(filtered_u); std_v = np.std(filtered_v)
            norm_u = filtered_u / std_u if std_u > 1e-9 else filtered_u
            norm_v = filtered_v / std_v if std_v > 1e-9 else filtered_v
            u_rand = self.random_amplitude * norm_u
            v_rand = self.random_amplitude * norm_v

        # forces/velocities
        # a_shake contributes acceleration*dt, u_rand/v_rand contribute velocity
        #print("AHHHHH")
        #print(self.step_function_decay())
        u_forced += u_rand * self.step_function_decay()
        v_forced += ( a_shake * self.dt + v_rand ) * self.step_function_decay()
        
        #boundary conditions before projection
        u_forced, v_forced = self.apply_boundary_conditions(u_forced, v_forced)
        # Projection Step
        self.u_vel, self.v_vel = self.project_velocity(u_forced, v_forced) 

        # Conc advect
        conc_advected = self.step_advection(self.concentration, self.u_vel, self.v_vel)
        self.concentration = self.step_diffusion(conc_advected, self.diffusion_coef)

        #non-negative conc
        self.concentration = np.maximum(self.concentration, 0)
        
        #mass conservation
        if self.initial_mass > 0:
            current_mass = np.sum(self.concentration) * self.dx * self.dy
            if current_mass > 0: 
                #maintain the initial mass
                correction_factor = self.initial_mass / current_mass
                self.concentration *= correction_factor
        self.time += self.dt

    def looptheSim(self, steps=200):
        self.total_runtime = steps * self.dt
        conc_history = [self.concentration.copy()]; vel_history = [(self.u_vel.copy(), self.v_vel.copy())]
        print(f"Starting simulation ({self.mode}) for {steps} steps with dt={self.dt}...")
        start_time = time.time()
        for step in range(steps):
            self.simulate_step()
            if step % 50 == 0 or step == steps - 1:
                elapsed = time.time() - start_time
                max_vel_mag = np.max(np.sqrt(self.u_vel**2 + self.v_vel**2)) if self.u_vel.size > 0 else 0
                total_conc = np.sum(self.concentration) * self.dx * self.dy if self.concentration.size > 0 else 0
                print(f"Step {step+1}/{steps}. Time: {self.time:.2f}. Max Vel: {max_vel_mag:.3f}. Total Conc: {total_conc:.4f}. Elapsed: {elapsed:.2f}s")
            if (step + 1) % 5 == 0:
                conc_history.append(self.concentration.copy()); vel_history.append((self.u_vel.copy(), self.v_vel.copy()))
        print("Self.time: ", self.time) 
        end_time = time.time(); print(f"Simulation finished in {end_time - start_time:.2f} seconds.")
        print(f"History lengths: Conc={len(conc_history)}, Vel={len(vel_history)}"); return conc_history, vel_history

    def visualisequiver(self, conc_history, vel_history):
       #quiver plot instead of streamlines
        fig, ax = plt.subplots(figsize=(10, 8))
        cmap = cm.twilight
        # vmin/vmax again
        vmax = max(np.max(frame) for frame in conc_history)
        vmin = 0
        # initial plot
        im = ax.imshow(conc_history[0], cmap=cmap, origin='lower', 
                      extent=[-self.domain_size/2, self.domain_size/2, 
                              -self.domain_size/2, self.domain_size/2],
                      vmin=vmin, vmax=vmax)
                      
        plt.colorbar(im, ax=ax, label='Concentration')
        skip = 4  # spacing between arrows #try changing this to different integer values, get quite funky quiver plots
        sample_points = self.grid_size // skip
        x_sample = np.linspace(-self.domain_size/2, self.domain_size/2, sample_points)
        y_sample = np.linspace(-self.domain_size/2, self.domain_size/2, sample_points)
        X_sample, Y_sample = np.meshgrid(x_sample, y_sample)
        
        #velocity data matches the sample grid dimensions
        indices = np.linspace(0, self.grid_size-1, sample_points).astype(int)
        u_sample = vel_history[0][0][np.ix_(indices, indices)]
        v_sample = vel_history[0][1][np.ix_(indices, indices)]
        
        # nromalising velocity arrows for better look
        magnitude = np.sqrt(u_sample**2 + v_sample**2)
        max_magnitude = np.max(magnitude) if np.max(magnitude) > 0 else 1.0
        u_norm = u_sample / max_magnitude
        v_norm = v_sample / max_magnitude
        
        quiver = ax.quiver(X_sample, Y_sample, u_norm, v_norm, 
                         scale=25, width=0.002, color='black', alpha=0.999)
        
        ax.set_xlabel('X position')
        ax.set_ylabel('Y position')
        ax.set_title('Time: 0.0')
        
        def update(frame):
            # update concentration field
            im.set_array(conc_history[frame])
            
            # update quiver plot using same indices for consistent dimensions
            u_sample = vel_history[frame][0][np.ix_(indices, indices)]
            v_sample = vel_history[frame][1][np.ix_(indices, indices)]
            
            # normalis
            magnitude = np.sqrt(u_sample**2 + v_sample**2)
            max_magnitude = np.max(magnitude) if np.max(magnitude) > 0 else 1.0
            u_norm = u_sample / max_magnitude
            v_norm = v_sample / max_magnitude
            
            quiver.set_UVC(u_norm, v_norm)
            
            ax.set_title(f'Time: {frame * 5 * self.dt:.1f}')
            
            return [im, quiver]
        
        ani = animation.FuncAnimation(fig, update, frames=len(conc_history), 
                                     interval=100, blit=True)
        # plt.tight_layout()
        # plt.show()
        plt.close()
        
        return ani

    def save_animation(self, ani, filename='ocean_blood_simulation.mp4', fps=10):
        if ani is None: print("Animation is None."); return
        try:
            print(f"Saving MP4 to {filename}..."); writer=animation.FFMpegWriter(fps=fps,metadata=dict(artist='Me'),bitrate=1800)
            ani.save(filename,writer=writer); print(f"Saved MP4.")
        except Exception as e:
            print(f"Failed MP4 save: {e}. Trying GIF...")
            try:
                writer=animation.PillowWriter(fps=fps); gif_filename=filename.replace('.mp4','.gif')
                print(f"Saving GIF to {gif_filename}..."); ani.save(gif_filename,writer=writer); print(f"Saved GIF.")
            except Exception as e2: print(f"Failed GIF save: {e2}. Animation not saved.")


if __name__ == "__main__":
    common_params = {
        'grid_size': 100, 'domain_size': 20, 'diffusion_coef': 0.002,
        'viscosity': 0.02, 'dt': 0.1, 'pressure_iterations': 75
    }
    num_steps = 600
    history_save_interval = 5 # Must match vis update time calc: time = frame * interval * dt

    # uncommenting relevant block

     # Stirring
    #current_mode_params = { 'mode': 'stirring', 'stir_strength': 1.5 }
    #output_filename = 'martini_stirring.mp4'

    # Shake
    #current_mode_params = { 'mode': 'shaking', 'shake_accel_amplitude': 50.0, 'shake_frequency': 1.5 }
    #output_filename = 'martini_oscillating_shake.mp4'

    # Random Eddies
    # current_mode_params = { 'mode': 'random_eddies', 'random_amplitude': 1.5, 'random_correlation_sigma': 8.0 }
    # output_filename = 'martini_random_eddies.mp4'

    #Combined Shake (osc + eddies)
    current_mode_params = {
        'mode': 'combined_shake',
        'shake_accel_amplitude': 50.0,   # Amplitude of oscillation
        'shake_frequency': 1.5,          # Fq of oscillation
        'random_amplitude': 0.75,         # Amp of eddies
        'random_correlation_sigma': 6.0  # Size of eddies
    }
    #output_filename = 'martini_combined_shake.mp4'

    print(f"{current_mode_params['mode'].upper()} SIMULATION ---")
    sim = OceanBloodSimulation(**common_params, **current_mode_params)
    sim.source(x_pos=0, y_pos=0, volumeOfBlob=1.0, std=2.8) # Martini ratio blob

    conc_history, vel_history = sim.looptheSim(steps=num_steps)
    print(f"Final avg concentration: {np.mean(conc_history[-1]):.6f}")
    print(f"Final max concentration: {np.max(conc_history[-1]):.6f}")
    # Do this for both the stirring run and the combined_shake run

    print("\nVisualizing Results...")
    try:
        ani = sim.visualisequiver(conc_history, vel_history)
        # sim.save_animation(ani, output_filename, fps=20)
    except Exception as e:
        print(f"Visualization error: {e}")
        import traceback; traceback.print_exc()