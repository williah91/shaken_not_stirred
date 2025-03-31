import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, cm
# from scipy.ndimage import gaussian_filter
from scipy.ndimage import map_coordinates
import cmocean
from matplotlib.cm import twilight
import time


class fluidsim:
    def __init__(self, grid_size=100, domain_size=20, diffusion_coef=0.01, viscosity=0.02, dt=0.1, radius = 10):
        self.grid_size = grid_size
        self.domain_size = domain_size
        self.dx = domain_size / grid_size
        self.dy = domain_size/grid_size
        self.dt = dt
        #okay so i think the diff coef is key in our new sim, we will need to make a function for this
        self.diffusion_coef = diffusion_coef
        self.viscosity = viscosity
        self.time = 0
        self.x = np.linspace(-domain_size/2, domain_size/2, grid_size)
        self.y = np.linspace(-domain_size/2, domain_size/2, grid_size)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        #grid for vel field amd pressure
        self.u_vel = np.zeros((grid_size, grid_size))
        self.v_vel = np.zeros((grid_size, grid_size))
        self.p = np.zeros((grid_size, grid_size))
        # grid for blood conc
        self.concentration = np.zeros((grid_size, grid_size))
        # current
        self.vortex(strength=1.0)
        # boundary radius
        self.boundary_radius = radius
        print("Simulation initialized successfully.")

        #Ignore these notes if you prefer to decifer the code, but, 
        #Consider this simulation as a grid /matrix which make up the space we are talking about
        #We have a grid for velocity, pressure, and blob concentration.
        #We will be iteratively adding /changing the values of the above different matrices
        #the previous code we used mapping to like, estimate conc at non integer value grid points
        #then we movedd that conc field to the current place

        # this next function here essentially is like the old function for 
        # current/vortexes but it's a circular current that doesn't drop off 
        # with gaussian (although we could) like the other vortexes we defined.
        #This is a single vortex, at a specific point. This is like a birds eye view
        #of stirring the drink w a spoon. 
        

    def vortex(self, strength=1.0):
        #if self.time<10: #how long we stir for
            center_x, center_y = 0 , 0 #centering the vortex at the center of plot. #Can change if want
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    x, y = self.x[j], self.y[i]
                    dx, dy = x - center_x, y - center_y
                    r = np.sqrt(dx**2 + dy**2) + 0.00000000001 #run into issues here if r is 0
                    # vortex that decreases with distance
                    if r < 8:  #Okay so here is just an arbitrary radius beyond which the streamlines don't affect. It'slike the radius of the circle we spin oour spoon with
                        #we can decide what fall off measure to use, and size of vortex.
                        factor = np.exp(-r/4)  # falloff
                        self.u_vel[i, j] += -strength * dy / r * factor #so basically, we find the vector from a point in the grid to the center of a voretx and we want to
                        #alter the velocoty matrix by the strenght of vortex at that point. We take the tangent vector fir that specific point and apply that to velocity grid.
                        self.v_vel[i, j] += strength * dx / r * factor #v component has x component, think cylindrical coords theta unit vector vs r unit vector
        #How this works: we have a grid, we go through each column, and row of each column, get the coordinates for each point
        # then we find distance between that point and the center of vortex
        # then we get the euclidean distance
        # turn into a perp vector, make a fall off factor and then add that velocity field
        #else: 
            #strength = 0
            #pass
    print("Added circular current to simulation.") #Added these to ensure code was running and working
        #This below was basically sine waves throughout the medium to emulate ocean waves. 
    def add_time_varying_current(self):
        #Add a time-varying component to the current
       #Create a secondary current that varies with time
        t = self.time
        strength = 0.5 * np.sin(t * 0.2) #time variation. We could change these function, pretty arbitrary.
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x, y = self.x[j], self.y[i]
                # Add a horizontally moving component
                self.u_vel[i, j] += strength * np.cos(y + t) #here's the spatial variance
                self.v_vel[i, j] += 0.3 * strength * np.sin(x + t)
                #self.u_vel[i,j]*= np.cos(y+t)
                #self.v_vel[i,j] *= np.sin(x+t)

    def source(self, x_pos=0, y_pos=0, volumeOfBlob=15.0, std=5): #std is effectively the radius of the blob
        #so like the previous code, this is just basically getting a grid location from number we put in. centres th eplot at 0,0 
        i = int((y_pos + self.domain_size/2) / self.domain_size * self.grid_size)
        j = int((x_pos + self.domain_size/2) / self.domain_size * self.grid_size)
        i = np.clip(i, 0, self.grid_size-1) #this clip function bascially returns 0 if i is less than 0 and returns max grid size if i is greater than it.
        #it's a quikcer way of doing if i <0, i =0, elif i>grid-size-1 then ...
        j = np.clip(j, 0, self.grid_size-1)
        
        y_idx, x_idx = np.ogrid[:self.grid_size, :self.grid_size] #This is our grid, it's values are it's coordinates
        distancesq = np.sqrt((y_idx-i)**2 + (x_idx-j)**2) #euclid distance broadcasting
        self.concentration += volumeOfBlob * np.exp(-distancesq**2/(2*std**2)) # we now update the concentration grid with the values calculated by gaussian spread of blob. 
        print(f"Added blood source at position ({x_pos}, {y_pos}) with volumeOfBlob {volumeOfBlob} and radius {std}.")
#Crucial in our experiment. We need to consider dynamics. Here, we just set the velocity at the walls to equal zero
#to contain the fluid. but this doesn't work for our mixing assessment, consider bounce back
    def apply_boundary_conditions(self):
        self.u_vel[0,:] = 0
        self.u_vel[-1,:] = 0
        self.u_vel[:,0] = 0
        self.u_vel[:,-1] = 0
        self.v_vel[0,:] = 0
        self.v_vel[-1,:] = 0
        self.v_vel[:,0] = 0
        self.v_vel[:,-1] = 0

    def step(self, field):
        jgrid,igrid = np.meshgrid(np.arange(self.grid_size), np.arange(self.grid_size)) #creates a matrix for our plot
        #distance = vel . time, cells traversed is distance/cellsize 
        #basically we are calculating the distance the blood has traveled in the x and y direction
        #we are then dividing this by the cell size to get the number of cells traversed
        #we are then using map_coordinates to interpolate the concentration values at these new positions
        #it's to do with advection 
        x_start = jgrid - self.u_vel * self.dt / self.dx #we are finding where it was before the time step
        y_start = igrid - self.v_vel * self.dt / self.dy
        coords = np.vstack((y_start.ravel(), x_start.ravel())) #this is the original position of the blood, we need to collapse the matrix for map coord to work. when we stack it, we get [y...] ontop of [x...] just a formatting thing for map.
        new_advectionfactor = map_coordinates(field, coords, order=1, mode='nearest').reshape(self.grid_size, self.grid_size) #map coordinates is a method of basically for those values that are not whole integer for the grid poinst, it looks at the surrounding grid points that make up the block its in and then averages them out. 
        #self.advectedstep = new_advectionfactor * (1) #- self.degrade*self.dt)
        #I haven't advected any velocity field in here. 

        return new_advectionfactor

    def step_diffusion(self, field, diffusion_coef):
        #This step is essentially the diffusion equation, we could use numerical methods if we want
        #Currently, the is just an approximated version but using gaussian to emulate diffusion effect
        #solution to diff eq is a(x,y,t) = 1/4piDt exp(-(r^2)/4Dt)
        #variance is sqrt(2Dt) so we divide that by dx to get in units of grid cells
        # Gaussian apparently makes it more stabile


        # sigma = np.sqrt(2 * diffusion_coef * self.dt) / self.dx
        # return gaussian_filter(field, sigma=sigma)

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

        # apply boundary conditions
        laplacian = map_coordinates(laplacian, coords, order=1, mode='nearest').reshape(self.grid_size, self.grid_size) #map coordinates is a method of basically for those values that are not whole integer for the grid poinst, it looks at the surrounding grid points that make up the block its in and then averages them out. 


        return field + diffusion_coef * self.dt * laplacian
     
    def project_velocity(self):
        #Here's where we need to implement numerical methods. Essentially, 
        #we need to enforce that divu =0 which governs compression of a fluid
        #we need to make it so that the amount of volume of fluid in a specific space on grid is capped
        #this is to do with the divergence of u. from the continuity equation
        #this is the pressure component of navier stokes
        #by the end of this step, we have essentially reconstructed navier stokes by evaluating what steps are at play.
        # Reapply boundary conditions
        self.apply_boundary_conditions()

    def simulate_step(self):
        #where we bring together all the functions
        self.add_time_varying_current() #changing the vel field
        # Advect vel fields
        self.u_vel = self.step(self.u_vel) 
        self.v_vel = self.step(self.v_vel)
        # Apply viscosity
        self.u_vel = self.step_diffusion(self.u_vel, self.viscosity)
        self.v_vel = self.step_diffusion(self.v_vel, self.viscosity)
        
        # Project for incompressibility
        self.project_velocity()
        #self.step()
        # Advect conc with velocity
        self.concentration = self.step(self.concentration)

        #self.concentration = self.step(self.concentration, self.u_vel, self.v_vel)
        #self.concentration = self.step_diffusion(self.concentration,self.diffusion_coef)
        # Apply diffusion to concentration
        self.concentration = self.step_diffusion(self.concentration, self.diffusion_coef)
        # non-neg concentration
        self.concentration = np.maximum(self.concentration, 0) #I think we could mess around with the conditions here, to incorporate. A stop when a average conc is reached (mixing measure)
        self.time += self.dt

    def looptheSim(self, steps=200):
        #Run the simulation for the specified number of steps
        # Store concentration and velocity history
        conc_history = [self.concentration.copy()]
        vel_history = [(self.u_vel.copy(), self.v_vel.copy())] #These are needed for simulation to run
        for step in range(steps):
            if step % 20 == 0:
                print(f"Running step {step+1}/{steps}")            #I got this to just write this to ensure it was running. Because wasn't sure if was running. Can delete.
                print(f"Total parts in grid: {np.sum(self.concentration)}")
            self.simulate_step()
        
            if step % 5 == 0: #can be changes to whatever fram rate increment we wanna do
                conc_history.append(self.concentration.copy())
                vel_history.append((self.u_vel.copy(), self.v_vel.copy()))
        #Core of the visualisation set up. This is probably only for the web element. As not fundamentally necessary.
        return conc_history, vel_history

#No physics here, just formalities of structuring the visualisation
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
                         scale=25, width=0.002, color='white', alpha=0.999)
        
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
        plt.tight_layout()
        plt.show()
        
        return ani

    def save_animation(self, ani, filename= 'fluidsim.mp4', fps=10, save = False):
       if save:
        write = animation.FFMpegWriter(fps=fps)
        ani.save(filename, writer=write)
        print(f"Animation saved as {filename} at {fps} fps.")
       else:
        print("Animation not saved.")


if __name__ == "__main__":
    # create simulation
    sim = fluidsim(
        grid_size=100, 
        domain_size=20, 
        diffusion_coef=0.002,
        viscosity=0.02,
        dt=0.1
    )
    
    # source volumeOfBlob and radius
    sim.source(x_pos=0, y_pos=0, volumeOfBlob=15.0, std=10)
    print("Running simulation...")
    conc_history, vel_history = sim.looptheSim(steps=300)

    sim.visualisequiver(conc_history, vel_history)

    