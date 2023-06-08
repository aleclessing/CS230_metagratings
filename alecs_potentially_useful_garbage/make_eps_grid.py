import numpy as np

## Code to generate random grids with more complicated patterns that we didn't have time to use

def add_shape(eps_grid, mask):

    background_type = np.random.choice(['uniform', 'gradient_sawtooth', 'egg'], p=[0.1, .3, .6])

    if background_type == 'uniform':
        background = np.random.uniform(low=1.5, high=2.5)
    else:
        xs, ys = np.meshgrid(np.arange(eps_grid.shape[1]), np.arange(eps_grid.shape[0]))

        if background_type == 'gradient_sawtooth':
            x_slope = np.random.uniform(low=-0.08, high=0.08)
            y_slope = np.random.uniform(low=-0.08, high=0.08)
            intercept = np.random.uniform(low=0, high=100)

            background = x_slope*xs + y_slope*ys +intercept

        elif background_type == 'egg':
            x_wavenums = np.random.uniform(low=-0.6, high=0.6, size=(2))
            y_wavenums = np.random.uniform(low=-0.6, high=0.6, size=(2))
            offsets = np.random.uniform(low=0, high=100, size=(2))

            background = np.sin(x_wavenums[0]*xs + y_wavenums[0]*ys + offsets[0]) + np.cos(x_wavenums[1]*xs + y_wavenums[1]*ys + offsets[1])

            amp = np.random.uniform(low=0.5, high=0.8)

            background *= amp


        low_cutoff = np.random.uniform(low=1, high=1.5)
        cutoff = np.random.uniform(low=low_cutoff-1, high=3)

        background = low_cutoff + np.mod(background, cutoff-1)

    has_noise = np.random.choice([True, False], p=(0.3, 0.7))
    if has_noise:
        var = np.random.uniform(low=0, high=.2)
    else:
        var = 0

    proposal_grid = np.maximum(1, background + var*np.random.randn(*eps_grid.shape))

    eps_grid[np.where(mask)] = proposal_grid[np.where(mask)]

def add_circle(eps_grid, center=None, radius=None, size_scale=1):

    if center==None:
        x = np.random.uniform(low=0, high=eps_grid.shape[1])
        y = np.random.uniform(low=0, high=eps_grid.shape[0])
        center=(x,y)
    if radius == None:
        radius = np.random.uniform(low=4, high=40*size_scale)
    
    xs, ys = np.meshgrid(np.arange(eps_grid.shape[1]), np.arange(eps_grid.shape[0]))
    mask = (xs-center[0])**2 + (ys-center[1])**2 < radius**2

    add_shape(eps_grid, mask)


def add_square(eps_grid, center=None, half_width=None, size_scale=1):
    if center==None:
        x = np.random.uniform(low=0, high=eps_grid.shape[1])
        y = np.random.uniform(low=0, high=eps_grid.shape[0])
        center=(x,y)
    if half_width == None:
        half_width = np.random.uniform(low=3, high=40*size_scale)

    xs, ys = np.meshgrid(np.arange(eps_grid.shape[1]), np.arange(eps_grid.shape[0]))
    mask = np.maximum(np.abs(xs-center[0]), np.abs(ys-center[1])) < half_width

    add_shape(eps_grid, mask)
    

def gen_eps_grid(shape=(64, 256), num_shapes=None, size_scale=None):

    eps_grid = np.ones(shape)

    if num_shapes == None:
        num_shapes = np.random.randint(low=2, high=70)

    if size_scale == None:
        size_scale = np.random.uniform(low=0.2, high=1.5)

    for _ in range(num_shapes):
        
        shape = np.random.choice(['circ', 'sq'])

        if shape == 'circ':
            add_circle(eps_grid, size_scale=size_scale)
        elif shape == 'sq':
            add_square(eps_grid, size_scale=size_scale)

    return eps_grid



if __name__ == '__main__':

    eps_grid = gen_eps_grid()
    #np.save("data/eps_grids/1", eps_grid)

    if True:
        import matplotlib.pyplot as plt
        
        plt.imshow(eps_grid)
        plt.show()

