import numpy as np
import pickle
import numpy as np
import matplotlib.pyplot as plt

def generate_vectors(size, dim, pickled_name=None):
    array = np.random.rand(size, dim)
    
    if pickled_name is not None:
        with open(pickled_name, 'wb') as f:
            pickle.dump(array, f)
    
    return array

def load_pickle(pickled_name):
    with open(pickled_name, 'rb') as f:
        array = pickle.load(f)
    
    return array

def plot_data(data, focused_data=None, center = None, radial_cutoff = None):
    def plot_sphere(ax, center, radius, color, alpha=1.0, n_points=100):
        u = np.linspace(0, 2 * np.pi, n_points)
        v = np.linspace(0, np.pi, n_points)
        x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
        y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
        z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]

        ax.plot_surface(x, y, z, color=color, alpha=alpha)

    dim = data.shape[1]
    
    if dim == 2:
        plt.scatter(data[:, 0], data[:, 1], c='blue', label='Data')
        
        if focused_data is not None:
            plt.scatter(focused_data[:, 0], focused_data[:, 1], c='red', label='Focused Data')
        
        if center is not None:
            plt.scatter(center[0], center[1], c='black', label='Center')
            if radial_cutoff is not None:
                circle = plt.Circle(center, radial_cutoff, fill=False, edgecolor='black')
                plt.gca().add_patch(circle)
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.legend()
        plt.show()
    
    elif dim == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='blue', label='Data')
        
        if focused_data is not None:
            ax.scatter(focused_data[:, 0], focused_data[:, 1], focused_data[:, 2], c='red', label='Focused Data')
        if center is not None:
            ax.scatter(center[0], center[1],center[2], c='black', label='Center')
            if radial_cutoff is not None:
                plot_sphere(ax, center, radial_cutoff, color='red', alpha=0.2)
                
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        ax.legend()
        plt.show()
    
    else:
        raise ValueError("The dimensionality of the data should be 2 or 3 for plotting.")

if __name__ == "__main__":
    print("Generating Data")
    plot_data(generate_vectors(10, 3), generate_vectors(10, 3))