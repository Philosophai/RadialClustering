import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import pickle
import pandas as pd
import DataGenerator as dg

class RadialFocusGroup:
    def __init__(self):
        self.foci = []

    def distance_ordered_foci(self, index : int):
        """
        Order the foci by closeness for filtering
        """
        distances = []
        for foci_index, foci in enumerate(self.foci):
            distances.append((foci_index, foci.index_distance(index)))

        return sorted(distances, key = lambda x:x[1])

    def process_ring_filter(self, index : int, test_data):
        '''
        Given a global index and the data:
            Build a radial group for that index
        '''
        def pass_through_ring_filter(distance_dataframes):
            distance_len = len(distance_dataframes)
            merged_df = pd.concat(distance_dataframes, keys=range(len(distance_dataframes)), names=['source']).reset_index()
            common_indices = merged_df.groupby('index').filter(lambda x: len(x) == distance_len)
            filtered_points = common_indices.groupby('index')['value'].apply(list).reset_index().values.tolist()

            return filtered_points

        ring_list = []
        for x in radial_focus.distance_ordered_foci(index):
            result = radial_focus.foci[x[0]].find_radial_group(index, radial_focus.radial_buffer)
            ring_list.append(pd.DataFrame(result, columns=['index', 'value']))
            graph_list = [] ; index_list = []
            for r_index in result:
                graph_list.append(test_data[r_index[0]])
                index_list.append(r_index[0])
            focus = []
            for r_index in range(len(test_data)):
                if(r_index not in index_list):
                    focus.append(test_data[r_index])

        filtered_points = pass_through_ring_filter(ring_list)
        radial_cluster = []
        for index in filtered_points:
            radial_cluster.append((index[0], test_data[index[0]]))
        return radial_cluster

    def radial_cluster_distance_search(self, center_index, center, cluster_points, radial_cutoff):
        # Convert cluster_points to a dictionary
        cluster_points_dict = {index: point for index, point in cluster_points}
        print("Cluster Points Dictionary:", cluster_points_dict)
        
        if(len(cluster_points) == 0):
            return [], []

        # Step 1: Calculate distances between the center and cluster_points
        cluster_radi = {}
        for index, point in cluster_points:
            distance = np.linalg.norm(center - point)
            cluster_radi[index] = [distance, []]
        print("Cluster Radii and empty lists:", cluster_radi)
        
        # Step 2: Find the index of the minimum distance
        
        min_index, min_distance = min(
                    ((x, np.linalg.norm(cluster_radi[x] - cluster_radi[center_index])) for x in indices),
                    key=lambda x: x[1]
                )
        print("Index of minimum distance to center:", min_index)
        
        # Step 3: For each point in cluster_points, find the points within the distance threshold
        for index1, point1 in cluster_points:
            for index2, point2 in cluster_points:
                if index1 == index2:
                    continue
                distance = np.linalg.norm(point1 - point2)
                if distance < radial_cutoff - cluster_radi[index1][0]:
                    cluster_radi[index1][1].append(index2)
        print("Cluster Radii with updated lists of nearby points:", cluster_radi)
                    
        # Step 4: For each point with non-empty list, find the closest point
        indices_mapped = [center_index]
        minimum_in_cluster = [(center_index, min_index, min_distance)]
        for index, (radi, indices) in cluster_radi.items():
            if indices:
                indices_mapped.append(index)
                closest_point_index, min_distance = min(
                    ((x, np.linalg.norm(cluster_points_dict[x] - cluster_points_dict[index])) for x in indices),
                    key=lambda x: x[1]
                )
                minimum_in_cluster.append((index, closest_point_index, min_distance ))
        print("Minimum distance pairs within each cluster:", minimum_in_cluster)

        return minimum_in_cluster, indices_mapped

    def generate_radial_buffer_length(self,
                                      data : np.ndarray,
                                      sample_size : int, 
                                      n_closest : int):
        """
        generates a useful radial distance that will create groups of rough size n_closest.
        """
        # Randomly sample sample_size points from the data
        sampled_indices = np.random.choice(data.shape[0], size=sample_size, replace=False)
        sampled_points = data[sampled_indices]

        max_distances = []

        # Compute the distance between each sampled point and every other point in data
        for sampled_point in sampled_points:
            distances = np.linalg.norm(data - sampled_point, axis=1)

            # Find the n_closest distances
            closest_distances = np.partition(distances, n_closest)[:n_closest]

            # Find the maximum distance among the n_closest distances
            max_distance = np.max(closest_distances)
            max_distances.append(max_distance)

        # Return the average of the maximum distances
        self.radial_buffer =  np.mean(max_distances)
    
    def animate_radial_buffer_length(self, data, sample_size, n_closest, framerate=1):
        

        def plot_sphere(ax, center, radius, color, alpha=1.0, n_points=100):
            u = np.linspace(0, 2 * np.pi, n_points)
            v = np.linspace(0, np.pi, n_points)
            x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
            y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
            z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]

            ax.plot_surface(x, y, z, color=color, alpha=alpha)


        if data.shape[1] not in (2, 3):
            raise ValueError("The dimensionality of the data must be 2 or 3 for animation.")

        def update(frame):
            ax.clear()
            if frame < sample_size:
                # Plot only the current sampled_point
                ax.scatter(*sampled_points[frame], c='black', marker='o')
            elif frame < 2 * sample_size:
                i = frame - sample_size
                sampled_point = sampled_points[i]
                closest_points = n_closest_points[i]
                furthest_point = closest_points[-1]

                # Plot sampled_point and n_closest_points with the furthest_point in red
                ax.scatter(*closest_points.T, c='blue', marker='o')
                ax.scatter(*furthest_point, c='red', marker='o')
                ax.scatter(*sampled_point, c='black', marker='o')
            else:
                i = frame - 2 * sample_size
                sampled_point = sampled_points[i]
                furthest_point = n_closest_points[i][-1]

                # Plot sampled_point and the furthest_point in n_closest
                ax.scatter(*furthest_point, c='red', marker='o')
                ax.scatter(*sampled_point, c='black', marker='o')

                # Draw a circle (2D) or translucent sphere (3D) with radius equal to the radial_buffer_length
                if data.shape[1] == 2:
                    circle = Circle(sampled_point, radial_buffer_length, fill=False, edgecolor='black')
                    ax.add_patch(circle)
                else:
                    plot_sphere(ax, sampled_point, radial_buffer_length, color='red', alpha=0.2)


            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            if data.shape[1] == 3:
                ax.set_zlim(0, 1)

        sampled_indices = np.random.choice(data.shape[0], size=sample_size, replace=False)
        sampled_points = data[sampled_indices]

        n_closest_points = []
        max_vals = []
        for sampled_point in sampled_points:
            distances = np.linalg.norm(data - sampled_point, axis=1)
            closest_indices = np.argpartition(distances, n_closest)[:n_closest]
            closest_points = data[closest_indices]
            n_closest_points.append(closest_points)
            
            max_vals.append(np.linalg.norm(sampled_point - data[closest_indices[-1]]))
        radial_buffer_length = np.mean(np.array(max_vals))

        fig = plt.figure()
        if data.shape[1] == 2:
            ax = fig.add_subplot(111)
        else:
            ax = fig.add_subplot(111, projection='3d')

        ani = FuncAnimation(fig, update, frames=3 * sample_size, interval=1000 / framerate, blit=False)
        plt.show()

class RadialFoci:
    def __init__(self ,
                vector : np.ndarray):
        """
        creates the radial distances used for the clustering 
        """
        self.distances = []
        self.global_index_to_distance_index = {}
        self.value= vector

    def index_distance(self, index : int):
        return self.distances[self.global_index_to_distance_index[index]][1]
    
    def find_radial_group(self,
                          index : int,
                          radial_cutoff : float,
                          expansion_start : int = 2,):
        """
        Finds the group of indices in 4*log N time
        """
        def binary_barrier_search(boundary_condition,
                                  floor=None,
                                  ceiling=None):
            if not self.distances:
                return floor if floor is not None else ceiling

            low, high = 0, len(self.distances) - 1

            if self.distances[low][1] > boundary_condition:
                print("returning floor")
                return floor

            if self.distances[high][1] <= boundary_condition:
                print("returning ceil")
                return ceiling

            while low <= high:
                mid = (low + high) // 2

                if self.distances[mid][1] <= boundary_condition and self.distances[mid + 1][1] > boundary_condition:
                    print("MID:", mid, self.distances[mid][1],self.distances[mid + 1][1] )
                    return mid
                elif self.distances[mid][1] <= boundary_condition:
                    low = mid + 1
                else:
                    high = mid - 1
            print("RETURNING NONE")
            return None
        
        origin_value = self.index_distance(index)
        index = self.global_index_to_distance_index[index]
        expansion_value = expansion_start
        # first find the upward / downward limits
        upwards_floor_limit = index
        upward_ceil_limit = index + expansion_value
        while(upward_ceil_limit < self.distances.__len__() and index + expansion_value < self.distances.__len__() and self.distances[index + expansion_value][1] - origin_value < origin_value + radial_cutoff):
            expansion_value *= 2
            upward_ceil_limit = expansion_value
        if(upward_ceil_limit > self.distances.__len__()): upward_ceil_limit = self.distances.__len__()

        downward_ceil_limit = index
        downward_floor_limit = index - expansion_value
        while(downward_floor_limit > 0 and index - expansion_value > 0 and origin_value - self.distances[index - expansion_value][1] > origin_value - radial_cutoff):
            expansion_value *= 2
            downward_floor_limit = expansion_value
        if(downward_floor_limit < 0): downward_floor_limit = 0
        #print("upwards ",upwards_floor_limit, upward_ceil_limit)
        #print("downwards", downward_floor_limit, downward_ceil_limit)
        result =    self.distances[binary_barrier_search(origin_value - radial_cutoff, downward_floor_limit, downward_ceil_limit):upwards_floor_limit] + \
                    self.distances[downward_ceil_limit + 1: binary_barrier_search(origin_value + radial_cutoff, upwards_floor_limit, upward_ceil_limit)]
        return  result
    
def radial_map_data(unmapped_data : np.ndarray):
    """
    Maps out data to the radial dimensionality.
    """
    dim = unmapped_data.shape[1]
    radial_focus_group = RadialFocusGroup()
    # Create the radial_foci dataset
    radial_foci = np.eye(dim)
    radial_foci = np.vstack((radial_foci, np.zeros(dim)))  # Add the origin point

    # Compute the distance between each point in unmapped_data and radial_foci
    
    for j, radial_point in enumerate(radial_foci):
        radial_foci_distance = []

        for i, unmapped_point in enumerate(unmapped_data):
            radial_foci_distance.append((i, np.linalg.norm(unmapped_point - radial_point)))
            print("radial_focus_distnace", radial_foci_distance[i])
        radial_foci_distance = sorted(radial_foci_distance, key = lambda x: x[1])
        new_radial_foci = RadialFoci(radial_point)
        new_radial_foci.distances = radial_foci_distance
        new_radial_foci.global_index_to_distance_index = { entry[0] : index for index, entry in enumerate(new_radial_foci.distances)}
        radial_focus_group.foci.append(new_radial_foci)
    

    return radial_focus_group

### VISUAL TESTING ###

def test_animate_radial_buffer_length():
        with open("testdata.obj", 'rb') as f:
            test_data = pickle.load(f)
            # Create a RadialFocusGroup object
        radial_focus_group = radial_map_data(test_data)
        radial_focus_group.animate_radial_buffer_length( data = test_data, sample_size=10, n_closest=5, framerate=1)

def test_ring_filter():
    with open("testdata.obj", 'rb') as f:
        test_data = pickle.load(f)
    radial_focus = radial_map_data(test_data)
    radial_focus.generate_radial_buffer_length(test_data, 10, 5)
    filtered_points = radial_focus.process_ring_filter(30, test_data)

    indice_list = []
    index_list = []
    for indec in filtered_points:
        print(indec[0])
        index_list.append(indec[0])
        indice_list.append(test_data[indec[0]])
    excluded = []
    for n in range(len(test_data)):
        if(n not in index_list):
            excluded.append(test_data[n])
    dg.plot_data(np.array(indice_list), np.array(excluded), test_data[30], radial_focus.radial_buffer)

def test_radial_cluster_distance_search():
    with open("testdata.obj", 'rb') as f:
        test_data = pickle.load(f)
    radial_focus = radial_map_data(test_data)
    radial_focus.generate_radial_buffer_length(test_data, 10, 3)
    filtered_points = radial_focus.process_ring_filter(30, test_data)
    radial_focus.radial_cluster_distance_search(30, test_data[30],filtered_points, radial_focus.radial_buffer )

if __name__ == "__main__":
    #test_animate_radial_buffer_length()
    with open("testdata.obj", 'rb') as f:
        test_data = pickle.load(f)
    radial_focus = radial_map_data(test_data)
    radial_focus.generate_radial_buffer_length(test_data, 10, 5)
    filtered_points = radial_focus.process_ring_filter(30, test_data)
    radial_focus.radial_cluster_distance_search(30, test_data[30],filtered_points, radial_focus.radial_buffer )

    #radial_focus.animate_radial_buffer_length(test_data, 10, 5, 2)
    #test_ring_filter()
    pass