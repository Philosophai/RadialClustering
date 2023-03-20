import unittest
from RadialFoci import radial_map_data
import DataGenerator
import numpy as np
import pickle 

class TestRadialMapData(unittest.TestCase):
    def test_radial_map_data(self):
        # Generate test data
        dim = 3
        with open("testdata.obj", 'rb') as f:
            test_data = pickle.load(f)

        # Create a RadialFocusGroup object using the radial_map_data function
        radial_focus_group = radial_map_data(test_data)

        # Check if the correct number of radial_foci are generated
        self.assertEqual(len(radial_focus_group.foci), dim+1)
        self.assertGreater(len(radial_focus_group.foci), 0)
        # Check if the index_to_distance mapping is correct for each RadialFoci in RadialFocusGroup.foci
        for j, radial_point in enumerate(radial_focus_group.foci):
            radial_foci_distance = []

            for i, unmapped_point in enumerate(test_data):
                distance = np.linalg.norm(unmapped_point - radial_point.value)
                self.assertEqual( radial_point.distances[radial_point.global_index_to_distance_index[i]][1], distance)
                #print(f"""Radial foci {j} : point {i} : distances {radial_point.distances[radial_point.index_to_distance[i]][1]} : 
                #            {distance}""")
                
    def test_generate_radial_buffer_length(self):
        # Generate test data
        with open("testdata.obj", 'rb') as f:
            test_data = pickle.load(f)
        # Create a RadialFocusGroup object
        radial_focus_group = radial_map_data(test_data)

        # Test the generate_radial_buffer_length function
        sample_size = 50
        n_closest = 2

        radial_focus_group.generate_radial_buffer_length(test_data, sample_size, n_closest)
        radial_buffer_length_2 = radial_focus_group.radial_buffer
        radial_focus_group.generate_radial_buffer_length(test_data, sample_size, 10)
        radial_buffer_length_10 = radial_focus_group.radial_buffer
        print(f"Average Radial Buffer Length n = 2: {radial_buffer_length_2} : n = 10 {radial_buffer_length_10}")
        self.assertAlmostEqual(radial_buffer_length_2, 0.1757662387179271)
        self.assertAlmostEqual(radial_buffer_length_10, 0.40775918525943444)
        self.assertGreater(radial_buffer_length_10, radial_buffer_length_2)

if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

