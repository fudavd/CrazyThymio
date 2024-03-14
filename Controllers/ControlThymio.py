
from typing import List

import numpy
import numpy as np
rng = numpy.random.default_rng()

class Controller(object):
    def __init__(self, n_states, n_actions, gcn_output_dim=0):
        self.n_input = n_states
        self.n_output = n_actions
        self.gcn_output_dim = gcn_output_dim
        self.controller_type = "default"
        self.umax_const = 0.1
        self.wmax = 1.5708 / 2.5

    @staticmethod
    def velocity_commands(state: np.ndarray) -> np.ndarray:
        return np.array([0])

    @staticmethod
    def geno2pheno(genotype: np.array):
        return

    @staticmethod
    def save_geno(path: str):
        return

    @staticmethod
    def load_geno(path: str):
        return

class NumpyNetwork:
    def __init__(self, n_input, n_hidden, n_output, reservoir=True):
        self.reservoir = reservoir
        self.n_con1 = n_input * n_hidden
        self.n_con2 = n_hidden * n_output
        self.lin1 = np.random.uniform(-1, 1, (n_hidden, n_input))
        if reservoir:
            self.lin2 = np.random.uniform(-1, 1, (n_hidden, n_input))
        self.output = np.random.uniform(-1, 1, (n_output, n_hidden))

    def set_weights(self, weights: np.array):
        """
        Set the weights of the Neural Network controller
        """
        if self.reservoir:
            assert len(weights) == self.n_con2, f"Got {len(weights)} but expected {self.n_con2}"
            weight_matrix = weights[-self.n_con2:].reshape(self.output.shape)
            self.output = weight_matrix
        else:
            assert len(
                weights) == self.n_con1 + self.n_con2, f"Got {len(weights)} but expected {self.n_con1 + self.n_con2}"
            weight_matrix1 = weights[:self.n_con1].reshape(self.lin1.shape)
            weight_matrix2 = weights[-self.n_con2:].reshape(self.output.shape)
            self.lin1 = weight_matrix1
            self.output = weight_matrix2

    def forward(self, state: numpy.array):
        # hid_l = np.maximum(np.dot(self.lin1, state)*0.01, np.dot(self.lin1, state))
        hid_l = np.log(1 + np.exp(np.dot(self.lin1, state)))
        if self.reservoir:
            hid_l = np.log(1 + np.exp(np.dot(self.lin2, state)))
        output_l = 1 / (1 + np.exp(-np.dot(self.output, hid_l)))
        output_l[1] = output_l[1] * 2 - 1
        return output_l


class NNController(Controller):
    def __init__(self, n_states, n_actions):
        super().__init__(n_states, n_actions)
        self.controller_type = "NN"
        self.model = NumpyNetwork(n_states, n_states, n_actions)

    def geno2pheno(self, genotype: np.array):
        self.model.set_weights(genotype)

    def map_state(self, min_from, max_from, min_to, max_to, state_portion):
        return min_to + np.multiply((max_to - min_to), np.divide((state_portion - min_from), (max_from - min_from)))

    def velocity_commands(self, state: np.ndarray) -> np.ndarray:
        """
        Given a state, give an appropriate action

        :param <np.array> state: A single observation of the current state, dimension is (state_dim)
        :return: <np.array> action: A vector of motor inputs
        """

        assert (len(state) == self.n_input), "State does not correspond with expected input size"
        state[:4] = self.map_state(0, 2, -1, 1, state[:4])
        state[4:8] = self.map_state(-np.pi, np.pi, -1, 1, state[4:8])  # Assumed distance sensing range is 2.0 meters. If not, check!
        state[-1] = self.map_state(0, 255.0, -1, 1, state[-1])  # Gradient value, [0, 255]

        action = self.model.forward(state)
        control_input = action * np.array([self.umax_const, self.wmax])
        return control_input

    def save_geno(self, path: str):
        if self.model.reservoir:
            np.save(path + "/reservoir", [self.model.lin1, self.model.lin2, self.model.output], allow_pickle=True)

    def load_geno(self, path: str):
        if self.model.reservoir:
            self.model.lin1, self.model.lin2, self.model.output = np.load(path + "/reservoir.npy", allow_pickle=True)




class adaptiveNNController(Controller):
    def __init__(self, n_states, n_actions):
        super().__init__(n_states, n_actions)
        self.controller_type = "aNN"
        self.rnn1 = NNController(n_states, n_actions)
        self.rnn1.controller_type = "rnn1"
        self.rnn2 = NNController(n_states, n_actions)
        self.rnn2.controller_type = "rnn2"
        self.probabilities = np.array([0., 0.25, 0.5, 0.75, 0.75])
        self.intensity_thr = np.array([229.14699, 178.0845, 127.02098, 75.957306, 0])
        self.current_controller = None
        self.refract_time = 10
        self.refract_n = 0

    def velocity_commands(self, state: np.ndarray) -> np.ndarray:
        """
        Given a state, give an appropriate action

        :param <np.array> state: A single observation of the current state, dimension is (state_dim)
        :return: <np.array> action: A vector of motor inputs
        """
        assert (len(state) == self.n_input), "State does not correspond with expected input size"
        local_intensity = state[-1]  # Gradient value, [0, 255]

        if (self.refract_n % self.refract_time) == 0:
            prob = self.probabilities[np.argmax(self.intensity_thr <= local_intensity)]
            if rng.random() < prob:
                self.current_controller = self.rnn1
            else:
                self.current_controller = self.rnn2
            self.refract_n = 0
        self.refract_n += 25
        control_input = self.current_controller.velocity_commands(state)
        return control_input

    def geno2pheno(self, genotype: List[np.array]):
        self.rnn1.geno2pheno(genotype[0])
        self.rnn2.geno2pheno(genotype[1])

    def load_geno(self, path: List[str]):
        self.rnn1.load_geno(path[0])
        self.rnn2.load_geno(path[1])
