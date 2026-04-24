"""
Wrapper around https://github.com/ertsiger/gym-subgoal-automata/tree/master
which provides environments such as coffee world and water world.

Sample usage:
# See the link above for other options
env = gym.make("gym_subgoal_automata:OfficeWorldDeliverCoffee-v0",
               params={"hide_state_variables": true, ...})
env = DanielGymAdapter(env)
"""
import abc
import math
from typing import List, Set

import numpy as np
      
class LabelExtractor(abc.ABC):
    @abc.abstractmethod
    def get_labels(self, observation, info: dict):
        raise NotImplementedError("get_labels")

    @abc.abstractmethod
    def get_labels_without_probability(self, observation, info: dict) -> Set[str]:
        raise NotImplementedError("get_labels_without_probability")
      
class NoisyLabelingFunctionComposer(LabelExtractor):
    def __init__(self, label_funs: List[LabelExtractor]):
        assert len(label_funs) > 0

        super().__init__()
        self.label_funs = label_funs

    def get_labels(self, observation, info):
        labels = {}
        for label_fun in self.label_funs:
            labels.update(label_fun.get_labels(observation, info))
        return labels

    def get_labels_without_probability(self, observation, info: dict) -> Set[str]:
        labels = []
        for label_fun in self.label_funs:
            labels.extend(label_fun.get_labels_without_probability(observation, info))
        return labels

class OfficeWorldAbstractLabelExtractor(LabelExtractor):
    def __init__(self, sensor_true_confidence: float,
                 sensor_false_confidence: float,
                 label: str,
                 seed: int = 0, value_true_prior=1 / (5 * 5)):
        super().__init__()
        self.sensor_true_confidence = sensor_true_confidence
        self.sensor_false_confidence = sensor_false_confidence
        
        self.label = label

        self.rng = np.random.default_rng(seed)
        self.num_steps = 0
        self.value_true_prior = value_true_prior

    """
    We use the following binary sensor model:
        - It takes in two parameters:
            - p(true_value_predicated | true_value) = sensor_true_confidence
                - can compute p(false_value_predicted | true_value) = 1 - sensor_true_confidence
            - p(false_value_predicated | false_value) = sensor_false_confidence
                - can compute p(true_value_predicted | false_value) = 1 - sensor_false_confidence
        - Assumes the prior probabilities as 0.5, i.e p(true_value) = 0.5 (=sensor_true_prior), p(false_value) = 0.5
        - We can see a sensor prediction, but need to compute our belief that the value is true.
            - This can be done with Bayes rule
            -   p(true_value | true_value_predicted) =
                   p(true_value_predicated | true_value) * p(true_value) / 
                   p(true_value_predicated | true_value) * p(true_value) 
                      + p(true_value_predicted | false_value) * p(false_value)
           - Opposite case p(true_value | false_value_predicted)  is similar
    """

    def get_label_confidence(self, label_true_pred: bool, value_true_prior: float = 0.5):
        value_false_prior = 1 - value_true_prior
        # case: p(true_value | true_value_predicted)
        if label_true_pred:
            p_true_and_true_pred = self.sensor_true_confidence * value_true_prior
            p_true_pred = (self.sensor_true_confidence * value_true_prior +
                           (1 - self.sensor_false_confidence) * value_false_prior)
            return p_true_and_true_pred / p_true_pred
        # case p(true_value | false_value_predicted)
        else:
            p_true_and_false_pred = (1 - self.sensor_true_confidence) * value_true_prior
            p_false_pred = ((1 - self.sensor_true_confidence) * value_true_prior
                            + self.sensor_false_confidence * value_false_prior)
            return p_true_and_false_pred / p_false_pred


    def get_labels(self, observation, info: dict):
        self.num_steps += 1
        # TODO: this may be slow; as we do it a number of times
        if self.get_label() in info.get("observations", set()):
            coffee_predicted = bool(self.rng.binomial(1, self.sensor_true_confidence))
        else:
            coffee_predicted = bool(1 - self.rng.binomial(1, self.sensor_false_confidence))
        labels = {self.get_label(): self.get_label_confidence(coffee_predicted, value_true_prior=self.value_true_prior)}
        return labels

    def get_labels_without_probability(self, observation, info: dict) -> Set[str]:
        if self.get_label() in info.get("observations", set()):
            coffee_predicted = bool(self.rng.binomial(1, self.sensor_true_confidence))
        else:
            coffee_predicted = bool(1 - self.rng.binomial(1, self.sensor_false_confidence))
        
        if not coffee_predicted:
            return []
        return [self.get_label()]

    def get_label(self):
        return self.label
