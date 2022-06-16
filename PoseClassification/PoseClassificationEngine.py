import numpy as np
import json
import torch
import sys
import yaml
from .ModelLoader import ModelLoader

class PoseClassificationEngine():
    """ Activity Recognition Engine"""

    def __init__(self, MODEL_PATH):

        # load model parameters

        self.interpreter = ModelLoader(MODEL_PATH)
        self.activity_list = self.interpreter.labels
        self.T = self.interpreter.T
        self.input_buffer = torch.zeros([1, self.T, 17, 3], dtype=torch.float64, requires_grad=False)
        self.current_recognition = [None, None]

    def load_yaml(self, PATH='./parameters.yaml'):
        """
        Reads yaml configuration script
        """
        stream = open(PATH, 'r')
        dictionary = yaml.safe_load(stream)

        return dictionary

    def _outputInterpretation(self, output):
        """ Gets the action with the most confidence """

        activity = self.activity_list[torch.argmax(output).item()]
        self.current_recognition = [activity, torch.max(output).item()]
        pass

    def generateJsonResponse(self, objs):
        activity_raw_output = objs[2]
        obj_list = [1]
        activity = objs[0][0]
        score = float(objs[0][1])
        y_coordinates = np.transpose(objs[1][0][0])[0]
        x_coordinates = np.transpose(objs[1][0][0])[1]
        keypoint_scores = np.transpose(objs[1][0][0])[2]

        obj_list.append({
            "activity": activity,
            "activity_score": score,
            "keypoint_scores": list(map(lambda score: float(score), keypoint_scores)),
            "x_coordinates": list(map(lambda x: float(x), x_coordinates)),
            "y_coordinates": list(map(lambda y: float(y), y_coordinates)),
            "activity_raw_output": list(map(lambda score: float(score), activity_raw_output)),
            "activity_list": self.activity_list
        })

        return json.dumps(obj_list)

    def _updateInputBuffer(self, input):
        '''Joins several time frames together'''
        self.input_buffer[:,:self.T-1,:,:] = self.input_buffer[:,1:,:,:].clone()
        self.input_buffer[:,self.T-1:,:,:] = input[:,0,:,:]


    def detect(self, input):
        ''' Performs Classification: this method is called from main.py to make the classification'''
        self._updateInputBuffer(torch.tensor(input, requires_grad=False)) # contains keypoints and score
        network_output = self.interpreter.detect(self.input_buffer)
        self._outputInterpretation(network_output)

        return self.current_recognition, network_output