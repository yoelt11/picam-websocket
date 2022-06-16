import torch
import yaml
import sys
import importlib

class ModelLoader():

	def __init__(self, PATH):
		# ----- add the model path so that we can import the module
		sys.path.insert(0, PATH)
		# ------ load model parameters
		model_param = self.load_yaml(PATH) 
		parameters = model_param['MODEL_INIT']
		self.labels = model_param['MODEL_LABELS'] # labels
		self.T = model_param['MODEL_INIT']['T'] # Timeframes
		self.model_type = model_param['MODEL_TYPE']

		if model_param['MODEL_TYPE'] == "tf":
			import tflite_runtime.interpreter as tflite
			import tensorflow as tf
			self.interpreter = tf.lite.Interpreter(PATH + '/model.tflite',
											   experimental_op_resolver_type=tf.lite.experimental.OpResolverType.AUTO)
			self.input_details = self.interpreter.get_input_details()
			self.output_details = self.interpreter.get_output_details()
			self.interpreter.allocate_tensors()
		elif model_param['MODEL_TYPE'] == "torch":
			model = getattr(importlib.import_module('model'), 'Model')
			self.interpreter = model(**parameters)
			self.interpreter.load_weights(PATH)
		elif model_param['MODEL_TYPE'] == "jit-torch":
			self.interpreter = torch.jit.load(PATH+'/jit_model.pt')

	def load_yaml(self, PATH='./parameters.yaml'):
		"""
			Reads yaml configuration script
		"""
		stream = open(PATH + '/parameters.yaml', 'r')
		dictionary = yaml.safe_load(stream)

		return dictionary

	def trimFacialKp(self, input):
		'''Some models were trained without the face features resulting of an array of 37 instead of 51'''
		global_pose_score = torch.tensor([input[0, i, :, 2].mean() for i in range(input.shape[1])]).unsqueeze(0)

		T = input.shape[1]
		trimmed_input = torch.zeros([1, T, 37])
		trimmed_input[:, :, 1:37] = input[:, :, 5:, :].flatten(2)
		trimmed_input[:, :, 0] = global_pose_score

		return trimmed_input

	def detect(self, input):
		 if self.model_type =="tf":
			 if self.input_details[0]['shape'][2] == 37:
			 	input = self.trimFacialKp(input)
			 self.interpreter.set_tensor(self.input_details[0]['index'], input.to(torch.float32)) # .flatten(2) we must adapt tf net to the same input as torch
			 self.interpreter.invoke()
			 output = torch.tensor(self.interpreter.get_tensor(self.output_details[0]['index'])[0])
		 elif self.model_type  == "torch" or self.model_type == "jit-torch":
			 output = self.interpreter(input)
		 return output
