#import tflite_runtime.interpreter as tflite
import tensorflow as tf
import cv2
import math
import numpy as np
from .constants import KEYPOINT_EDGE_INDS_TO_COLOR
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

class PoseDetectionEngine():
    """Pose Detection Class has all the utilities needed to simply take an image as input and
    return interpreted output information or images

    Args:
        MODEL_PATH: model path to posedetection model
        """
    def __init__(self, MODEL_PATH):
        #self.interpreter = tflite.Interpreter(MODEL_PATH, experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
        #self.interpreter = tflite.Interpreter(MODEL_PATH)
        self.interpreter = tf.lite.Interpreter(MODEL_PATH)
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.interpreter.allocate_tensors()

    def _preprocessImage(self, image):
        """Returns Resized image according to model
        Args:
            image: input image
        Returns:
            * Resized Image
            """
        # Formats image into an format required by the model
        # Lightning
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = cv2.resize(image, (192, 192), interpolation=cv2.INTER_LINEAR)
        # image = tf.cast(image.reshape(1, 192, 192, 3), dtype=tf.float32)
        #Thunder
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
        image = tf.cast(image.reshape(1, 256, 256, 3), dtype=tf.float32)

        return image

    def _keypoints_and_edges_for_display(self,keypoints_with_scores, height, width, keypoint_threshold=0.11): # Only used to plot images here in python
        """Returns high confidence keypoints and edges for visualization.

        Args:
        keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
          the keypoint coordinates and scores returned from the MoveNet model.
        height: height of the image in pixels.
        width: width of the image in pixels.
        keypoint_threshold: minimum confidence score for a keypoint to be
          visualized.

        Returns:
        A (keypoints_xy, edges_xy, edge_colors) containing:

          * the coordinates of all keypoints of all detected entities;
          * the coordinates of all skeleton edges of all detected entities;
          * the colors in which the edges should be plotted.
        """
        keypoints_all = []
        keypoint_edges_all = []
        edge_colors = []
        num_instances, _, _, _ = keypoints_with_scores.shape
        for idx in range(num_instances):
            kpts_x = keypoints_with_scores[0, idx, :, 1]
            kpts_y = keypoints_with_scores[0, idx, :, 0]
            kpts_scores = keypoints_with_scores[0, idx, :, 2]
            kpts_absolute_xy = np.stack(
                [width * np.array(kpts_x), height * np.array(kpts_y)], axis=-1)
            kpts_above_thresh_absolute = kpts_absolute_xy[kpts_scores > keypoint_threshold, :]
            keypoints_all.append(kpts_above_thresh_absolute)

        for edge_pair, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
            if (kpts_scores[edge_pair[0]] > keypoint_threshold and
                    kpts_scores[edge_pair[1]] > keypoint_threshold):
                x_start = kpts_absolute_xy[edge_pair[0], 0]
                y_start = kpts_absolute_xy[edge_pair[0], 1]
                x_end = kpts_absolute_xy[edge_pair[1], 0]
                y_end = kpts_absolute_xy[edge_pair[1], 1]
                line_seg = np.array([[x_start, y_start], [x_end, y_end]])
                keypoint_edges_all.append(line_seg)
                edge_colors.append(color)
        if keypoints_all:
            keypoints_xy = np.concatenate(keypoints_all, axis=0)
        else:
            keypoints_xy = np.zeros((0, 17, 2))

        if keypoint_edges_all:
            edges_xy = np.stack(keypoint_edges_all, axis=0)
        else:
            edges_xy = np.zeros((0, 2, 2))
        return keypoints_xy, edges_xy, edge_colors

    def _output_interpretation(self, keypoints_with_scores, input_image): # Only used to plot images here in python
        """Draws the keypoint predictions on image.

        Args:
        image: A numpy array with shape [height, width, channel] representing the
        pixel values of the input image.
        keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
        the keypoint coordinates and scores returned from the MoveNet model.
        crop_region: A dictionary that defines the coordinates of the bounding box
        of the crop region in normalized coordinates (see the init_crop_region
        function below for more detail). If provided, this function will also
        draw the bounding box on the image.
        output_image_height: An integer indicating the height of the output image.
        Note that the image aspect ratio will be the same as the input image.

        Returns:
        A numpy array with shape [out_height, out_width, channel] representing the
        image overlaid with keypoint predictions.
    """
        height, width, channel = input_image.shape

        (keypoint_locs, keypoint_edges, edge_colors) = self._keypoints_and_edges_for_display(keypoints_with_scores, height,
                                                                                        width)

        # Draw pose
        for edge in keypoint_edges:
            cv2.line(input_image, pt1=(int(edge[0][0]), int(edge[0][1])), pt2=(int(edge[1][0]), int(edge[1][1])), thickness=2,
                     color=(254, 76, 120))

        ref1_radius = math.sqrt(math.pow(keypoint_locs[1][0], 2) + math.pow(keypoint_locs[2][0], 2))
        ref2_radius = math.sqrt(math.pow(keypoint_locs[2][0], 2) + math.pow(keypoint_locs[3][0], 2))
        r = (ref1_radius / ref2_radius) * 5

        for keypoint in keypoint_locs:
            cv2.circle(img=input_image, center=(int(keypoint[0]), int(keypoint[1])), radius=int(r), thickness=-1,
                       color=(255, 255, 0))

        return input_image, keypoint_locs, keypoint_edges, edge_colors


    def detect(self, input_image):
        """Runs model and returns interpreted network outputs
        Args:
            input_image: the input image
        Returns:
                output_image: the output image
                keypoints_locs: keypoint x y coordinates
                keypoint_edges: connection between keypoints "
                edge_colors: color for edges """

        mod_image = self._preprocessImage(input_image)
        mod_image = mod_image.astype(np.uint8)
        self.interpreter.set_tensor(self.input_details[0]['index'], mod_image.numpy())  # runs inference
        #start = time.time()
        self.interpreter.invoke()
        #print(time.time() - start)
        keypoints_with_scores = self.interpreter.get_tensor(self.output_details[0]['index'])

        # output_image, keypoint_locs, keypoint_edges, edge_colors = self._output_interpretation(keypoints_with_scores, input_image)
        #

        # x_keypoints, y_keypoints, scores = self._output_for_activity_recognition(keypoints_with_scores, input_image)

        return keypoints_with_scores

    def _output_for_activity_recognition(self, raw_output, input_image): #Not used for json response

        r_o = np.reshape(raw_output.T, (3,17),'F')
        width = input_image.shape[0]
        height = input_image.shape[1]

        y_keypoints_scaled = list(map(lambda y : y * width,r_o[:][0]))
        x_keypoints_scaled = list(map(lambda x : x * height,r_o[:][1]))
        scores = r_o[:][1]

        return x_keypoints_scaled, y_keypoints_scaled, scores
