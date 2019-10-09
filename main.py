# Importing all necessary libraries 
import cv2 
# Importing numpy library
import numpy as np
import os, sys
import options

# Script for extracting and saving frames to the disk, used to prepare Dataset
from util import preprocessing
from util import objectDetection

import tensorflow as tf
from tqdm import tqdm

def train():
    print("Function train")

def test(input_video_mode=None):
    print("Function test Video Mode:", input_video_mode)
    # We use Opencv to extract frames from both local video file or Web Cam
    if input_video_mode == "cam":
        cam = cv2.VideoCapture(0) # Use the deafault video cam, if you have multiple cams change the value! 
    else:
        try:
            cam = cv2.VideoCapture(args.input_dir) 
        except Exception as e:
            print("Unable to find the video file --> ", e)
    
    # Continue to call the Inference function
    # Import modules to visualize lable_map
    from object_detection.utils import label_map_util
    from object_detection.utils import visualization_utils as vis_util
    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_FROZEN_GRAPH = '/home/akhil/MyProjects/tensorflow-models-git-repo/downloaded_models/ObjectDetection_COCO/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb'

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = os.path.join('/home/akhil/MyProjects/tensorflow-models-git-repo/models/research/object_detection/data/', 'mscoco_label_map.pbtxt')


    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')


    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

    with detection_graph.as_default():
        with tf.Session() as sess:
        # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
                
            while True:
                ret, image_np = cam.read()
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                # Actual detection.
                output_dict = objectDetection.run_inference_for_single_image(image_np_expanded, tensor_dict, sess)
                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                            image_np,
                            output_dict['detection_boxes'],
                            output_dict['detection_classes'],
                            output_dict['detection_scores'],
                            category_index,
                            instance_masks=output_dict.get('detection_masks'),
                                            use_normalized_coordinates=True,
                                            line_thickness=8
                            )

                cv2.imshow('object detection', cv2.resize(image_np, (800,600)))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cam.release()
                    cv2.destroyAllWindows()
                    break

if __name__ == "__main__":
    args = options.parseArguments()
    
    if args.mode == "prepare_train_data":
        preprocessing.extractVideoFrames(filename=args.input_dir, output_folder=args.output_dir)
    elif args.mode == "train":
        print("Start Training....")
        train()
    elif args.mode == "test":
        test(args.video_mode)
    else:
        raise NameError