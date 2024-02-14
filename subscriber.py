import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import sys
from ultralytics import YOLO
import supervision as sv


# define class that defines the face recognition model
class FaceRecognitionModel:
    def __init__(self,model_size="n"):
        self.model_size = model_size
        self.load_yolo_face()
        # initialize the face recognition model
        pass
    def load_model(self):
        # load the model
        pass
    
    #import yolo face
    def load_yolo_face(self):
        self.model = YOLO('https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8' + size + '-face.pt')
        self.box_annotator = sv.BoundingBoxAnnotator()


def image_callback(msg):
    #convert image to opencv format
    print("Received an image!")
    
    #cv_image = imgmsg_to_cv2(msg)
    cv_image = bridge.imgmsg_to_cv2(msg)
    cv2.imshow('frame', cv_image)
    #save the image
    cv2.imwrite('image_copy.png', cv_image)
    
    cv2.waitKey(1)
    # Process the image here
    # You can access the image data using msg.data
    

def main():
    rospy.init_node('image_reader')
    rospy.Subscriber('/webcam', Image, image_callback)
    rospy.spin()

if __name__ == '__main__':
    bridge = CvBridge()
    face_model = FaceRecognitionModel()
    main()
