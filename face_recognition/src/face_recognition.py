import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import supervision as sv
import os


# define class that defines the face recognition model
class FaceRecognitionModel:
    def __init__(self, model_size="n", verbose=False, device=0):
        self.model_size = model_size
        self.verbose = verbose
        self.device = device
        self.load_yolo()

    def recognize_faces(self, frame):
        """Recognize faces in a frame and return the result and the annotated image.
        Args:
            frame (np.ndarray): The input frame.
            verbose (bool): Whether to print the result.
            device (int): The device to use.
        """
        result = self.model(frame,verbose=self.verbose, device=self.device)[0]
        detected_faces = sv.Detections.from_ultralytics(result)
        annotated_image = self.box_annotator.annotate(scene=frame, detections=detected_faces) #label_annotator.annotate(scene=frame, detections=detected_faces,

        return result, annotated_image
    
    #import yolo face
    def load_yolo(self):
        # go to directory this file
        current_directory = os.path.dirname(os.path.abspath(__file__))
        print(current_directory)
        target_directory = os.path.join(current_directory, "../models")
        print(target_directory)
        os.chdir(target_directory)
        # load the yolo face model
        self.model = YOLO('https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8' + self.model_size + '-face.pt')
        self.box_annotator = sv.BoundingBoxAnnotator() 


def recognize_faces(camera):
    rospy.init_node('face_recogniton', anonymous=True)
    pub = rospy.Publisher('/annotated_faces', Image, queue_size=10)
    rate = rospy.Rate(15)  # 10 Hz

    while not rospy.is_shutdown():
        # Read a frame from the camera
        ret, frame = camera.read()

        # Inference
        results, annotated_image = face_model.recognize_faces(frame)

        # DEBUG ONLY
        cv2.imshow('frame', annotated_image)
        cv2.waitKey(1)

        # Publish the image message on the /face topic
        image_msg = bridge.cv2_to_imgmsg(annotated_image)
        pub.publish(image_msg)
        rate.sleep()

def main():
    rospy.init_node('face_recognition', anonymous=True)
    #rospy.Subscriber('/webcam', Image, image_callback)
    rospy.spin()

if __name__ == '__main__':
    bridge = CvBridge()
    cap = cv2.VideoCapture(0)


    face_model = FaceRecognitionModel(model_size="n", verbose=False)
    try:
        recognize_faces(cap)
    except rospy.ROSInterruptException:
        pass
