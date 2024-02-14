import rospy
from sensor_msgs.msg import Image
import cv2 
from cv_bridge import CvBridge

cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

def camera_publisher(camera):
    rospy.init_node('camera_publisher', anonymous=True)
    pub = rospy.Publisher('/webcam', Image, queue_size=10)
    rate = rospy.Rate(1)  # 10 Hz

    while not rospy.is_shutdown():
        # Read a frame from the camera
        ret, frame = camera.read()
        print(frame.shape)

        cv2.imshow('frame', frame)
        cv2.waitKey(1)

        # Convert the frame to a ROS Image message
        #image_msg = convert_frame_to_image_msg(frame)
        
        image_msg = bridge.cv2_to_imgmsg(frame)

        # Publish the image message on the /face topic
        pub.publish(image_msg)

        rate.sleep()

def initialize_camera():
    # TODO: Initialize the camera and return the camera object
    pass

def convert_frame_to_image_msg(frame):
    # TODO: Convert the frame to a ROS Image message and return it
    
    pass

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    bridge = CvBridge()
    
    try:
        camera_publisher(cap)
    except rospy.ROSInterruptException:
        pass

