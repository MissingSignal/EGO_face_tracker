import rospy
from geometry_msgs.msg import Quaternion
from std_msgs.msg import String
import cv2 
import json

def main_loop():
    while not rospy.is_shutdown():
        
        pub.publish(Quaternion)

        rate.sleep()

def compute_error(msg):
    """
    Compute the percentual error from the center of the image.
    Args:
        msg (bounding_box): The bounding box of the detected face.
    
    Returns:
        error (float): The percentual error.
    """
    # Compute the center of the image
    center = (672/2,376/2)

    # Compute the center of the bounding box on the image plane
    x = (msg[0] + msg[2]) / 2
    y = (msg[1] + msg[3]) / 2
    center_box = (x, y)

    print("FACE CENTER: ", center_box)
    print("IMAGE CENTER: ", center)

    # Compute the error in percentage
    error_x = (center_box[0] - center[0]) / center[0]
    error_y = (center_box[1] - center[1]) / center[1]

    return error_x, error_y

def compute_head_pose(error):
    """
    Compute the head pose.
    Args:
        error (float): The percentual error.
    
    Returns:
        head_pose (quaternione): The head pose.
    """
    # Compute the head pose
    head_pose = Quaternion(error, 0, 0, 0)

    return head_pose

def tracker_callback(msg):
    """
    Callback function that receives the bounding box of the detected face, then compute the percentual error from of the bounding box wrt the center of the image.
    The error is then used to compute the head pose, which is then published on the /head_pose topic.
    Args:
        msg (String): The bounding box of the detected face in json format.
    
    Returns:
        msg (quaternione): The head pose.
    """

    # parse message converting it to json
    msg = json.loads(msg.data)

    if not msg:
        return
    if len(msg) == 0:
        return

    # sort message by area knowing that box is expressed as [x1, y1, x2, y2]
    boxes = sorted(msg.items(), key=lambda item: (item[1][2]-item[1][0])*(item[1][3]-item[1][1]), reverse=True)

    #take the first box, we assume the bigger the closer to the camera
    target = boxes[0]
    print("TARGET BOX: ", target[1])

    # Compute the error, here we assume the center of the image is (640, 360)
    error_x, error_y = compute_error(target[1])

    print("error_x: ", int(error_x * 100), "error_y: ", int(error_y * 100) )

    # here we control the head pose by incrementing the pan angle and the tilt angle (yaw and pitch respectively)
    
    head_pose = Quaternion(error_x, error_y, 0, 0)

    # # Publish the head pose
    pub.publish(head_pose)


if __name__ == '__main__':
    rospy.init_node('face_tracker', anonymous=True)
    sub = rospy.Subscriber('/detections', String, tracker_callback)
    # define a subscriber for the head pose, we read the head pose from the /head_pose topic when needed
    sub_head_pose = rospy.Subscriber('/head_pose', Quaternion, head_pose_callback)
    pub = rospy.Publisher('/head_pose', Quaternion, queue_size=10)

    rate = rospy.Rate(15)  # 10 Hz

    try:
        main_loop()
    except rospy.ROSInterruptException:
        pass
