import rospy
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import Pose
from std_msgs.msg import String
from tf.listener import TransformListener
import cv2 
import json
import tf
import math

 #define global variable for x_error and y_error
error_x = 0
error_y = 0

def main_loop():
    while not rospy.is_shutdown():

        # (1) get current pose 
        try:
            listener.waitForTransform('/torso', '/head_state', rospy.Time(0), rospy.Duration(4.0))
            (trans,rot) = listener.lookupTransform('/torso','/head_state', rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            print("[ERROR]: Can't get the current pose.")
            return
        
        # (2) compute target pose 
        print("CURRENT QUAT: ", rot)

        # convert rot quaternion to euler in degrees
        roll, pitch, yaw = tf.transformations.euler_from_quaternion(rot)
        #multiply by 180/pi to convert to degrees
        roll = roll * 180 / math.pi
        pitch = pitch * 180 / math.pi
        yaw = yaw * 180 / math.pi

        print("CURRENT EULER (DEG): ", roll, pitch, yaw)

        #define target pose: control the yaw and pitch
        Kp = 1
        target_yaw = yaw - (Kp * 0 )#error_x
        target_pitch = pitch + (Kp * 0) #error_y 
        target_roll = roll + (Kp * 0) # the roll is not actuated, then we force the error to be 0

        # DEFINE MAX AND MIN VALUES FOR THE HEAD POSE
        min_pitch =  -28 #sguardo alto
        max_pitch = 15 #sguardao basso

        if target_pitch > max_pitch:
            target_pitch = max_pitch
        elif target_pitch < min_pitch:
            target_pitch = min_pitch

        #start print in yellow
        print("\033[93m")
        print("INCREASING ROLL: ", target_roll-roll)
        print("INCREASING PITCH: ", target_pitch-pitch)
        print("INCREASING YAW: ", target_yaw-yaw)
        print("\033[0m")

        #print in red
        print("\033[91m")
        print("CURR ROLL: ", roll)
        print("CURR PITCH: ", pitch)
        print("CURR YAW: ", yaw)
        print("\033[0m")

        # finally convert the target head pose (degrees) to quaternion
        head_pose_quat = tf.transformations.quaternion_from_euler(target_roll*math.pi/180, target_pitch*math.pi/180, target_yaw*math.pi/180)
        print("TARGET QUAT: ", head_pose_quat)


        #as double check we convert the quaternion to euler again
        r,p,y = tf.transformations.euler_from_quaternion(head_pose_quat)
        print("TARGET EULER (DEG): ", r*180/math.pi, p*180/math.pi, y*180/math.pi)
            
        #head_pose = compute_head_pose(error_x)

        # (3) publish the target pose
        #define pose placeholder
        head_pose_target = Pose()
        # head_pose_target.orientation.x = rot[0]#head_pose_quat[0]
        # head_pose_target.orientation.y = rot[1]#head_pose_quat[1]
        # head_pose_target.orientation.z = rot[2]#head_pose_quat[2]
        # head_pose_target.orientation.w = rot[3]#head_pose_quat[3]
        head_pose_target.orientation.x = head_pose_quat[0]
        head_pose_target.orientation.y = head_pose_quat[1]
        head_pose_target.orientation.z = head_pose_quat[2]
        head_pose_target.orientation.w = head_pose_quat[3]
        head_pose_target.position.x = 0
        head_pose_target.position.y = 0
        head_pose_target.position.z = 0


        #= Pose(0, 0, 0, head_pose_quat[0], head_pose_quat[1], head_pose_quat[2], head_pose_quat[3])# Pose(trans, head_pose_quat)
        #Pose(0, 0, 0, head_pose_quat[0], head_pose_quat[1], head_pose_quat[2], head_pose_quat[3])

        #head_pose = Quaternion(error_x, error_y, 0, 0)
        ############################################

        # # Publish the head pose
        #pub.publish(head_pose_target)

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
    #print("TARGET BOX: ", target[1])

    # Compute the error, here we assume the center of the image is (640, 360)
    error_x, error_y = compute_error(target[1])

    #print("error_x: ", int(error_x * 100), "error_y: ", int(error_y * 100) )

    # if error is below a certain threshold we don't move the head
    if abs(error_x) < 0.1 and abs(error_y) < 0.1:
        return


if __name__ == '__main__':
    rospy.init_node('face_tracker', anonymous=True)

    sub = rospy.Subscriber('/detections', String, tracker_callback)
    listener = tf.TransformListener()
    #sub_head_pose = rospy.Subscriber('/head_pose', Quaternion, head_pose_callback)
    pub = rospy.Publisher('/robot_alterego3/head/head_pos', Pose, queue_size=10)
    rate = rospy.Rate(100)  # 10 Hz

    try:
        main_loop()
    except rospy.ROSInterruptException:
        pass
