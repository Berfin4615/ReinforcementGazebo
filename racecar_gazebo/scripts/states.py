 #!/usr/bin/env python

import rospy
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import LaserScan
from threading import Thread
import math
#Publisher class
class CommandPublisher(Thread):
    def __init__(self):
        super(CommandPublisher, self).__init__()
        self.pub = rospy.Publisher('/vesc/low_level/ackermann_cmd_mux/input/navigation', AckermannDriveStamped, queue_size=10)
        self.rate = rospy.Rate(1) #1 Hz

    def run(self):
            rospy.loginfo("Command Publisher Node Started")
            while not rospy.is_shutdown():
                msg = AckermannDriveStamped()
                msg.drive.speed = 0.5
                msg.drive.steering_angle = 0.4189
                msg.drive.steering_angle_velocity = 1
                msg.drive.acceleration = 0.0
                msg.drive.jerk = 0.0
                self.pub.publish(msg)
                rospy.loginfo("Published drive command: Speed=%.2f, Steering Angle=%.2f", msg.drive.speed, msg.drive.steering_angle)
                self.rate.sleep()

#Subscriber class
class LidarSubscriber(Thread):
    def __init__(self):
        super(LidarSubscriber, self).__init__()
        rospy.Subscriber('/scan', LaserScan, self.callback)
        rospy.loginfo("LIDAR Subscriber Node Initialized")
        self.pub = rospy.Publisher('/vesc/low_level/ackermann_cmd_mux/input/navigation', AckermannDriveStamped, queue_size=10)
        self.rate = rospy.Rate(1)  # 1 Hz

    def callback(self, data):
        rospy.loginfo("LIDAR Data Received")
        
        # Process Lidar data
        ranges = data.ranges
        angle_min = data.angle_min
        angle_increment = data.angle_increment
        
        #Define the range and angle to monitor
        range_threshold = 2.0  # meters
        angle_range = (math.pi / 4)  # 45 degrees
        dots = []
        # # Loop through the ranges and find obstacles within the angle range
        for i, r in enumerate(ranges):
            angle = angle_min + i * angle_increment
            if r < range_threshold and -angle_range/2 < angle < angle_range/2:
                # rospy.loginfo("Obstacle detected at angle %.2f with distance %.2f", angle, r)
                dots.append(r)
                rospy.loginfo(data.ranges)
                # Simple behavior: if obstacle is detected within angle range, turn away
                steering_angle = -0.5 if angle < 0 else 0.5
                break
        
        rospy.loginfo(dots)
        rospy.loginfo("\n-----------------------------------\n")

    def run(self):
        rospy.loginfo("Starting LIDAR Subscriber Spin")
        rospy.spin()

if __name__ == "__main__":
    try:
        rospy.init_node('combined_node', anonymous=True)

        command_publisher = CommandPublisher()
        lidar_subscriber = LidarSubscriber()
        command_publisher.start()
        lidar_subscriber.start()

        command_publisher.join()
        lidar_subscriber.join()

    except rospy.ROSInterruptException:
        rospy.logerr("ROS Interrupt Exception")


#Camera sınıfı:
#!/usr/bin/env python

# import rospy
# import cv2
# import cv_bridge
# import numpy as np
# from ackermann_msgs.msg import AckermannDriveStamped
# from sensor_msgs.msg import Image
# from threading import Thread
# # Subscriber class
# class CameraSubscriber(Thread):
#     def __init__(self):
#         super(CameraSubscriber, self).__init__()
#         self.bridge = cv_bridge.CvBridge()
#         self.image_sub = rospy.Subscriber('/camera/zed/rgb/image_rect_color', Image, self.callback)
#         self.cmd_vel_pub = rospy.Publisher('/vesc/low_level/ackermann_cmd_mux/input/navigation', AckermannDriveStamped, queue_size=10)
#         self.ack = AckermannDriveStamped()
#         rospy.loginfo("Camera Subscriber Node Initialized")

#     def callback(self, data):
#         try:
#             # Convert ROS Image message to OpenCV image
#             image = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
#             hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
#             # Define color range for masking
#             lower_yellow = np.array([0, 0, 100])
#             upper_yellow = np.array([180, 30, 255])
#             mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
            
#             # Process mask to find the center of the yellow region
#             h, w, d = image.shape
#             search_top = h // 2 + 50
#             search_bot = h - 20
#             mask[0:search_top, 0:w] = 0
#             mask[search_bot:h, 0:w] = 0
#             M = cv2.moments(mask)
            
#             if M['m00'] > 0:
#                 cx = int(M['m10'] / M['m00'])
#                 cy = int(M['m01'] / M['m00'])
#                 cv2.circle(mask, (cx, cy), 20, (0, 0, 255), -1)
                
#                 # Control based on the center of yellow region
#                 err = cx - w / 2
#                 self.ack.drive.speed = 0.5
#                 angle = -float(err) / 25
#                 max_angle = 0.6
#                 if angle > max_angle:
#                     angle = max_angle
#                 elif angle < -max_angle:
#                     angle = -max_angle
                
#                 self.ack.drive.steering_angle = angle
#                 self.cmd_vel_pub.publish(self.ack)
                
#                 rospy.loginfo("Center: (%d, %d), Error: %.2f", cx, cy, err)
#             else:
#                 rospy.loginfo("No yellow region detected")
        
#         except cv_bridge.CvBridgeError as e:
#             rospy.logerr("CvBridge Error: %s", str(e))

#     def run(self):
#         rospy.loginfo("Starting Camera Subscriber Spin")
#         rospy.spin()

# if __name__ == "__main__":
#     try:
#         rospy.init_node('combined_node', anonymous=True)

#         command_publisher = CommandPublisher()
#         camera_subscriber = CameraSubscriber()
#         command_publisher.start()
#         camera_subscriber.start()

#         command_publisher.join()
#         camera_subscriber.join()

#     except rospy.ROSInterruptException:
#         rospy.logerr("ROS Interrupt Exception")
