#!/usr/bin/env python

# import rospy
# from ackermann_msgs.msg import AckermannDriveStamped
# from sensor_msgs.msg import LaserScan
# from std_srvs.srv import Empty
# from threading import Thread
# import time

# class CommandPublisher(Thread):
#     def __init__(self):
#         super(CommandPublisher, self).__init__()
#         self.pub = rospy.Publisher('/vesc/low_level/ackermann_cmd_mux/input/navigation', AckermannDriveStamped, queue_size=10)
#         self.rate = rospy.Rate(10)  # 10 Hz
#         self.is_active = True

#     def run(self):
#         rospy.loginfo("Command Publisher Node Started")
#         while not rospy.is_shutdown():
#             if self.is_active:
#                 msg = AckermannDriveStamped()
#                 msg.drive.speed = 0.5
#                 msg.drive.steering_angle = 0.7  # Move straight
#                 msg.drive.steering_angle_velocity = 1
#                 msg.drive.acceleration = 0.0
#                 msg.drive.jerk = 0.0
#                 self.pub.publish(msg)
#                 #rospy.loginfo("Published drive command: Speed=%.2f, Steering Angle=%.2f", msg.drive.speed, msg.drive.steering_angle)
#             # self.rate.sleep()


# class LidarSubscriber(Thread):
#     def __init__(self):
#         super(LidarSubscriber, self).__init__()
#         rospy.Subscriber('/scan', LaserScan, self.callback)
#         rospy.loginfo("LIDAR Subscriber Node Initialized")
#         self.pub = rospy.Publisher('/vesc/low_level/ackermann_cmd_mux/input/navigation', AckermannDriveStamped, queue_size=10)
#         self.reset_srv = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)  # Service to reset simulation
#         self.rate = rospy.Rate(1)  # 1 Hz
#         self.command_publisher = None
#     def set_command_publisher(self, command_publisher):
#         self.command_publisher = command_publisher

#     def callback(self, data):
#         rospy.loginfo("LIDAR Data Received")
        

#         # Process Lidar data
#         ranges = data.ranges
#         rospy.loginfo(min(ranges))
#         range_threshold = 0.5  # meters, adjust this value as needed
#         rospy.loginfo(min(ranges))
#         # Check for obstacles directly in front of the racecar
#         if min(ranges) < range_threshold:
#             rospy.loginfo("Obstacle detected! Resetting simulation.")
#             self.reset_simulation()
#     def reset_simulation(self):
#         # try:
#         #     # self.command_publisher.stop()  # Temporarily stop publishing commands
#         #     self.reset_srv()
#         #     rospy.loginfo("Simulation reset successfully.")
#         #     # time.sleep(0.00001)  # Short delay to ensure the reset completes

#         #     # self.command_publisher.start_publishing()  # Resume command publishing
#         #     # rospy.loginfo("Command publisher resumed.")

#         # except rospy.ServiceException as e:
#         #     rospy.logerr("Service call failed: %s", e)
#         try:
#             self.reset_srv()
#             rospy.loginfo("Simulation reset successfully.")
#             # time.sleep(1)  # Short delay to ensure the reset completes
#             rospy.init_node('combined_node', anonymous=True)

#             command_publisher = CommandPublisher()
#             lidar_subscriber = LidarSubscriber()
#             lidar_subscriber.set_command_publisher(command_publisher)

#             command_publisher.start()
#             lidar_subscriber.start()

#             command_publisher.join()
#             lidar_subscriber.join()
#         except rospy.ServiceException as e:
#             rospy.logerr("Service call failed: %s", e)
    

    

#     # def run(self):
#     #     rospy.loginfo("Starting LIDAR Subscriber Spin")
#     #     rospy.sleep()
# if __name__ == "__main__":
#     try:
#         rospy.init_node('combined_node', anonymous=True)

#         command_publisher = CommandPublisher()
#         lidar_subscriber = LidarSubscriber()
#         lidar_subscriber.set_command_publisher(command_publisher)

#         command_publisher.start()
#         lidar_subscriber.start()

#         command_publisher.join()
#         lidar_subscriber.join()

#     except rospy.ROSInterruptException:
#         rospy.logerr("ROS Interrupt Exception")

        
# #!/usr/bin/env python

# import rospy
# from ackermann_msgs.msg import AckermannDriveStamped
# from sensor_msgs.msg import LaserScan
# from std_srvs.srv import Empty
# from threading import Thread
# import math
# import time

# class CommandPublisher(Thread):
#     def __init__(self):
#         super(CommandPublisher, self).__init__()
#         self.pub = rospy.Publisher('/vesc/low_level/ackermann_cmd_mux/input/navigation', AckermannDriveStamped, queue_size=10)
#         self.rate = rospy.Rate(10) # 1 Hz
#         self.reset_flag = False

#     def run(self):
#         rospy.loginfo("Command Publisher Node Started")
#         while not rospy.is_shutdown():
#             if not self.reset_flag:
#                 msg = AckermannDriveStamped()
#                 msg.drive.speed = 0.5
#                 msg.drive.steering_angle = 1.0  # Move straight
#                 msg.drive.steering_angle_velocity = 1
#                 msg.drive.acceleration = 0.0
#                 msg.drive.jerk = 0.0
#                 self.pub.publish(msg)
#                 rospy.loginfo("Published drive command: Speed=%.2f, Steering Angle=%.2f", msg.drive.speed, msg.drive.steering_angle)
#             self.rate.sleep()


# class LidarSubscriber(Thread):
#     def __init__(self):
#         super(LidarSubscriber, self).__init__()
#         rospy.Subscriber('/scan', LaserScan, self.callback)
#         rospy.loginfo("LIDAR Subscriber Node Initialized")
#         self.pub = rospy.Publisher('/vesc/low_level/ackermann_cmd_mux/input/navigation', AckermannDriveStamped, queue_size=10)
#         self.reset_srv = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)  # Service to reset simulation
#         self.rate = rospy.Rate(1)  # 1 Hz
#         self.command_publisher = None

#     def set_command_publisher(self, command_publisher):
#         self.command_publisher = command_publisher

#     def callback(self, data):
#         rospy.loginfo("LIDAR Data Received")

#         # Process Lidar data
#         ranges = data.ranges
#         range_threshold = 0.5  # meters, you may need to adjust this value

#         # Check for obstacles directly in front of the racecar
#         if min(ranges) < range_threshold:
#             rospy.loginfo("Obstacle detected! Resetting simulation.")
#             self.reset_simulation()

#     def reset_simulation(self):
#         try:
#             self.reset_srv()
#             rospy.loginfo("Simulation reset successfully.")
#             # time.sleep(1)  # Short delay to ensure the reset completes
#             rospy.init_node('combined_node', anonymous=True)

#             command_publisher = CommandPublisher()
#             lidar_subscriber = LidarSubscriber()
#             lidar_subscriber.set_command_publisher(command_publisher)

#             command_publisher.start()
#             lidar_subscriber.start()

#             command_publisher.join()
#             lidar_subscriber.join()
#         except rospy.ServiceException as e:
#             rospy.logerr("Service call failed: %s", e)

#     def run(self):
#         rospy.loginfo("Starting LIDAR Subscriber Spin")
#         rospy.spin()

# if __name__ == "__main__":
#     try:
#         rospy.init_node('combined_node', anonymous=True)

#         command_publisher = CommandPublisher()
#         lidar_subscriber = LidarSubscriber()
#         lidar_subscriber.set_command_publisher(command_publisher)

#         command_publisher.start()
#         lidar_subscriber.start()

#         command_publisher.join()
#         lidar_subscriber.join()

#     except rospy.ROSInterruptException:
#         rospy.logerr("ROS Interrupt Exception")
#!/usr/bin/env python

import rospy
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty
from threading import Thread
import time

class CommandPublisher(Thread):
    def __init__(self):
        super(CommandPublisher, self).__init__()
        self.pub = rospy.Publisher('/vesc/low_level/ackermann_cmd_mux/input/navigation', AckermannDriveStamped, queue_size=10)
        self.timer = rospy.Timer(rospy.Duration(0.1), self.timer_callback)  # 10 Hz
        self.is_active = True

    def timer_callback(self, event):
        if self.is_active:
            try:
                msg = AckermannDriveStamped()
                msg.drive.speed = 0.5
                msg.drive.steering_angle = 0.7  # Move straight
                msg.drive.steering_angle_velocity = 1
                msg.drive.acceleration = 0.0
                msg.drive.jerk = 0.0
                self.pub.publish(msg)
            except rospy.exceptions.ROSTimeMovedBackwardsException:
                rospy.logwarn("Caught ROS time backwards exception in CommandPublisher")
                rospy.sleep(1)  # Delay to allow time to stabilize

class LidarSubscriber(Thread):
    def __init__(self):
        super(LidarSubscriber, self).__init__()
        rospy.Subscriber('/racecar/scan', LaserScan, self.callback)
        rospy.loginfo("LIDAR Subscriber Node Initialized")
        self.reset_srv = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)  # Service to reset simulation
        self.reset_counter = 0

    def callback(self, data):
        rospy.loginfo("LIDAR Data Received")

        # Process Lidar data
        ranges = data.ranges
        rospy.loginfo(min(ranges))
        range_threshold = 0.5  # meters, adjust this value as needed

        # Check for obstacles directly in front of the racecar
        if min(ranges) < range_threshold:
            rospy.loginfo("Obstacle detected! Resetting simulation.")
            self.reset_simulation()

    def reset_simulation(self):
        if self.reset_counter < 5:  # Limit the number of resets to avoid excessive resets
            try:
                self.reset_srv()
                rospy.loginfo("Simulation reset successfully.")
                rospy.sleep(2)  # Delay to ensure the reset completes

                # Explicitly reset Gazebo time
                self.reset_ros_time()
                
                self.reset_counter += 1
            except rospy.ServiceException as e:
                rospy.logerr("Service call failed: %s", e)
        else:
            rospy.logwarn("Reset limit reached. No more resets.")

    def reset_ros_time(self):
        # Additional delay to handle time synchronization issues
        rospy.loginfo("Waiting for time synchronization after reset...")
        rospy.sleep(5)  # Adjust this value as needed to ensure time stabilization

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
