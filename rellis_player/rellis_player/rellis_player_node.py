import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image
from std_msgs.msg import Header
import numpy as np
import os
import cv2
from cv_bridge import CvBridge
import sensor_msgs_py.point_cloud2 as pc2


class RellisPlayerNode(Node):
    def __init__(self):
        super().__init__('rellis_player')

        # Ne plus déclarer ni récupérer des paramètres (ou supprimer ces lignes si tu veux)
        # self.declare_parameter('rellis_velodyne_path', '')
        # self.declare_parameter('rellis_image_path', '')

        # Mettre directement les chemins en dur ici :
        self.velodyne_path = '/home/Laura/data/rellis/vel_cloud_node'
        self.image_path = '/home/Laura/data/rellis/image_raw'

        self.velo_files = sorted([f for f in os.listdir(self.velodyne_path) if f.endswith('.bin')])
        self.image_files = sorted([f for f in os.listdir(self.image_path) if f.endswith('.png') or f.endswith('.jpg')])
        if len(self.velo_files) == 0 or len(self.image_files) == 0:
            self.get_logger().error('Aucun fichier RELLIS trouvé dans les dossiers spécifiés.')
            raise RuntimeError('RELLIS data missing')

        self.pub_pc = self.create_publisher(PointCloud2, '/vel_cloud_node', 10)
        self.pub_img = self.create_publisher(Image, 'camera/image_raw', 10)

        self.bridge = CvBridge()

        self.index = 0
        self.timer = self.create_timer(0.1, self.timer_callback)  # 10 Hz
#        self.timer = self.create_timer(0.05, self.timer_callback)  # 20 Hz

        self.get_logger().info(f'RellisPlayer lancé: {len(self.velo_files)} scans LiDAR, {len(self.image_files)} images')


    def timer_callback(self):
        max_len = min(len(self.velo_files), len(self.image_files))
        if self.index >= max_len:
            self.index = 0
            
#        if self.index >= len(self.velo_files):
#            self.index = 0

        # Lecture LiDAR
        velo_file = os.path.join(self.velodyne_path, self.velo_files[self.index])
        points = self.read_velodyne_bin(velo_file)
        
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'velodyne'  # ou 'velodyne_points' 

        pc_msg = pc2.create_cloud_xyz32(header, points)
        self.pub_pc.publish(pc_msg)


        # Lecture image
        image_file = os.path.join(self.image_path, self.image_files[self.index])
        cv_img = cv2.imread(image_file, cv2.IMREAD_COLOR)
        if cv_img is None:
            self.get_logger().warn(f"Impossible de lire l'image {image_file}")
        else:
            img_msg = self.bridge.cv2_to_imgmsg(cv_img, encoding='bgr8')
            img_msg.header.stamp = header.stamp
            self.pub_img.publish(img_msg)

        self.index += 1

    def read_velodyne_bin(self, filepath):
        scan = np.fromfile(filepath, dtype=np.float32).reshape(-1, 4)
        points = scan[:, :3]  # x,y,z sans intensity
        return points


def main(args=None):
    print("RELLIS Player Node started")
    rclpy.init(args=args)
    node = RellisPlayerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

