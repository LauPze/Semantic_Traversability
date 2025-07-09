import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image, PointField
from cv_bridge import CvBridge
import numpy as np
import sensor_msgs_py.point_cloud2 as pc2
import message_filters
import yaml


def quat_to_rot_matrix(q):
    x, y, z, w = q
    R = np.array([
        [1 - 2*(y**2 + z**2),     2*(x*y - z*w),         2*(x*z + y*w)],
        [2*(x*y + z*w),           1 - 2*(x**2 + z**2),   2*(y*z - x*w)],
        [2*(x*z - y*w),           2*(y*z + x*w),         1 - 2*(x**2 + y**2)]
    ])
    return R


def build_P2_from_camera_info(self, filepath):
    with open(filepath, 'r') as f:
        line = f.readline()
        fx, fy, cx, cy = map(float, line.strip().split())
    P2 = np.array([
        [fx, 0, cx, 0],
        [0, fy, cy, 0],
        [0,  0,  1, 0]
    ])
    return P2
    
def load_velo_to_cam_from_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        calib = yaml.safe_load(f)
    
    q_dict = calib['os1_cloud_node-pylon_camera_node']['q']
    t_dict = calib['os1_cloud_node-pylon_camera_node']['t']

    # Quaternion w,x,y,z --> x,y,z,w
    q = [q_dict['x'], q_dict['y'], q_dict['z'], q_dict['w']]
    t = np.array([t_dict['x'], t_dict['y'], t_dict['z']]).reshape(3,1)

    R = quat_to_rot_matrix(q)
    R = np.transpose(R)

    Tr = np.hstack((R, -R @ t))  # 3x4
    return Tr


class FusionNode(Node):
    def __init__(self):
        super().__init__('fusion_node')

        # Paramètres calibration RELLIS
        self.declare_parameter('calib_velo_to_cam_path', 'data/rellis/transforms.yaml')
        self.calib_velo_path = self.get_parameter('calib_velo_to_cam_path').get_parameter_value().string_value

        self.declare_parameter('calib_camera_info_path', 'data/rellis/camera_info.txt')

#        self.calib_transform_path = self.get_parameter('calib_transform_path').get_parameter_value().string_value
        self.calib_camera_info_path = self.get_parameter('calib_camera_info_path').get_parameter_value().string_value

        # Charger matrices calibration
        self.Tr_velo_to_cam = load_velo_to_cam_from_yaml(self.calib_velo_path) # 3x4
        self.P2 = self.build_P2_from_camera_info(self.calib_camera_info_path)
        

        self.bridge = CvBridge()

        # Subscribers synchronisés
        #pc_sub = message_filters.Subscriber(self, PointCloud2, '/cortex/ouster/points')
        pc_sub = message_filters.Subscriber(self, PointCloud2, '/os1_cloud_node/points')
        seg_sub = message_filters.Subscriber(self, Image, '/bisenetv2/segmentation_labels')

        ts = message_filters.ApproximateTimeSynchronizer([pc_sub, seg_sub], queue_size=50, slop=0.5)
        ts.registerCallback(self.callback)

        self.pub_fusion = self.create_publisher(PointCloud2, '/velodyne_points_semantic', 30)

        self.get_logger().info('FusionNode initialisé et synchronisation OK.')

    
    def load_transform_yaml(self, filepath, key):
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        q = data[key]['q']
        t = data[key]['t']

        # Quaternion to rotation matrix (x,y,z,w order)
        quat = [q['x'], q['y'], q['z'], q['w']]
        R_mat = quaternion_to_rotation_matrix(quat)

        t_vec = np.array([t['x'], t['y'], t['z']]).reshape(3, 1)

        Tr = np.hstack((R_mat, t_vec))
        return Tr

    def build_P2_from_camera_info(self, filepath):
        with open(filepath, 'r') as f:
            line = f.readline()
            fx, fy, cx, cy = map(float, line.strip().split())
        P2 = np.array([
            [fx, 0, cx, 0],
            [0, fy, cy, 0],
            [0, 0, 1, 0]
        ])
        return P2

    def voxel_grid_filter(self, points, voxel_size=0.1):
        discrete_coords = np.floor(points / voxel_size).astype(np.int32)
        voxel_keys = discrete_coords[:, 0] + discrete_coords[:, 1] * 10000 + discrete_coords[:, 2] * 100000000
        unique_keys, inverse_indices = np.unique(voxel_keys, return_inverse=True)
        sums = np.zeros((unique_keys.size, 3), dtype=np.float64)
        counts = np.zeros(unique_keys.size, dtype=np.int32)
        np.add.at(sums, inverse_indices, points)
        np.add.at(counts, inverse_indices, 1)
        filtered_points = sums / counts[:, None]
        return filtered_points.astype(np.float32)

    def callback(self, pc_msg, seg_label_msg):
        labels_img = self.bridge.imgmsg_to_cv2(seg_label_msg, desired_encoding='mono8')
        
        points = np.array([
            [pt[0], pt[1], pt[2]] for pt in pc2.read_points(pc_msg, field_names=("x", "y", "z"), skip_nans=True)
        ])
        
        
#        mask = (points[:, 0] > 0) & (points[:, 0] < 30)
#        points = points[mask]

#        if points.shape[0] > 0:
#            points = self.voxel_grid_filter(points, voxel_size=0.1)

        if points.shape[0] == 0:
            self.get_logger().warn("Point cloud vide.")
            return
        
        points = self.voxel_grid_filter(points, voxel_size=0.1)

        ones = np.ones((points.shape[0], 1), dtype=np.float32)
        points_hom = np.hstack([points, ones]).T  # 4xN

        pts_cam = self.Tr_velo_to_cam @ points_hom  # 3xN
        
        z_cam = pts_cam[2, :]

#        valid = pts_cam[2, :] #> 0
#        pts_cam = pts_cam[:, valid]
#        points = points[valid]

        ones_row = np.ones((1, pts_cam.shape[1]), dtype=np.float32)
        pts_2d_hom = self.P2 @ np.vstack((pts_cam, ones_row))  # 3xN

        

        u = (pts_2d_hom[0, :] / pts_2d_hom[2, :]).astype(int)
        v = (pts_2d_hom[1, :] / pts_2d_hom[2, :]).astype(int)
        
        img_h, img_w = labels_img.shape

        valid_proj = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)

        labels = np.zeros(points.shape[0], dtype=np.uint8)
        valid_label = valid_proj & (z_cam > 0)

        labels[valid_proj] = labels_img[v[valid_proj], u[valid_proj]]

        points = points[valid_proj]
        labels = labels[valid_proj]
        
        palette = np.array([
            [0, 0, 0], [108, 64, 20], [0, 102, 0], [0, 255, 0], [0, 153, 153],
            [0, 128, 255], [0, 0, 255], [255, 255, 0], [255, 0, 127], [64, 64, 64],
            [255, 0, 0], [102, 0, 0], [204, 153, 255], [102, 0, 204], [255, 153, 204],
            [0, 102, 102], [153, 204, 255], [102, 255, 255], [101, 101, 11], [114, 85, 47]
        ], dtype=np.uint8)

        colors = palette[labels]

        r = colors[:, 0].astype(np.uint32)
        g = colors[:, 1].astype(np.uint32)
        b = colors[:, 2].astype(np.uint32)
        rgb_uint32 = (r << 16) | (g << 8) | b
        rgb_float = rgb_uint32.view(np.float32)

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name='label', offset=16, datatype=PointField.UINT8, count=1)
        ]

        cloud_data_np = np.zeros(points.shape[0], dtype=[
            ('x', np.float32),
            ('y', np.float32),
            ('z', np.float32),
            ('rgb', np.float32),
            ('label', np.uint8)
        ])
        cloud_data_np['x'] = points[:, 0]
        cloud_data_np['y'] = points[:, 1]
        cloud_data_np['z'] = points[:, 2]
      
        
        cloud_data_np['rgb'] = rgb_float
        cloud_data_np['label'] = labels

        header = pc_msg.header
        header.frame_id = 'velodyne'

        pc_sem_msg = pc2.create_cloud(header, fields, cloud_data_np)

        self.pub_fusion.publish(pc_sem_msg)


def main(args=None):
    rclpy.init(args=args)
    node = FusionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

