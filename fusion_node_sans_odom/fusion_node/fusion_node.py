import rclpy #client ROS2 Python
from rclpy.node import Node #classe de base pour créer un noeud ROS2
from sensor_msgs.msg import PointCloud2, Image, PointField #type de messages ROS
from cv_bridge import CvBridge #conversion ROS Images / OpenCV
import numpy as np
import sensor_msgs_py.point_cloud2 as pc2 #manipulation PointCloud2
import message_filters #synchronisation de plusieurs topic ROS2


## Création d'une classe héritant de Node
class FusionNode(Node):
    def __init__(self):
    
        ## Appel du constructeur parent
        super().__init__('fusion_node')

        ## Paramètres chemins calibration à modifier au besoin
        self.declare_parameter('calib_velo_to_cam_path', 'data/kitti/2011_09_26/calib_velo_to_cam.txt')
        self.declare_parameter('calib_cam_intr_path', 'data/kitti/2011_09_26/calib_cam_to_cam.txt')

        self.calib_velo_path = self.get_parameter('calib_velo_to_cam_path').get_parameter_value().string_value
        self.calib_cam_path = self.get_parameter('calib_cam_intr_path').get_parameter_value().string_value

        ## Charger matrices calibration 
        ##### Attention à la "key" en fonction du fichier de calibration
        self.Tr_velo_to_cam = self.load_velo_to_cam_from_RT(self.calib_velo_path)  # 3x4 transformation rigid body du LiDAR vers la caméra 
        self.P2 = self.load_matrix(self.calib_cam_path, key='P_rect_02', shape=(3, 4))  # 3x4 Matrice de projection caméra
        
        ## Création d’un objet CvBridge pour convertir les images ROS en format OpenCV et inversement
        self.bridge = CvBridge()

        ## Subscribers synchronisés
        pc_sub = message_filters.Subscriber(self, PointCloud2, '/velodyne_points')
        seg_sub = message_filters.Subscriber(self, Image, '/bisenetv2/segmentation_labels')  # Image mono8 labels
        ## Synchronisation approximative des deux abonnements 
        ts = message_filters.ApproximateTimeSynchronizer([pc_sub, seg_sub], queue_size=10, slop=0.5)
        ts.registerCallback(self.callback)

        ## Publisher nuage fusionné
        self.pub_fusion = self.create_publisher(PointCloud2, '/velodyne_points_semantic', 10)
        
        ## Log pour dire que le noeud est prêt 
        self.get_logger().info('FusionNode initialisé et synchronisation OK.')
        
##### Chargement des matrices de calibration
    
    ## Calibration du LiDAR vers caméra, cette matrice transforme les points 3D LiDAR en coordonnées caméra
    def load_velo_to_cam_from_RT(self, filepath):
        R = None
        T = None
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith('R:'):
                    values = [float(x) for x in line.strip().split()[1:]]
                    R = np.array(values).reshape(3, 3)
                elif line.startswith('T:'):
                    values = [float(x) for x in line.strip().split()[1:]]
                    T = np.array(values).reshape(3, 1)
        if R is None or T is None:
            raise RuntimeError("R ou T non trouvés dans le fichier " + filepath)
        Tr = np.hstack((R, T))
        self.get_logger().info(f"Tr_velo_to_cam chargée:\n{Tr}")
        return Tr

    ## Fonction générique pour charger une matrice --> cherche la ligne commençant par key et convertit la suite en matrice
    def load_matrix(self, filepath, key, shape):
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith(key + ':'):
                    values = [float(v) for v in line.strip().split()[1:]]
                    matrix = np.array(values).reshape(shape)
                    self.get_logger().info(f'{key} chargée:\n{matrix}')
                    return matrix
        raise RuntimeError(f"{key} non trouvé dans {filepath}")


    ## Callback synchronisé LiDAR + image segmentation
    def callback(self, pc_msg, seg_label_msg):
        # Convertir image mono8 labels
        labels_img = self.bridge.imgmsg_to_cv2(seg_label_msg, desired_encoding='mono8')

        # Extraire points 3D du PointCloud2
        points = np.array([
            [pt[0], pt[1], pt[2]] for pt in pc2.read_points(pc_msg, field_names=("x", "y", "z"), skip_nans=True)
        ])
        
        # Protection si nuage vide
        if points.shape[0] == 0:
            self.get_logger().warn("Point cloud vide.")
            return

        # Homogénéiser les points (x,y,z,1)
        points_hom = np.hstack((points, np.ones((points.shape[0], 1))))

        # Transformer points LiDAR vers le repère camera via la matrice Tr_velo_to_cam
        pts_cam = self.Tr_velo_to_cam @ points_hom.T  # 3xN

        # Garder points devant la caméra
        valid = pts_cam[2, :] > 0
        pts_cam = pts_cam[:, valid]
        points = points[valid]
        points_hom = points_hom[valid]

        # Projection des points 3D sur l'image 2D et conversion en coordonnées pixels entières (u,v)
        pts_2d_hom = self.P2 @ np.vstack((pts_cam, np.ones((1, pts_cam.shape[1]))))  # 3xN
        u = (pts_2d_hom[0, :] / pts_2d_hom[2, :]).astype(int)
        v = (pts_2d_hom[1, :] / pts_2d_hom[2, :]).astype(int)
        
        # Filtrage des points projetés qui sont dans les limites de l'image
        img_h, img_w = labels_img.shape
        valid_proj = (u >= 0) & (u < img_w) & (v >= 0) & (v < img_h)

        # Initialiser labels à 0 (void) partout
        labels = np.zeros(points.shape[0], dtype=np.uint8)
        labels[valid_proj] = labels_img[v[valid_proj], u[valid_proj]]

        # Palette RUGD 
        palette = np.array([
            [0, 0, 0],           # 0 void
            [108, 64, 20],       # 1 dirt
            [255, 229, 204],     # 2 sand
            [0, 102, 0],         # 3 grass
            [0, 255, 0],         # 4 tree
            [0, 153, 153],       # 5 pole
            [0, 128, 255],       # 6 water
            [0, 0, 255],         # 7 sky
            [255, 255, 0],       # 8 vehicle
            [255, 0, 127],       # 9 container/generic-object
            [64, 64, 64],        # 10 asphalt
            [255, 128, 0],       # 11 gravel
            [255, 0, 0],         # 12 building
            [153, 76, 0],        # 13 mulch
            [102, 102, 0],       # 14 rock-bed
            [102, 0, 0],         # 15 log
            [0, 255, 128],       # 16 bicycle
            [204, 153, 255],     # 17 person
            [102, 0, 204],       # 18 fence
            [255, 153, 204],     # 19 bush
            [0, 102, 102],       # 20 sign
            [153, 204, 255],     # 21 rock
            [102, 255, 255],     # 22 bridge
            [101, 101, 11],      # 23 concrete
            [114, 85, 47],       # 24 picnic-table
        ], dtype=np.uint8)

        colors = palette[labels]

        # Conversion couleur RGB en float32 compatible PointCloud2
        r = colors[:, 0].astype(np.uint32)
        g = colors[:, 1].astype(np.uint32)
        b = colors[:, 2].astype(np.uint32)
        rgb_uint32 = (r << 16) | (g << 8) | b
        rgb_float = rgb_uint32.view(np.float32)

        # Définition des champs du PointCloud 3 float position 1 flot couleur
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        # Préparer les tuples (x,y,z,rgb)
        cloud_data = [
            (points[i, 0], points[i, 1], points[i, 2], rgb_float[i])
            for i in range(points.shape[0])
        ]

        # Construire et publier le PointCloud2 fusionné
        header = pc_msg.header
        header.frame_id = 'velodyne'
        pc_sem_msg = pc2.create_cloud(header, fields, cloud_data)
        self.pub_fusion.publish(pc_sem_msg)
        
        # Log d'info avec le nombre de points publiés (sert de vérification, à supprimer quand temps réel)
        self.get_logger().info(f"Nuage fusionné publié avec {points.shape[0]} points.")

## Initialisation ROS2, création du noeud, boucle d'attente des messages, destruction propre du noeud
def main(args=None):
    rclpy.init(args=args)
    node = FusionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

