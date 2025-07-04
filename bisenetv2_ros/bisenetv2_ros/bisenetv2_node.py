import os
import rclpy
from rclpy.node import Node
import torch
from ament_index_python.packages import get_package_share_directory
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import torchvision.transforms as T

class BiSeNetV2Node(Node):
    def __init__(self):
        super().__init__('bisenetv2_node')
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Charger chemin modèle depuis share
        model_dir = get_package_share_directory('bisenetv2_ros')
        model_path = os.path.join(model_dir, 'models', 'bisenetv2_scripted.pt')
        
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.to(self.device)
        self.model.eval()

        self.get_logger().info(f'Modèle TorchScript chargé depuis : {model_path}')

        self.bridge = CvBridge()
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 512)),  # Adapter au modèle
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        self.palette = np.array([
            [0, 0, 0], [108, 64, 20], [0, 102, 0], [0, 255, 0], [0, 153, 153],
            [0, 128, 255], [0, 0, 255], [255, 255, 0], [255, 0, 127], [64, 64, 64],
            [255, 0, 0], [102, 0, 0], [204, 153, 255], [102, 0, 204], [255, 153, 204],
            [0, 102, 102], [153, 204, 255], [102, 255, 255], [101, 101, 11], [114, 85, 47]
        ], dtype=np.uint8)

        self.subscription = self.create_subscription(
            Image,
            '/occam/raw_image0',
            self.image_callback,
            10
        )
        self.publisher_labels = self.create_publisher(Image, '/bisenetv2/segmentation_labels', 10)
        self.publisher_rgb = self.create_publisher(Image, '/bisenetv2/segmentation_rgb', 10)

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            img_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            original_size = (cv_image.shape[1], cv_image.shape[0])
            
            input_tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.model(input_tensor)
                if isinstance(output, tuple):
                    output = output[0]
            pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
            
            pred_resized = cv2.resize(pred, original_size, interpolation=cv2.INTER_NEAREST)

            labels_msg = self.bridge.cv2_to_imgmsg(pred_resized.astype(np.uint8), encoding='mono8')
            labels_msg.header = msg.header
            self.publisher_labels.publish(labels_msg)

            seg_img = self.palette[pred_resized]
            rgb_msg = self.bridge.cv2_to_imgmsg(seg_img, encoding='rgb8')
            rgb_msg.header = msg.header
            self.publisher_rgb.publish(rgb_msg)

        except Exception as e:
            self.get_logger().error(f'Erreur dans image_callback: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = BiSeNetV2Node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

