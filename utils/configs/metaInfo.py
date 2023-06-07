from utils.pis_utils import load_labels
import cv2

conf_thres=0.25  # confidence threshold
iou_thres=0.45  # NMS IOU threshold
max_det=1000  # maximum detections per image
classes=None  # filter by class: --class 0, or --class 0 2 3
agnostic_nms=False  # class-agnostic NMS
augment=False  # augmented inference
visualize=False  # visualize features
hide_labels=False  # hide labels
hide_conf=False  # hide confidences
line_thickness = 2
age_conf = 0.98
gen_conf = 0.98
fas_conf = 0.98
dx = 35
dy = 35
recog_threshold = 4.0
key_conf_threshold = 0.25
key_input_size = [256, 192]

font = cv2.FONT_HERSHEY_SIMPLEX
names = load_labels()
sizeOfEmbedding = 5
info_org = cv2.imread("utils/srcImg/info.png")
driver_background = cv2.cvtColor(cv2.imread("utils/srcImg/driver_pose.png"), cv2.COLOR_RGB2BGR) 
passenger_background = cv2.cvtColor(cv2.imread("utils/srcImg/pass_pose.png"), cv2.COLOR_RGB2BGR) 
driver_land_background = cv2.cvtColor(cv2.imread("utils/srcImg/driver_land.png"), cv2.COLOR_RGB2BGR) 
passenger_land_background = cv2.cvtColor(cv2.imread("utils/srcImg/pass_land.png"), cv2.COLOR_RGB2BGR) 
sourceImg = "utils/srcImg/"
video_path = "TestVideos/demo1.avi"
dirPath = "utils/enrolledEmbeddings/"
detect_model_trt_path = "utils/weights/yolov5s6_trt_fp16.pth"
keypoint_model_trt_path = "utils/weights/fs_keypoint_trt_fp16.pth"
recog_model_trt_path = "utils/weights/recognition_trt_fp16.pth"
agegen_model_trt_path = "utils/weights/agegender_trt_fp16.pth"
spoof_model_trt_path = "utils/weights/spoofing_trt_fp16.pth"


