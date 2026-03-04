#!/usr/bin/env python3
# SafeNode (ROS2) - Drop/Safe Safety Publisher
# - Webcam + YOLO bbox center y(t) 기반 낙하 감지(dy/vy/ay)
# - Stabilization: warm-up skip, consecutive confirm, loss≠drop, drop lock
# - Publishes {"hazard_type":"Safe"|"Drop"} to /robot/drop_detected
# - Hardware required: webcam + model weights (not included)
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String
import json
import numpy as np  
import time
import cv2
from ultralytics import YOLO
import torch




# ---------------- ROS2 통합 노드 ----------------
class SafeNode(Node):
    def __init__(self):
        super().__init__("rear_safe_vision_node")

        self.log = self.get_logger()
        self.log.set_level(rclpy.logging.LoggingSeverity.DEBUG)
        self.log.info("SafeNode initialized - LOG LEVEL = DEBUG")

        # 상태 변수
        self.object_current_coords = None

        # 속도 기반
        self.fall_count = 0
        self.prev_vy = 0
        self.last_distance = None
        self.last_time = None

        self.prev_down = False  
        self.prev_conf = None   

        self.drop_locked = False
        self.frame_count = 0
        self.WARMUP_FRAMES = 8  

        # ---------------- YOLO + Webcam ----------------
        # 카메라 설정
        self.cap = cv2.VideoCapture("/dev/video2")
        self.cap.set(cv2.CAP_PROP_FPS, 5)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        # YOLO 모델 로드
        self.log.info("YOLO loading start")
        self.yolo_model = YOLO('weights/best.pt')

        # GPU 있으면 GPU 사용
        if torch.cuda.is_available():
            self.device = 'cuda:0'
            self.yolo_model.to(self.device)
            self.log.info("YOLO device: CUDA")
        else:
            self.device = 'cpu'
            self.log.info("YOLO device: CPU")

        # 세그 화면 출력 여부 
        self.enable_seg_vis = True   
        self.yolo_imgsz = 270      

        self.log.info("YOLO loading done")
        self.log.info(f"YOLO model task: {self.yolo_model.task}")

        # 타이머 주기 0.1초 (10Hz)
        self.timer = self.create_timer(0.1, self.update_camera_object_coords)
        self.log.info("SafeNode timer active (10Hz).")

        # -------------------- 안전 관련 --------------------
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # 떨어짐 신호 발행
        self.drop_hazard_publisher = self.create_publisher(
            String, '/robot/drop_detected', qos_profile
        )

    # ---------------- 안전/실행 루프 ----------------
    # ----------------- 웹캠 + YOLO + 드랍 연계 -----------------
    def update_camera_object_coords(self):
        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.log.error("❌ Webcam frame NULL")
            return
        
        self.frame_count += 1

        # 처음 N프레임은 드랍 판단 안 함
        if self.frame_count <= self.WARMUP_FRAMES:
            self.log.debug(f"Warm-up frame {self.frame_count}/{self.WARMUP_FRAMES} (skip drop detection)")
            cv2.imshow("YOLO", frame)
            cv2.waitKey(1)
            return
    

        results = self.yolo_model(
            frame,
            imgsz=self.yolo_imgsz,
            device=self.device,
            verbose=False
        )[0]

        if self.enable_seg_vis:
            vis_frame = results.plot()
        else:
            vis_frame = frame.copy()

        # -----------------------
        #  1) 박스 없음 = 소실
        # -----------------------
        if results.boxes is None or len(results.boxes) == 0:

            if self.prev_down and (self.prev_conf is not None) and (self.prev_conf > 0.5):
                # 낙하 패턴 직후 소실 → 추적실패 가능성 높음
                self.log.warning("⚠️ Lost right after falling pattern (suspect drop), keep monitoring")
                self.fall_count = min(self.fall_count + 1, 999)
                self.publish_hazard("Safe")   # 확정은 아님
            else:
                # 그냥 소실 → Safe + 연속성 끊김
                self.log.warning("⚠️ Object lost (tracking loss) -> Safe")
                self.fall_count = 0
                self.publish_hazard("Safe")

            self.object_current_coords = None
            self.prev_down = False   # 소실 이후엔 패턴 끊긴 걸로 보는 게 일반적
            cv2.imshow("YOLO", vis_frame)
            cv2.waitKey(1)
            return


        # -----------------------
        #  2) 박스 있음
        # -----------------------
        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        clses = results.boxes.cls.cpu().numpy().astype(int)

        best = int(np.argmax(confs))
        x1, y1, x2, y2 = boxes[best]
        conf = confs[best]

        # 중심 좌표
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2

        curr_pos = [x_center, y_center, 0]

        # 드랍 로직 호출
        self.check_for_drop_hazard_vision(curr_pos, conf)
      
        cv2.imshow("YOLO", vis_frame)
        cv2.waitKey(1)


    # ----------------- 드랍 감지 로직 -----------------
    def check_for_drop_hazard_vision(self, curr_pos, curr_conf):
        now = time.time()

        if self.last_distance is None:
            self.last_distance = curr_pos
            self.last_time = now
            self.prev_vy = 0
            self.fall_count = 0
            self.prev_down = False
            self.prev_conf = curr_conf
            return

        dt = max(now - self.last_time, 1e-2) 
        dy = curr_pos[1] - self.last_distance[1]
        vy = dy / dt
        ay = (vy - self.prev_vy) / dt

        falling_pattern = (dy > 50 and vy > 300 and ay > 300)

        if falling_pattern:
            self.fall_count += 1
        else:
            self.fall_count = 0

        DROP_CONFIRM_FRAMES = 2
        if self.fall_count >= DROP_CONFIRM_FRAMES:
            self.publish_hazard("Drop")
        else:
            self.publish_hazard("Safe")

        # 다음 프레임을 위한 업데이트 (prev_down 활용의 핵심)
        self.prev_down = falling_pattern
        self.prev_conf = curr_conf
        self.last_distance = curr_pos
        self.last_time = now
        self.prev_vy = vy




    # ----------------- 토픽 발행 -----------------
    def publish_hazard(self, status: str):

        # Drop 발생 시 락
        if status == "Drop":
            self.drop_locked = True

        # 락이 걸리면 어떤 경우에도 Drop만 발행
        if self.drop_locked:
            status = "Drop"

        msg = String()
        msg.data = json.dumps({"hazard_type": status})
        self.drop_hazard_publisher.publish(msg)

        self.log.info(f"[PUBLISH] Hazard={status} (locked={self.drop_locked})")


# ----------------- main -----------------
def main(args=None):
    rclpy.init(args=args)

    # 3) SafeNode 생성
    safe_node = SafeNode()

    try:
        rclpy.spin(safe_node)
    except KeyboardInterrupt:
        pass
    finally:
        safe_node.cap.release()
        cv2.destroyAllWindows()
        safe_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
