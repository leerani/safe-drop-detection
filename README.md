# Safe Drop Detection (ROS2 Safety Node)

YOLO 탐지 결과의 bbox 중심 y(t) 변화를 이용해 낙하 패턴을 감지하고, SAFE/DROP 신호를 ROS2 토픽으로 발행하는 안전 보조 노드입니다.

## Demo
![demo](src/assets/safe_demo.gif)
> Note: 하드웨어(웹캠/모델 weights) 의존으로 전체 재현 실행은 장비가 필요하며, 본 레포는 로직/파이프라인 공유 목적입니다.


## What I did
- 낙하 감지 안정화 로직 설계 (Warm-up / 연속 검증 / Loss 분리 / Drop lock)
- ROS2 안전 신호 인터페이스 구현 및 메시지 발행


## Pipeline
Webcam → YOLO → bbox center y(t) → dy/vy/ay 낙하 패턴 → fall_count 연속 검증 → Drop lock → ROS2 Publish(SAFE/DROP)


## Key Logic
- Warm-up Skip: 초기 8프레임 Drop 판단 제외
- Consecutive Confirm: fall_count ≥ 2 연속일 때만 Drop 확정
- Loss ≠ Drop: bbox 소실은 Safe로 처리(직전 낙하 패턴이면 의심 상태 유지)
- Lock: Drop 확정 후 상태 흔들림 방지


## ROS2 I/O
- Pub: `/robot/drop_detected` (std_msgs/String)  
  payload: `{"hazard_type":"Safe" | "Drop"}`


### Output example
- Safe: {"hazard_type":"Safe"}
- Drop: {"hazard_type":"Drop"} (locked=true)


## Parameters (tunable)
- WARMUP_FRAMES = 8
- falling_pattern: dy>50, vy>300, ay>300
- DROP_CONFIRM_FRAMES = 2
