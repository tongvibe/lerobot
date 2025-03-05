import cv2

# 尝试不同的设备节点，直到找到正确的摄像头
for i in range(50):  # 通常摄像头设备节点从 0 到 9
    cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
    if cap.isOpened():
        print(f"Camera found at /dev/video{i}")
        break
    cap.release()
else:
    print("Camera not found.")
    exit()

# 设置摄像头参数（可选）
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 224)  # 设置宽度
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)  # 设置高度
cap.set(cv2.CAP_PROP_FPS, 30)  # 设置帧率

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    cv2.imshow('IMX258 Camera', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # 按 'q' 键退出
        break

cap.release()

cv2.destroyAllWindows()