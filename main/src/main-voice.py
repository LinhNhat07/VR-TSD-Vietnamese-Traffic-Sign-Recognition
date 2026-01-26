import cv2  # type: ignore
import numpy as np
from ultralytics import YOLO  # type: ignore
import time
import threading
from collections import deque
import logging
from gtts import gTTS  # type: ignore
import pygame  # type: ignore
import os
from datetime import datetime
from pathlib import Path

# ====================== LOGGING ======================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ====================== CONFIG ======================
CONFIG = {
    "model_path": "/Users/LinhNh/Main_Project/Traffic Sign/main/model/best.pt",
    "video_path": "/Users/LinhNh/Main_Project/Traffic Sign/data-test/test_voice.mp4",
    "output_dir": "/Users/LinhNh/Main_Project/Traffic Sign/main/results",
    "conf_threshold": 0.55,
    "iou_threshold": 0.45,
    "target_fps": 15,
    "history_length": 3,
    "cooldown_period": 5.0,
    "reset_cooldown": 10.0,
}

# ====================== CLASS NAMES ======================
CLASS_NAMES = [
    "102-cam-di-nguoc-chieu", "103a-cam-oto", "106-cam-oto-tai",
    "107-cam-oto-khach-va-oto-tai", "123a-cam-re-trai",
    "123b-cam-re-phai", "124a-cam-quay-dau-xe",
    "124c-cam-re-trai-va-quay-dau-xe", "125-cam-vuot",
    "127-toc-do-toi-da-20", "127-toc-do-toi-da-30",
    "127-toc-do-toi-da-40", "127-toc-do-toi-da-50",
    "127-toc-do-toi-da-60", "127-toc-do-toi-da-70",
    "127-toc-do-toi-da-80", "128-cam-su-dung-coi",
    "130-cam-dung-va-do-xe", "131a-cam-do-xe",
    "136-cam-di-thang", "137-cam-re-trai-va-re-phai",
    "201-cho-ngoat-nguy-hiem", "203-duong-bi-hep",
    "205-duong-giao-nhau", "207-giao-nhau-voi-duong-khong-uu-tien",
    "208-giao-nhau-voi-duong-uu-tien", "209-giao-nhau-co-tinh-hieu-den",
    "221-duong-go-ghe", "224-duong-nguoi-di-bo-cat-ngang",
    "225-chu-y-tre-em", "227-cong-truong", "245-di-cham",
    "301b-cac-xe-chi-duoc-re-phai", "301c-cac-xe-chi-duoc-re-phai",
    "302a-huong-phai-di-vong-chuong-ngai-vat",
    "303-noi-giao-nhau-chay-theo-vong-xuyen",
    "407a-duong-mot-chieu", "420-bat-dau-khu-dong-dan-cu",
    "421-ket-thuc-khu-dong-dan-cu", "442-cho"
]

# ====================== TEXT TO SPEECH LABEL ======================
SPEAK_LABELS = {
    # ===== BIỂN CẤM =====
    "102-cam-di-nguoc-chieu": "Cấm đi ngược chiều",
    "103a-cam-oto": "Cấm ô tô",
    "106-cam-oto-tai": "Cấm ô tô tải",
    "107-cam-oto-khach-va-oto-tai": "Cấm ô tô tải và xe khách",
    "123a-cam-re-trai": "Cấm rẽ trái",
    "123b-cam-re-phai": "Cấm rẽ phải",
    "124a-cam-quay-dau-xe": "Cấm quay đầu xe",
    "124c-cam-re-trai-va-quay-dau-xe": "Cấm rẽ trái và quay đầu xe",
    "125-cam-vuot": "Cấm vượt",
    "128-cam-su-dung-coi": "Cấm sử dụng còi",
    "130-cam-dung-va-do-xe": "Cấm dừng và đỗ xe",
    "131a-cam-do-xe": "Cấm đỗ xe",
    "136-cam-di-thang": "Cấm đi thẳng",
    "137-cam-re-trai-va-re-phai": "Cấm rẽ trái và rẽ phải",

    # ===== GIỚI HẠN TỐC ĐỘ =====
    "127-toc-do-toi-da-20": "Giới hạn tốc độ tối đa 20 km một giờ",
    "127-toc-do-toi-da-30": "Giới hạn tốc độ tối đa 30 km một giờ",
    "127-toc-do-toi-da-40": "Giới hạn tốc độ tối đa 40 km một giờ",
    "127-toc-do-toi-da-50": "Giới hạn tốc độ tối đa 50 km một giờ",
    "127-toc-do-toi-da-60": "Giới hạn tốc độ tối đa 60 km một giờ",
    "127-toc-do-toi-da-70": "Giới hạn tốc độ tối đa 70 km một giờ",
    "127-toc-do-toi-da-80": "Giới hạn tốc độ tối đa 80 km một giờ",

    # ===== BIỂN NGUY HIỂM / CẢNH BÁO =====
    "201-cho-ngoat-nguy-hiem": "Chú ý chỗ ngoặt nguy hiểm",
    "203-duong-bi-hep": "Chú ý đường bị hẹp",
    "205-duong-giao-nhau": "Chú ý đường giao nhau",
    "207-giao-nhau-voi-duong-khong-uu-tien": "Chú ý giao nhau với đường không ưu tiên",
    "208-giao-nhau-voi-duong-uu-tien": "Chú ý giao nhau với đường ưu tiên",
    "209-giao-nhau-co-tinh-hieu-den": "Chú ý giao nhau có tín hiệu đèn",
    "221-duong-go-ghe": "Chú ý đường gồ ghề",
    "224-duong-nguoi-di-bo-cat-ngang": "Chú ý người đi bộ",
    "225-chu-y-tre-em": "Chú ý trẻ em",
    "227-cong-truong": "Chú ý công trường",
    "245-di-cham": "Chú ý đi chậm",

    # ===== BIỂN HIỆU LỆNH =====
    "301b-cac-xe-chi-duoc-re-phai": "Các xe chỉ được rẽ phải",
    "301c-cac-xe-chi-duoc-re-phai": "Các xe chỉ được rẽ phải",
    "302a-huong-phai-di-vong-chuong-ngai-vat": "Hướng phải đi để tránh chướng ngại vật",
    "303-noi-giao-nhau-chay-theo-vong-xuyen": "Chú ý vòng xuyến",

    # ===== BIỂN CHỈ DẪN =====
    "407a-duong-mot-chieu": "Đường một chiều",
    "420-bat-dau-khu-dong-dan-cu": "Bắt đầu khu đông dân cư",
    "421-ket-thuc-khu-dong-dan-cu": "Kết thúc khu đông dân cư",
    "442-cho": "Chú ý khu vực chợ",
}


# ====================== AUTO COLOR ======================
def generate_class_colors(n):
    colors = {}
    for i in range(n):
        hue = int(180 * i / n)
        hsv = np.uint8([[[hue, 200, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        colors[i] = tuple(int(x) for x in bgr)
    return colors

CLASS_COLORS = generate_class_colors(len(CLASS_NAMES))

# ====================== TTS FUNCTIONS ======================
pygame.mixer.init()
announcement_queue = deque()
announcement_lock = threading.Lock()
last_announced = {}
sign_history = deque(maxlen=CONFIG["history_length"])

def play_tts(text):
    try:
        tts = gTTS(text=text, lang="vi")
        filename = f"tts_{time.time()}.mp3"
        tts.save(filename)

        with announcement_lock:
            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)

        os.remove(filename)
    except Exception as e:
        logger.error(f"TTS error: {e}")

def process_tts_queue():
    while announcement_queue:
        if not pygame.mixer.music.get_busy():
            sign = announcement_queue.popleft()
            text = SPEAK_LABELS.get(sign, sign.replace("-", " "))
            threading.Thread(target=play_tts, args=(text,), daemon=True).start()

# ====================== MAIN ======================
def main():
    model = YOLO(CONFIG["model_path"])
    cap = cv2.VideoCapture(CONFIG["video_path"])

    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    out_path = Path(CONFIG["output_dir"]) / f"output_{datetime.now():%Y%m%d_%H%M%S}.mp4"

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frame_time = 1.0 / CONFIG["target_fps"]

    while cap.isOpened():
        start = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=CONFIG["conf_threshold"], iou=CONFIG["iou_threshold"], verbose=False)

        current_signs = set()

        for box in results[0].boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            if conf < CONFIG["conf_threshold"]:
                continue

            sign_name = CLASS_NAMES[cls_id]
            current_signs.add(sign_name)

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = CLASS_COLORS[cls_id]

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{sign_name} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        sign_history.append(current_signs)

        if len(sign_history) == CONFIG["history_length"]:
            stable = set.intersection(*sign_history) if all(sign_history) else set()
            now = time.time()
            for sign in stable:
                if now - last_announced.get(sign, 0) > CONFIG["cooldown_period"]:
                    announcement_queue.append(sign)
                    last_announced[sign] = now
            if stable:
                threading.Thread(target=process_tts_queue, daemon=True).start()

        out.write(frame)
        cv2.imshow("Traffic Sign Detection + Voice", frame)

        elapsed = time.time() - start
        time.sleep(max(0, frame_time - elapsed))

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()
    logger.info("DONE")

if __name__ == "__main__":
    main()
