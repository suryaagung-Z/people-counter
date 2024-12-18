import threading
import tkinter as tk
from tkinter import filedialog
import cv2
from ultralytics import YOLO
import time
import pygame 

# Memuat model YOLOv8
model = YOLO("Model/yolov8n.pt")

# Inisialisasi variabel global
max_people = 5
count = 0
display_count = 0
last_update_time = time.time()

# List untuk menyimpan semua confidence score
confidence_scores = []

# Fungsi untuk memperbarui tampilan jumlah orang setiap 5 detik
def update_count():
    global display_count, count, last_update_time
    current_time = time.time()
    if current_time - last_update_time >= 5:
        display_count = count
        last_update_time = current_time
    threading.Timer(5, update_count).start()

pygame.mixer.init()

def play_sound_continuous():
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.load("assets/beep-warning.mp3")
        pygame.mixer.music.play(-1)

def stop_sound():
    pygame.mixer.music.stop()

def display_video_feed(video):
    global count, confidence_scores
    blink_state = False

    if not video.isOpened():
        print("Error membuka sumber video/kamera.")
        return

    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    while True:
        if stop_flag.is_set():
            break

        success, frame = video.read()
        if not success:
            break

        (H, W) = frame.shape[:2]
        scale = min(screen_width / W, screen_height / H)
        new_size = (int(W * scale), int(H * scale))
        resized_frame = cv2.resize(frame, new_size)

        results = model(resized_frame)
        local_count = 0

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls)  # ID kelas
                if class_id == 0:  # 0 adalah ID untuk "person"
                    local_count += 1
                    confidence_scores.append(float(box.conf))  # Tambahkan confidence score ke list
                    cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    y = y1 - 15 if y1 > 15 else y1 + 15
                    cv2.putText(
                        resized_frame,
                        f"person: {float(box.conf):.2f}",
                        (x1, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

        count = local_count

        if display_count > max_people:
            text = f"{display_count} people\nMax: {max_people}"
            play_sound_continuous()
            text_color = (0, 0, 255) if blink_state else (255, 255, 255)
            blink_state = not blink_state
        else:
            text = f"{display_count} people\nMax: {max_people}"
            stop_sound()
            text_color = (255, 255, 255)

        font_scale = 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, 2)
        position = ((new_size[0] - text_width) // 2, (new_size[1] + text_height) // 2)
        for i, line in enumerate(text.split("\n")):
            y_offset = position[1] + i * (text_height + 10)
            cv2.putText(
                resized_frame,
                line,
                (position[0], y_offset),
                font,
                font_scale,
                text_color,
                2,
            )

        cv2.imshow("Output", resized_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    video.release()
    cv2.destroyAllWindows()
    stop_sound()

    calculate_average_confidence()

def calculate_average_confidence():
    if len(confidence_scores) > 0:
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        print(f"Rata-rata Confidence Score: {avg_confidence:.2f}")
    else:
        print("Tidak ada deteksi untuk menghitung rata-rata confidence score.")

def select_camera():
    global stop_flag
    stop_flag.clear()
    stop_button.pack_forget()
    video = cv2.VideoCapture(0)
    video_thread = threading.Thread(target=display_video_feed, args=(video,))
    video_thread.start()
    stop_button.pack()

def select_video():
    global stop_flag
    stop_flag.clear()
    stop_button.pack_forget()
    file_path = filedialog.askopenfilename()
    video = cv2.VideoCapture(file_path)
    video_thread = threading.Thread(target=display_video_feed, args=(video,))
    video_thread.start()
    stop_button.pack()

def stop_video():
    stop_flag.set()
    calculate_average_confidence()

# Inisialisasi pembaruan hitungan
update_count()

# Membuat jendela utama aplikasi
root = tk.Tk()
root.title("Deteksi dan Penghitungan Orang")
root.geometry("1000x600")

# Membuat label untuk menampilkan gambar atau video
image_label = tk.Label(root)
image_label.pack(fill="both", expand=True)

# Membuat frame untuk menampung tombol-tombol
buttons_frame = tk.Frame(root)
buttons_frame.pack(side="top", fill="y")

# Membuat tombol untuk memilih kamera
camera_button = tk.Button(buttons_frame, text="Pilih Kamera", command=select_camera)
camera_button.pack(side="left")

# Membuat tombol untuk memilih video
video_button = tk.Button(buttons_frame, text="Pilih Video", command=select_video)
video_button.pack(side="left")

# Membuat tombol untuk menghentikan video
stop_button = tk.Button(buttons_frame, text="Hentikan Video", command=stop_video)
stop_button.pack(side="left")

# Membuat flag untuk menghentikan thread video
stop_flag = threading.Event()
stop_button.pack_forget()

# Menjalankan loop utama aplikasi
root.mainloop()
