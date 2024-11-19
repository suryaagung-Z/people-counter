import threading
import tkinter as tk
from tkinter import filedialog
import cv2
from ultralytics import YOLO
import time
import pygame 

# Memuat model YOLOv8
model = YOLO("Model/yolov8n.pt")

# Memuat nama-nama kelas COCO
with open("Model/COCO_labels.txt", "r") as f:
    class_names = f.read().split("\n")

# Inisialisasi variabel global
max_people = 5
count = 0
display_count = 0
last_update_time = time.time()

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

def display_video_feed(video, is_camera):
    global count
    blink_state = False  # Variabel untuk mengatur berkedipnya teks

    if not video.isOpened():
        print("Error membuka sumber video/kamera.")
        return

    # Resolusi frame output
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    out = cv2.VideoWriter(
        "resources/Camera_result.mp4" if is_camera else "resources/Video_result.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        20,
        (frame_width, frame_height),
    )

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
                class_id = box.cls
                label = class_names[int(class_id)]
                if label == "person":
                    local_count += 1
                    cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    y = y1 - 15 if y1 > 15 else y1 + 15
                    cv2.putText(
                        resized_frame,
                        f"{label}: {float(box.conf):.2f}",
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
        out.write(resized_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    video.release()
    out.release()
    cv2.destroyAllWindows()
    stop_sound()


def select_camera():
    global stop_flag
    stop_flag.clear()
    stop_button.pack_forget()
    video = cv2.VideoCapture(0)  # Buka kamera
    video_thread = threading.Thread(target=display_video_feed, args=(video, True))
    video_thread.start()
    stop_button.pack()

def select_video():
    global stop_flag
    stop_flag.clear()
    stop_button.pack_forget()
    file_path = filedialog.askopenfilename()  # Pilih file video
    video = cv2.VideoCapture(file_path)
    video_thread = threading.Thread(target=display_video_feed, args=(video, False))
    video_thread.start()
    stop_button.pack()

def stop_video():
    stop_flag.set()  # Set flag untuk menghentikan tampilan video

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
