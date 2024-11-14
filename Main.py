import threading
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO

# Memuat model YOLOv8
model = YOLO("Model/yolov8n.pt")

# memuat nama-nama kelas COCO
with open("Model/COCO_labels.txt", "r") as f:
    class_names = f.read().split("\n")

def select_camera():
    global video, camera_thread
    # Mereset flag berhenti
    stop_flag.clear()
    # Menyembunyikan tombol stop
    stop_button.pack_forget()
    video = cv2.VideoCapture(0)
    # Memulai thread terpisah untuk menampilkan frame video dari kamera
    camera_thread = threading.Thread(target=display_camera, args=(video,))
    camera_thread.start()

    # Menampilkan tombol stop
    stop_button.pack()

def display_camera(video):
    if not video.isOpened():
        print("Error membuka kamera")

    # mendapatkan lebar dan tinggi frame video untuk penyimpanan yang benar
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    # membuat objek `VideoWriter()`
    out = cv2.VideoWriter(
        "resources/Camera_result.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        20,
        (frame_width, frame_height),
    )

    # Mendapatkan ukuran layar laptop
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # Membaca dan memproses setiap frame
    while True:
        # Memeriksa apakah flag berhenti diatur
        if stop_flag.is_set():
            break
        # Membaca frame
        success, frame = video.read()

        # Memeriksa apakah frame berhasil dibaca
        if not success:
            break

        # Mendapatkan dimensi frame
        (H, W) = frame.shape[:2]

        # Resize frame sesuai ukuran layar laptop
        scale = min(screen_width / W, screen_height / H)
        new_size = (int(W * scale), int(H * scale))
        resized_frame = cv2.resize(frame, new_size)

        # Deteksi objek dengan YOLOv8
        results = model(resized_frame)

        # Menginisialisasi penghitung orang
        count = 0

        # Menggambar kotak pembatas dan label pada frame
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = box.cls
                label = class_names[int(class_id)]
                confidence = box.conf

                # Menambahkan penghitung orang
                if label == "person":
                    count += 1
                    # Menggambar kotak di sekitar orang
                    label_text = f"{label}: {float(confidence):.2f}"
                    cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    y = y1 - 15 if y1 > 15 else y1 + 15
                    cv2.putText(
                        resized_frame,
                        label_text,
                        (x1, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

        # Menghitung posisi teks tengah
        text = f"{count} orang"
        font_scale = 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, 2)
        position = ((new_size[0] - text_width) // 2, (new_size[1] + text_height) // 2)

        # Menambahkan teks jumlah orang di tengah frame
        cv2.putText(
            resized_frame,
            text,
            position,
            font,
            font_scale,
            (255, 255, 255),
            2,
        )

        # Menampilkan frame
        cv2.imshow("Output", resized_frame)
        out.write(resized_frame)

        # Menunggu penekanan tombol
        key = cv2.waitKey(1) & 0xFF

        # Jika tombol 'q' ditekan, hentikan loop
        if key == ord("q"):
            break

    # Melepaskan sumber daya
    cv2.destroyAllWindows()
    video.release()
    out.release()

def select_video():
    global video, video_thread
    # Mereset flag berhenti
    stop_flag.clear()
    # Membuka dialog file untuk memilih video
    file_path = filedialog.askopenfilename()
    # Memuat video yang dipilih
    video = cv2.VideoCapture(file_path)
    # Memulai thread terpisah untuk menampilkan frame video
    video_thread = threading.Thread(target=display_video, args=(video,))
    video_thread.start()

    # Menampilkan tombol stop
    stop_button.pack()

def display_video(video):
    if not video.isOpened():
        print("Error membuka file video")

    # mendapatkan lebar dan tinggi frame video untuk penyimpanan yang benar
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    # membuat objek `VideoWriter()`
    out = cv2.VideoWriter(
        "resources/video_result.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        20,
        (frame_width, frame_height),
    )

    # Mendapatkan ukuran layar laptop
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # Membaca dan memproses setiap frame
    while True:
        # Memeriksa apakah flag berhenti diatur
        if stop_flag.is_set():
            break
        # Membaca frame
        success, frame = video.read()

        # Memeriksa apakah frame berhasil dibaca
        if not success:
            break

        # Mendapatkan dimensi frame
        (H, W) = frame.shape[:2]

        # Resize frame sesuai ukuran layar laptop
        scale = min(screen_width / W, screen_height / H)
        new_size = (int(W * scale), int(H * scale))
        resized_frame = cv2.resize(frame, new_size)

        # Deteksi objek dengan YOLOv8
        results = model(resized_frame)

        # Menginisialisasi penghitung orang
        count = 0

        # Menggambar kotak pembatas dan label pada frame
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = box.cls
                label = class_names[int(class_id)]
                confidence = box.conf

                # Menambahkan penghitung orang
                if label == "person":
                    count += 1
                    # Menggambar kotak di sekitar orang
                    label_text = f"{label}: {float(confidence):.2f}"
                    cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    y = y1 - 15 if y1 > 15 else y1 + 15
                    cv2.putText(
                        resized_frame,
                        label_text,
                        (x1, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

        # Menghitung posisi teks tengah
        text = f"{count} orang"
        font_scale = 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, 2)
        position = ((new_size[0] - text_width) // 2, (new_size[1] + text_height) // 2)

        # Menambahkan teks jumlah orang di tengah frame
        cv2.putText(
            resized_frame,
            text,
            position,
            font,
            font_scale,
            (255, 255, 255),
            2,
        )

        # Menampilkan frame
        cv2.imshow("Output", resized_frame)
        out.write(resized_frame)

        # Menunggu penekanan tombol
        key = cv2.waitKey(1) & 0xFF

        # Jika tombol 'q' ditekan, hentikan loop
        if key == ord("q"):
            break

    # Melepaskan sumber daya
    cv2.destroyAllWindows()
    video.release()
    out.release()

    # Menyembunyikan tombol stop
    stop_button.pack_forget()
    image_label.config(image="")

def stop_video():
    # Menetapkan flag stop untuk menghentikan tampilan video
    stop_flag.set()

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
# Menyembunyikan tombol stop di awal
stop_button.pack_forget()

# Menjalankan loop utama aplikasi
root.mainloop()