import threading
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk

# memuat nama-nama kelas COCO
with open("Model/COCO_labels.txt", "r") as f:
    class_names = f.read().split("\n")

# Memuat model deteksi objek OpenCV
net = cv2.dnn.readNet(
    model="Model/frozen_inference_graph_V2.pb",
    config="Model/ssd_mobilenet_v2_coco_2018_03_29.pbtxt",
    framework="TensorFlow",
)


def select_image():
    # Membuka dialog file untuk memilih gambar
    file_path = filedialog.askopenfilename()
    # Memuat gambar yang dipilih
    image = cv2.imread(file_path)
    # Melakukan deteksi dan penghitungan orang
    # Mendapatkan dimensi gambar
    (H, W) = image.shape[:2]

    # Membuat blob dari gambar dan melakukan deteksi objek
    blob = cv2.dnn.blobFromImage(image, size=(300, 300), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward()

    # Menginisialisasi penghitung orang
    count = 0

    # Melakukan loop pada setiap deteksi
    for i in range(0, detections.shape[2]):
        # Mengambil tingkat kepercayaan (probabilitas) dari deteksi
        confidence = detections[0, 0, i, 2]

        # Menyaring deteksi dengan kepercayaan rendah
        if confidence > 0.4:
            # mendapatkan ID kelas
            class_id = detections[0, 0, i, 1]
            # mencocokkan ID kelas dengan nama kelas
            class_name = class_names[int(class_id) - 1]
            # Menghitung koordinat (x, y) kotak deteksi menggunakan dimensi dan posisi tengahnya
            box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")

            if class_name == "person":
                # Menggambar kotak deteksi dan menampilkan label serta kepercayaan
                label = "{}: {:.2f}%".format("person", confidence * 100)
                cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
                y = startY - 15 if startY > 15 else startY + 15
                cv2.putText(
                    image,
                    label,
                    (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
                # Memperbarui penghitung orang
                count += 1

    cv2.putText(
        image,
        f"{count} orang",
        (W - 200, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2,
    )
    # Menampilkan hasil di GUI
    # cv2.imshow("Hasil", image)
    display_image(image)
    cv2.imwrite("ressources/image_result.jpg", image)
    # Menyembunyikan tombol stop
    stop_button.pack_forget()


def select_video():
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
        "ressources/video_result.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        20,
        (frame_width, frame_height),
    )

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

        # Membuat blob dari frame dan melakukan deteksi objek
        blob = cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True, crop=False)
        net.setInput(blob)
        detections = net.forward()

        # Menginisialisasi penghitung orang
        count = 0

        # Melakukan loop pada setiap deteksi
        for i in range(0, detections.shape[2]):
            # Mengambil tingkat kepercayaan (probabilitas) dari deteksi
            confidence = detections[0, 0, i, 2]

            # Menyaring deteksi dengan kepercayaan rendah
            if confidence > 0.4:
                # mendapatkan ID kelas
                class_id = detections[0, 0, i, 1]
                # mencocokkan ID kelas dengan nama kelas
                class_name = class_names[int(class_id) - 1]

                # Mengambil label kelas dan koordinat kotak pembatas
                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype("int")
                # Menambahkan penghitung orang
                if class_name == "person":
                    count += 1
                    # Menggambar kotak di sekitar orang
                    label = "{}: {:.2f}%".format("person", confidence * 100)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    y = startY - 15 if startY > 15 else startY + 15
                    cv2.putText(
                        frame,
                        label,
                        (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

        cv2.putText(
            frame,
            f"{count} orang",
            (W - 200, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )

        # Menampilkan frame
        cv2.imshow("Output", frame)
        # display_image(frame)
        out.write(frame)

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


def display_image(image):
    # Mengubah ukuran gambar agar sesuai dengan ukuran label gambar
    width, height = image_label.winfo_width(), image_label.winfo_height()
    if width > 0 and height > 0:
        image = cv2.resize(image, (width, height))
    # Mengonversi gambar dari BGR ke RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Mengonversi gambar dari array NumPy ke PhotoImage
    image = Image.fromarray(image)
    image = ImageTk.PhotoImage(image)
    # Memperbarui label gambar dengan gambar baru
    image_label.config(image=image)
    image_label.image = image

    image_label.config(image=image)
    image_label.image = image


def select_camera():
    # Menyembunyikan tombol stop
    stop_button.pack_forget()
    video = cv2.VideoCapture(0)
    # mendapatkan lebar dan tinggi frame video untuk penyimpanan yang benar
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    # membuat objek `VideoWriter()`
    out = cv2.VideoWriter(
        "ressources/Camera_result.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        20,
        (frame_width, frame_height),
    )

    # Membaca dan memproses setiap frame
    while True:
        # Membaca frame
        success, frame = video.read()

        # Memeriksa apakah frame berhasil dibaca
        if not success:
            break

        # Mendapatkan dimensi frame
        (H, W) = frame.shape[:2]

        # Membuat blob dari frame dan melakukan deteksi objek
        blob = cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True, crop=False)
        net.setInput(blob)
        detections = net.forward()

        # Menginisialisasi penghitung orang
        count = 0

        # Melakukan loop pada setiap deteksi
        for i in range(0, detections.shape[2]):
            # Mengambil tingkat kepercayaan (probabilitas) dari deteksi
            confidence = detections[0, 0, i, 2]

            # Menyaring deteksi dengan kepercayaan rendah
            if confidence > 0.4:
                # mendapatkan ID kelas
                class_id = detections[0, 0, i, 1]
                # mencocokkan ID kelas dengan nama kelas
                class_name = class_names[int(class_id) - 1]

                # Mengambil label kelas dan koordinat kotak pembatas
                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])

                (startX, startY, endX, endY) = box.astype("int")
                # Menambahkan penghitung orang
                if class_name == "person":
                    count += 1
                    # Menggambar kotak di sekitar orang
                    label = "{}: {:.2f}%".format("person", confidence * 100)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    y = startY - 15 if startY > 15 else startY + 15
                    cv2.putText(
                        frame,
                        label,
                        (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

        cv2.putText(
            frame,
            f"{count} orang",
            (W - 200, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )

        # Menampilkan frame
        cv2.imshow("Output", frame)
        out.write(frame)

        # Menunggu penekanan tombol
        key = cv2.waitKey(1) & 0xFF

        # Jika tombol 'q' ditekan, hentikan loop
        if key == ord("q"):
            break

    # Melepaskan sumber daya
    cv2.destroyAllWindows()
    video.release()
    out.release()


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

# Membuat tombol untuk memilih gambar
image_button = tk.Button(buttons_frame, text="Pilih Gambar", command=select_image)
image_button.pack(side="left")

# Membuat tombol untuk memilih video
video_button = tk.Button(buttons_frame, text="Pilih Video", command=select_video)
video_button.pack(side="left")

# Membuat tombol untuk memilih kamera
camera_button = tk.Button(buttons_frame, text="Pilih Kamera", command=select_camera)
camera_button.pack(side="left")

# Membuat tombol untuk menghentikan video
stop_button = tk.Button(buttons_frame, text="Hentikan Video", command=stop_video)
stop_button.pack(side="left")

# Membuat flag untuk menghentikan thread video
stop_flag = threading.Event()
# Menyembunyikan tombol stop di awal
stop_button.pack_forget()

# Menjalankan loop utama aplikasi
root.mainloop()
