import os
from uuid import uuid4
from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Optional
import time
from skimage.feature import graycomatrix, graycoprops

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

if not os.path.exists("static/uploads"):
    os.makedirs("static/uploads")

if not os.path.exists("static/histograms"):
    os.makedirs("static/histograms")

if not os.path.exists("static/results"):
    os.makedirs("static/results")

# Face Detection related variables and functions
face_detection_data = {
    "active_capture": False,
    "num_images": 0,
    "max_images": 20,
    "current_person": None
}

def detect_faces(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(30, 30)
    )
    return faces

def generate_frames(new_person: str):
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        yield b''
        return

    save_path = os.path.join('dataset', new_person)
    
    try:
        while face_detection_data["active_capture"] and face_detection_data["num_images"] < face_detection_data["max_images"]:
            ret, frame = cap.read()
            if not ret:
                break

            faces = detect_faces(frame)
            
            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    face = frame[y:y+h, x:x+w]
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    
                    img_name = os.path.join(save_path, f"img_{face_detection_data['num_images']}.jpg")
                    cv2.imwrite(img_name, face)
                    face_detection_data["num_images"] += 1

                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    break

            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.1)
    finally:
        cap.release()

@app.get("/face-detection/", response_class=HTMLResponse)
async def face_detection_page(request: Request):
    return templates.TemplateResponse("face_detection.html", {
        "request": request,
        "message": None,
        "video_feed": False
    })

@app.post("/face-detection/", response_class=HTMLResponse)
async def start_face_detection(
    request: Request,
    new_person: str = Form(...)
):
    if not new_person:
        return templates.TemplateResponse("face_detection.html", {
            "request": request,
            "message": "Silakan masukkan nama orang baru.",
            "message_type": "warning",
            "video_feed": False
        })
    
    save_path = os.path.join('dataset', new_person)
    
    if not os.path.exists('dataset'):
        os.makedirs('dataset')
    
    if os.path.exists(save_path):
        return templates.TemplateResponse("face_detection.html", {
            "request": request,
            "message": "Nama sudah ada di dataset. Silakan pilih nama lain.",
            "message_type": "warning",
            "video_feed": False
        })
    
    face_detection_data["active_capture"] = True
    face_detection_data["num_images"] = 0
    face_detection_data["current_person"] = new_person
    
    return templates.TemplateResponse("face_detection.html", {
        "request": request,
        "video_feed": True,
        "new_person": new_person,
        "progress": 0,
        "status": "Memulai deteksi wajah..."
    })

@app.get("/video-feed/{new_person}")
async def video_feed(new_person: str):
    return StreamingResponse(
        generate_frames(new_person),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/face-detection/status")
async def get_detection_status():
    progress = (face_detection_data["num_images"] / face_detection_data["max_images"]) * 100
    return {
        "num_images": face_detection_data["num_images"],
        "progress": progress,
        "status": f"Menyimpan gambar {face_detection_data['num_images']} dari {face_detection_data['max_images']}..."
    }

@app.post("/face-detection/stop")
async def stop_face_detection():
    face_detection_data["active_capture"] = False
    return {"status": "stopped"}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.post("/upload/", response_class=HTMLResponse)
async def upload_image(request: Request, file: UploadFile = File(...)):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    file_path = save_image(img, "uploaded")

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": file_path,
        "modified_image_path": file_path
    })

@app.post("/operation/", response_class=HTMLResponse)
async def perform_operation(
    request: Request,
    file: UploadFile = File(...),
    operation: str = Form(...),
    value: int = Form(...)
):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    original_path = save_image(img, "original")

    if operation == "add":
        result_img = cv2.add(img, np.full(img.shape, value, dtype=np.uint8))
    elif operation == "subtract":
        result_img = cv2.subtract(img, np.full(img.shape, value, dtype=np.uint8))
    elif operation == "multiply":
        result_img = cv2.multiply(img, np.full(img.shape, value, dtype=np.uint8))
    elif operation == "divide":
        # Pastikan tidak ada pembagian dengan nol
        value = max(1, value)
        result_img = cv2.divide(img, np.full(img.shape, value, dtype=np.uint8))
    elif operation == "max":
        result_img = np.maximum(img, np.full(img.shape, value, dtype=np.uint8))
    elif operation == "min":
        result_img = np.minimum(img, np.full(img.shape, value, dtype=np.uint8))
    elif operation == "inverse":
        result_img = cv2.bitwise_not(img)

    modified_path = save_image(result_img, "modified", is_processed=True)

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": original_path,
        "modified_image_path": modified_path
    })

@app.post("/logic_operation/", response_class=HTMLResponse)
async def perform_logic_operation(
    request: Request,
    file1: UploadFile = File(...),
    file2: UploadFile = None,
    operation: str = Form(...)
):
    try:
        # Read first image
        image_data1 = await file1.read()
        np_array1 = np.frombuffer(image_data1, np.uint8)
        img1 = cv2.imdecode(np_array1, cv2.IMREAD_COLOR)
        
        if img1 is None:
            return templates.TemplateResponse("basic_operations.html", {
                "request": request,
                "error": "Tidak dapat membaca gambar pertama"
            })

        # For NOT operation, we don't need second image
        if operation == "not":
            result_img = cv2.bitwise_not(img1)
        else:
            # For other operations, we need second image
            if file2 is None:
                return templates.TemplateResponse("basic_operations.html", {
                    "request": request,
                    "error": "Operasi AND, OR, dan XOR memerlukan dua gambar"
                })
                
            image_data2 = await file2.read()
            np_array2 = np.frombuffer(image_data2, np.uint8)
            img2 = cv2.imdecode(np_array2, cv2.IMREAD_COLOR)
            
            if img2 is None:
                return templates.TemplateResponse("basic_operations.html", {
                    "request": request,
                    "error": "Tidak dapat membaca gambar kedua"
                })
                
            # Check if images have the same dimensions
            if img1.shape != img2.shape:
                return templates.TemplateResponse("basic_operations.html", {
                    "request": request,
                    "error": "Kedua gambar harus memiliki dimensi yang sama"
                })

            # Perform the selected operation
            if operation == "and":
                result_img = cv2.bitwise_and(img1, img2)
            elif operation == "or":
                result_img = cv2.bitwise_or(img1, img2)
            elif operation == "xor":
                result_img = cv2.bitwise_xor(img1, img2)
            else:
                return templates.TemplateResponse("basic_operations.html", {
                    "request": request,
                    "error": "Operasi tidak valid"
                })

        original_path = save_image(img1, "original")
        modified_path = save_image(result_img, "modified", is_processed=True)

        return templates.TemplateResponse("result.html", {
            "request": request,
            "original_image_path": original_path,
            "modified_image_path": modified_path
        })
        
    except Exception as e:
        return templates.TemplateResponse("basic_operations.html", {
            "request": request,
            "error": f"Terjadi kesalahan: {str(e)}"
        })

@app.get("/grayscale/", response_class=HTMLResponse)
async def grayscale_form(request: Request):
    # Menampilkan form untuk upload gambar ke grayscale
    return templates.TemplateResponse("grayscale.html", {"request": request})

@app.post("/grayscale/", response_class=HTMLResponse)
async def convert_grayscale(request: Request, file: UploadFile = File(...)):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    original_path = save_image(img, "original")
    modified_path = save_image(gray_img, "grayscale", is_processed=True)

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": original_path,
        "modified_image_path": modified_path
    })

@app.get("/histogram/", response_class=HTMLResponse)
async def histogram_form(request: Request):
    # Menampilkan halaman untuk upload gambar untuk histogram
    return templates.TemplateResponse("histogram.html", {"request": request})

@app.post("/histogram/", response_class=HTMLResponse)
async def generate_histogram(request: Request, file: UploadFile = File(...)):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    # Pastikan gambar berhasil diimpor
    if img is None:
        return HTMLResponse("Tidak dapat membaca gambar yang diunggah", status_code=400)

    # Buat histogram grayscale dan berwarna
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayscale_histogram_path = save_histogram(gray_img, "grayscale")

    color_histogram_path = save_color_histogram(img)

    return templates.TemplateResponse("histogram.html", {
        "request": request,
        "grayscale_histogram_path": grayscale_histogram_path,
        "color_histogram_path": color_histogram_path
    })

@app.get("/equalize/", response_class=HTMLResponse)
async def equalize_form(request: Request):
    # Menampilkan halaman untuk upload gambar untuk equalisasi histogram
    return templates.TemplateResponse("equalize.html", {"request": request})

@app.post("/equalize/", response_class=HTMLResponse)
async def equalize_histogram(
    request: Request, 
    file: UploadFile = File(...),
    mode: str = Form(...)  # 'grayscale' atau 'color'
):
    try:
        image_data = await file.read()
        np_array = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        
        is_valid, error_msg = validate_image(img)
        if not is_valid:
            return templates.TemplateResponse("equalize.html", {
                "request": request,
                "error": error_msg
            })

        if mode == "grayscale":
            # Konversi ke grayscale dan equalize
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            equalized_img = cv2.equalizeHist(gray_img)
            # Konversi kembali ke BGR untuk konsistensi tampilan
            equalized_img = cv2.cvtColor(equalized_img, cv2.COLOR_GRAY2BGR)
        else:  # color
            # Equalize setiap channel warna secara terpisah
            (b, g, r) = cv2.split(img)
            b_eq = cv2.equalizeHist(b)
            g_eq = cv2.equalizeHist(g)
            r_eq = cv2.equalizeHist(r)
            equalized_img = cv2.merge((b_eq, g_eq, r_eq))

        original_path = save_image(img, "original")
        modified_path = save_image(equalized_img, "equalized", is_processed=True)

        return templates.TemplateResponse("result.html", {
            "request": request,
            "original_image_path": original_path,
            "modified_image_path": modified_path
        })
    except Exception as e:
        return templates.TemplateResponse("equalize.html", {
            "request": request,
            "error": f"Terjadi kesalahan: {str(e)}"
        })

@app.get("/specify/", response_class=HTMLResponse)
async def specify_form(request: Request):
    # Menampilkan halaman untuk upload gambar dan referensi untuk spesifikasi histogram
    return templates.TemplateResponse("specify.html", {"request": request})

@app.post("/specify/", response_class=HTMLResponse)
async def specify_histogram(
    request: Request, 
    file: UploadFile = File(...), 
    ref_file: UploadFile = File(...),
    mode: str = Form(...)  # 'grayscale' atau 'color'
):
    try:
        # Baca gambar sumber
        image_data = await file.read()
        np_array = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        
        # Baca gambar referensi
        ref_image_data = await ref_file.read()
        ref_np_array = np.frombuffer(ref_image_data, np.uint8)
        ref_img = cv2.imdecode(ref_np_array, cv2.IMREAD_COLOR)
        
        # Validasi kedua gambar
        is_valid, error_msg = validate_image(img)
        if not is_valid:
            return templates.TemplateResponse("specify.html", {
                "request": request,
                "error": error_msg
            })
            
        is_valid, error_msg = validate_image(ref_img)
        if not is_valid:
            return templates.TemplateResponse("specify.html", {
                "request": request,
                "error": "Gambar referensi: " + error_msg
            })

        if mode == "grayscale":
            # Konversi ke grayscale
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
            
            # Hitung CDF untuk kedua gambar
            hist_img = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
            hist_ref = cv2.calcHist([ref_gray], [0], None, [256], [0, 256])
            
            cdf_img = hist_img.cumsum()
            cdf_ref = hist_ref.cumsum()
            
            # Normalisasi CDF
            cdf_img = (cdf_img - cdf_img.min()) * 255 / (cdf_img.max() - cdf_img.min())
            cdf_ref = (cdf_ref - cdf_ref.min()) * 255 / (cdf_ref.max() - cdf_ref.min())
            
            # Buat lookup table
            lookup_table = np.zeros(256)
            j = 0
            for i in range(256):
                while j < 256 and cdf_ref[j] <= cdf_img[i]:
                    j += 1
                lookup_table[i] = j - 1 if j > 0 else 0
            
            # Aplikasikan spesifikasi histogram
            specified_img = cv2.LUT(gray_img, lookup_table.astype(np.uint8))
            # Konversi kembali ke BGR untuk konsistensi tampilan
            specified_img = cv2.cvtColor(specified_img, cv2.COLOR_GRAY2BGR)
            
        else:  # color
            # Proses setiap channel warna secara terpisah
            (b, g, r) = cv2.split(img)
            (ref_b, ref_g, ref_r) = cv2.split(ref_img)
            
            channels = []
            for src_channel, ref_channel in zip([b, g, r], [ref_b, ref_g, ref_r]):
                # Hitung CDF untuk kedua channel
                hist_src = cv2.calcHist([src_channel], [0], None, [256], [0, 256])
                hist_ref = cv2.calcHist([ref_channel], [0], None, [256], [0, 256])
                
                cdf_src = hist_src.cumsum()
                cdf_ref = hist_ref.cumsum()
                
                # Normalisasi CDF
                cdf_src = (cdf_src - cdf_src.min()) * 255 / (cdf_src.max() - cdf_src.min())
                cdf_ref = (cdf_ref - cdf_ref.min()) * 255 / (cdf_ref.max() - cdf_ref.min())
                
                # Buat lookup table
                lookup_table = np.zeros(256)
                j = 0
                for i in range(256):
                    while j < 256 and cdf_ref[j] <= cdf_src[i]:
                        j += 1
                    lookup_table[i] = j - 1 if j > 0 else 0
                
                # Aplikasikan spesifikasi histogram
                specified_channel = cv2.LUT(src_channel, lookup_table.astype(np.uint8))
                channels.append(specified_channel)
            
            # Gabungkan kembali channel-channel yang telah diproses
            specified_img = cv2.merge(channels)

        original_path = save_image(img, "original")
        modified_path = save_image(specified_img, "specified", is_processed=True)

        return templates.TemplateResponse("result.html", {
            "request": request,
            "original_image_path": original_path,
            "modified_image_path": modified_path
        })
    except Exception as e:
        return templates.TemplateResponse("specify.html", {
            "request": request,
            "error": f"Terjadi kesalahan: {str(e)}"
        })

@app.post("/statistics/", response_class=HTMLResponse)
async def calculate_statistics(request: Request, file: UploadFile = File(...)):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_GRAYSCALE)

    mean_intensity = np.mean(img)
    std_deviation = np.std(img)

    image_path = save_image(img, "statistics")

    return templates.TemplateResponse("statistics.html", {
        "request": request,
        "mean_intensity": mean_intensity,
        "std_deviation": std_deviation,
        "image_path": image_path
    })
    
@app.get("/convolution/", response_class=HTMLResponse)
async def convolution_form(request: Request):
    return templates.TemplateResponse("convolution.html", {"request": request})

@app.post("/convolution/", response_class=HTMLResponse)
async def apply_convolution(
    request: Request,
    file: UploadFile = File(...),
    kernel_type: str = Form(...)
):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if kernel_type == "average":
        kernel = np.ones((3, 3), np.float32) / 9
    elif kernel_type == "sharpen":
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    elif kernel_type == "edge":
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

    result_img = cv2.filter2D(img, -1, kernel)

    original_path = save_image(img, "original")
    modified_path = save_image(result_img, "convolution", is_processed=True)

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": original_path,
        "modified_image_path": modified_path
    })

@app.get("/padding/", response_class=HTMLResponse)
async def padding_form(request: Request):
    return templates.TemplateResponse("padding.html", {"request": request})

@app.post("/padding/", response_class=HTMLResponse)
async def apply_padding(
    request: Request,
    file: UploadFile = File(...),
    padding_size: int = Form(...)
):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    # Terapkan zero padding
    padded_img = cv2.copyMakeBorder(
        img,
        padding_size,
        padding_size,
        padding_size,
        padding_size,
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0]
    )

    original_path = save_image(img, "original")
    modified_path = save_image(padded_img, "padded", is_processed=True)

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": original_path,
        "modified_image_path": modified_path
    })

@app.get("/filter/", response_class=HTMLResponse)
async def filter_form(request: Request):
    return templates.TemplateResponse("filter.html", {"request": request})

@app.post("/filter/", response_class=HTMLResponse)
async def apply_filter(
    request: Request,
    file: UploadFile = File(...),
    filter_type: str = Form(...),
    kernel_size: int = Form(...)
):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if filter_type == "gaussian":
        filtered_img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    elif filter_type == "median":
        filtered_img = cv2.medianBlur(img, kernel_size)
    elif filter_type == "bilateral":
        filtered_img = cv2.bilateralFilter(img, kernel_size, 75, 75)

    original_path = save_image(img, "original")
    modified_path = save_image(filtered_img, "filtered", is_processed=True)

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": original_path,
        "modified_image_path": modified_path
    })

@app.get("/fourier/", response_class=HTMLResponse)
async def fourier_form(request: Request):
    return templates.TemplateResponse("fourier.html", {"request": request})

@app.post("/fourier/", response_class=HTMLResponse)
async def apply_fourier(
    request: Request,
    file: UploadFile = File(...),
    transform_type: str = Form(...)
):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_GRAYSCALE)

    # Aplikasikan FFT
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    
    magnitude_spectrum = None
    phase_spectrum = None
    
    if transform_type in ["magnitude", "both"]:
        magnitude_spectrum = 20 * np.log(np.abs(fshift))
        magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    if transform_type in ["phase", "both"]:
        phase_spectrum = np.angle(fshift)
        phase_spectrum = cv2.normalize(phase_spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    original_path = save_image(img, "original")
    
    if transform_type == "both":
        # Gabungkan magnitude dan phase spectrum secara horizontal
        combined = np.hstack((magnitude_spectrum, phase_spectrum))
        modified_path = save_image(combined, "fourier", is_processed=True)
    else:
        result_img = magnitude_spectrum if transform_type == "magnitude" else phase_spectrum
        modified_path = save_image(result_img, "fourier", is_processed=True)

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": original_path,
        "modified_image_path": modified_path
    })

@app.get("/compression/", response_class=HTMLResponse)
async def compression_form(request: Request):
    return templates.TemplateResponse("compression.html", {
        "request": request
    })

@app.post("/compression/", response_class=HTMLResponse)
async def compress_image(
    request: Request,
    file: UploadFile = File(...),
    method: str = Form(...),
    quality: int = Form(None),
    level: int = Form(None)
):
    try:
        # Baca gambar
        image_data = await file.read()
        np_array = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        
        if img is None:
            return templates.TemplateResponse("compression.html", {
                "request": request,
                "error": "Gagal membaca gambar"
            })

        # Simpan gambar original
        original_path = save_image(img, "original")
        original_size = os.path.getsize(os.path.join("static/uploads", os.path.basename(original_path)))
        
        if method == "jpeg":
            quality = quality if quality is not None else 95
            # Simpan sebagai JPEG
            jpeg_path = f"static/results/compressed_{uuid4()}.jpg"
            cv2.imwrite(jpeg_path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
            compressed_size = os.path.getsize(jpeg_path)
            
            # Baca kembali untuk analisis
            compressed_img = cv2.imread(jpeg_path)
            
            # Hitung metrik
            psnr = cv2.PSNR(img, compressed_img)
            mse = np.mean((img.astype(float) - compressed_img.astype(float)) ** 2)
            ratio = original_size / compressed_size if compressed_size > 0 else float('inf')
            
            return templates.TemplateResponse("compression.html", {
                "request": request,
                "original_image_path": original_path,
                "compressed_image_path": f"/{jpeg_path}",
                "analysis": {
                    "original_size": f"{original_size / 1024:.2f}",
                    "compressed_size": f"{compressed_size / 1024:.2f}",
                    "compression_ratio": f"{ratio:.2f}",
                    "psnr": f"{psnr:.2f}",
                    "mse": f"{mse:.2f}",
                    "method": "JPEG",
                    "quality": quality
                }
            })
        else:  # PNG
            level = level if level is not None else 9
            # Simpan sebagai PNG
            png_path = f"static/results/compressed_{uuid4()}.png"
            cv2.imwrite(png_path, img, [cv2.IMWRITE_PNG_COMPRESSION, level])
            compressed_size = os.path.getsize(png_path)
            
            # Baca kembali untuk analisis
            compressed_img = cv2.imread(png_path)
            
            # Hitung metrik
            is_identical = np.array_equal(img, compressed_img)
            ratio = original_size / compressed_size if compressed_size > 0 else float('inf')
            
            return templates.TemplateResponse("compression.html", {
                "request": request,
                "original_image_path": original_path,
                "compressed_image_path": f"/{png_path}",
                "analysis": {
                    "original_size": f"{original_size / 1024:.2f}",
                    "compressed_size": f"{compressed_size / 1024:.2f}",
                    "compression_ratio": f"{ratio:.2f}",
                    "is_identical": is_identical,
                    "method": "PNG",
                    "level": level
                }
            })
            
    except Exception as e:
        return templates.TemplateResponse("compression.html", {
            "request": request,
            "error": f"Error: {str(e)}"
        })

def save_image(image, prefix, is_processed=False):
    filename = f"{prefix}_{uuid4()}.png"
    if is_processed:
        path = os.path.join("static/results", filename)
        return_path = f"/static/results/{filename}"
    else:
        path = os.path.join("static/uploads", filename)
        return_path = f"/static/uploads/{filename}"
    cv2.imwrite(path, image)
    return return_path

def save_histogram(image, prefix):
    histogram_path = f"static/histograms/{prefix}_{uuid4()}.png"
    plt.figure()
    plt.hist(image.ravel(), 256, [0, 256])
    plt.savefig(histogram_path)
    plt.close()
    return f"/{histogram_path}"

def save_color_histogram(image):
    color_histogram_path = f"static/histograms/color_{uuid4()}.png"
    plt.figure()
    for i, color in enumerate(['b', 'g', 'r']):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=color)
    plt.savefig(color_histogram_path)
    plt.close()
    return f"/{color_histogram_path}"

# Face Processing Routes and Functions
@app.get("/face-processing/", response_class=HTMLResponse)
async def face_processing_page(request: Request):
    return templates.TemplateResponse("face_processing.html", {
        "request": request
    })

def add_salt_pepper_noise(image, amount):
    output = np.copy(image)
    # Salt noise
    num_salt = np.ceil(amount * image.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    output[coords[0], coords[1]] = 255
    
    # Pepper noise
    num_pepper = np.ceil(amount * image.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    output[coords[0], coords[1]] = 0
    return output

def validate_image(image):
    """Validasi format dan ukuran gambar"""
    if image is None:
        return False, "Gambar tidak valid"
    if image.shape[0] > 2000 or image.shape[1] > 2000:
        return False, "Ukuran gambar terlalu besar"
    return True, None

def cleanup_old_files():
    """Membersihkan file hasil pengolahan yang lebih dari 1 jam"""
    current_time = time.time()
    for directory in ["static/uploads", "static/results", "static/histograms"]:
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.getmtime(filepath) < current_time - 3600:  # 1 jam
                os.remove(filepath)

@app.on_event("startup")
async def startup_event():
    """Jalankan cleanup saat aplikasi dimulai"""
    cleanup_old_files()

@app.post("/face-processing/noise", response_class=HTMLResponse)
async def add_noise(
    request: Request,
    file: UploadFile = File(...),
    amount: float = Form(...)
):
    try:
        if amount < 0 or amount > 0.1:
            return templates.TemplateResponse("face_processing.html", {
                "request": request,
                "error": "Jumlah noise harus antara 0 dan 0.1"
            })

        image_data = await file.read()
        np_array = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        
        is_valid, error_msg = validate_image(img)
        if not is_valid:
            return templates.TemplateResponse("face_processing.html", {
                "request": request,
                "error": error_msg
            })
        
        # Add salt and pepper noise
        noisy_img = add_salt_pepper_noise(img, amount)
        
        original_path = save_image(img, "original")
        modified_path = save_image(noisy_img, "noisy", is_processed=True)
        
        return templates.TemplateResponse("face_processing.html", {
            "request": request,
            "original_image_path": original_path,
            "modified_image_path": modified_path
        })
    except Exception as e:
        return templates.TemplateResponse("face_processing.html", {
            "request": request,
            "error": f"Terjadi kesalahan: {str(e)}"
        })

@app.post("/face-processing/denoise", response_class=HTMLResponse)
async def remove_noise(
    request: Request,
    file: UploadFile = File(...),
    method: str = Form(...)
):
    try:
        image_data = await file.read()
        np_array = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        
        is_valid, error_msg = validate_image(img)
        if not is_valid:
            return templates.TemplateResponse("face_processing.html", {
                "request": request,
                "error": error_msg
            })
        
        # Apply denoising based on selected method
        if method == "gaussian":
            denoised_img = cv2.GaussianBlur(img, (5, 5), 0)
        elif method == "median":
            denoised_img = cv2.medianBlur(img, 5)
        elif method == "bilateral":
            denoised_img = cv2.bilateralFilter(img, 9, 75, 75)
        else:
            return templates.TemplateResponse("face_processing.html", {
                "request": request,
                "error": "Metode penghilangan noise tidak valid"
            })
        
        original_path = save_image(img, "original")
        modified_path = save_image(denoised_img, "denoised", is_processed=True)
        
        return templates.TemplateResponse("face_processing.html", {
            "request": request,
            "original_image_path": original_path,
            "modified_image_path": modified_path
        })
    except Exception as e:
        return templates.TemplateResponse("face_processing.html", {
            "request": request,
            "error": f"Terjadi kesalahan: {str(e)}"
        })

@app.post("/face-processing/sharpen", response_class=HTMLResponse)
async def sharpen_image(
    request: Request,
    file: UploadFile = File(...),
    method: str = Form(...)
):
    try:
        image_data = await file.read()
        np_array = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        
        is_valid, error_msg = validate_image(img)
        if not is_valid:
            return templates.TemplateResponse("face_processing.html", {
                "request": request,
                "error": error_msg
            })
        
        if method == "laplacian":
            # Laplacian sharpening
            kernel = np.array([[0, -1, 0],
                             [-1, 5, -1],
                             [0, -1, 0]])
            sharpened_img = cv2.filter2D(img, -1, kernel)
        elif method == "unsharp":
            # Generate a blurred version of the image
            gaussian = cv2.GaussianBlur(img, (5, 5), 0)
            # Calculate the unsharp mask
            sharpened_img = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)
        else:
            return templates.TemplateResponse("face_processing.html", {
                "request": request,
                "error": "Metode penajaman tidak valid"
            })
        
        original_path = save_image(img, "original")
        modified_path = save_image(sharpened_img, "sharpened", is_processed=True)
        
        return templates.TemplateResponse("face_processing.html", {
            "request": request,
            "original_image_path": original_path,
            "modified_image_path": modified_path
        })
    except Exception as e:
        return templates.TemplateResponse("face_processing.html", {
            "request": request,
            "error": f"Terjadi kesalahan: {str(e)}"
        })

@app.get("/basic-operations/", response_class=HTMLResponse)
async def basic_operations_page(request: Request):
    return templates.TemplateResponse("basic_operations.html", {
        "request": request
    })

@app.get("/reduce-noise/", response_class=HTMLResponse)
async def reduce_noise_page():
    return templates.TemplateResponse("reduce_noise.html", {"request": {}})

@app.post("/reduce-noise/", response_class=HTMLResponse)
async def reduce_noise(
    file: UploadFile = File(...),
    d0: int = Form(10),
    n: int = Form(4),
    uk: int = Form(50),
    vk: int = Form(50)
):
    try:
        # Baca gambar
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return templates.TemplateResponse(
                "reduce_noise.html",
                {"request": {}, "error": "Gagal membaca gambar"}
            )
        
        # Konversi ke grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Lakukan FFT
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        
        # Hitung magnitude spectrum
        magnitude_spectrum = np.log(np.abs(fshift) + 1)
        magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Buat filter notch
        rows, cols = gray.shape
        crow, ccol = rows//2, cols//2
        u = np.arange(rows)
        v = np.arange(cols)
        u, v = np.meshgrid(u - crow, v - ccol)
        
        # Hitung jarak dari pusat
        d = np.sqrt(u**2 + v**2)
        
        # Buat filter notch
        h = 1 / (1 + (d0**2 / (d**2 + 1e-10))**n)
        
        # Terapkan filter
        fshift_filtered = fshift * h
        
        # Inverse FFT
        f_ishift = np.fft.ifftshift(fshift_filtered)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Simpan gambar
        original_path = save_image(img, "original")
        spectrum_path = save_image(magnitude_spectrum, "spectrum", is_processed=True)
        result_path = save_image(img_back, "reduced_noise", is_processed=True)
        
        return templates.TemplateResponse(
            "reduce_noise.html",
            {
                "request": {},
                "original_image_path": original_path,
                "spectrum_path": spectrum_path,
                "modified_image_path": result_path
            }
        )
        
    except Exception as e:
        return templates.TemplateResponse(
            "reduce_noise.html",
            {"request": {}, "error": f"Error: {str(e)}"}
        )

@app.get("/image-analysis/", response_class=HTMLResponse)
async def image_analysis_form(request: Request):
    return templates.TemplateResponse("image_analysis.html", {
        "request": request
    })

@app.post("/image-analysis/freeman/", response_class=HTMLResponse)
async def freeman_chain_code(
    request: Request,
    file: UploadFile = File(...),
    threshold: int = Form(127),
    invert: bool = Form(False)
):
    try:
        image_data = await file.read()
        np_array = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_array, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            return templates.TemplateResponse("image_analysis.html", {
                "request": request,
                "error": "Gagal membaca gambar"
            })

        mode_thresh = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        _, binary = cv2.threshold(img, threshold, 255, mode_thresh)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if not contours:
            return templates.TemplateResponse("image_analysis.html", {
                "request": request,
                "error": "Tidak ada kontur yang terdeteksi"
            })
            
        largest = max(contours, key=cv2.contourArea)
        overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(overlay, [largest], -1, (0, 255, 0), 2)
        
        # Hitung Freeman Chain Code
        directions = {
            (1, 0): 0, (1, 1): 1, (0, 1): 2, (-1, 1): 3,
            (-1, 0): 4, (-1, -1): 5, (0, -1): 6, (1, -1): 7
        }
        chain_code = []
        for i in range(len(largest)):
            p1 = largest[i][0]
            p2 = largest[(i+1)%len(largest)][0]
            dx, dy = np.sign(p2[0]-p1[0]), np.sign(p2[1]-p1[1])
            code = directions.get((dx,dy))
            if code is not None:
                chain_code.append(code)
        
        # Evaluasi kontur
        area = cv2.contourArea(largest)
        perimeter = cv2.arcLength(largest, True)
        ratio = len(chain_code)/perimeter if perimeter>0 else 0
        
        # Save images
        original_path = save_image(img, "original")
        binary_path = save_image(binary, "binary")
        contour_path = save_image(overlay, "contour")
        
        return templates.TemplateResponse("image_analysis.html", {
            "request": request,
            "original_image_path": original_path,
            "binary_image_path": binary_path,
            "contour_image_path": contour_path,
            "chain_code": chain_code,
            "analysis": {
                "area": f"{area:.2f}",
                "perimeter": f"{perimeter:.2f}",
                "chain_code_length": len(chain_code),
                "ratio": f"{ratio:.2f}"
            },
            "mode": "freeman"
        })
    except Exception as e:
        return templates.TemplateResponse("image_analysis.html", {
            "request": request,
            "error": f"Error: {str(e)}"
        })

@app.post("/image-analysis/crack/", response_class=HTMLResponse)
async def crack_code(
    request: Request,
    file: UploadFile = File(...),
    low_threshold: int = Form(50),
    high_threshold: int = Form(150),
    kernel_size: int = Form(5)
):
    try:
        image_data = await file.read()
        np_array = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_array, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            return templates.TemplateResponse("image_analysis.html", {
                "request": request,
                "error": "Gagal membaca gambar"
            })
        
        blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        edges = cv2.Canny(blurred, low_threshold, high_threshold)
        
        count_edges = int(np.count_nonzero(edges))
        total_px = edges.size
        percent_edges = count_edges / total_px * 100
        
        original_path = save_image(img, "original")
        blur_path = save_image(blurred, "blurred")
        edges_path = save_image(edges, "edges")
        
        return templates.TemplateResponse("image_analysis.html", {
            "request": request,
            "original_image_path": original_path,
            "blur_image_path": blur_path,
            "edges_image_path": edges_path,
            "analysis": {
                "edge_pixels": count_edges,
                "total_pixels": total_px,
                "edge_percentage": f"{percent_edges:.2f}"
            },
            "mode": "crack"
        })
    except Exception as e:
        return templates.TemplateResponse("image_analysis.html", {
            "request": request,
            "error": f"Error: {str(e)}"
        })

@app.post("/image-analysis/projection/", response_class=HTMLResponse)
async def projection_analysis(
    request: Request,
    file: UploadFile = File(...),
    threshold: int = Form(127),
    invert: bool = Form(False)
):
    try:
        image_data = await file.read()
        np_array = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_array, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            return templates.TemplateResponse("image_analysis.html", {
                "request": request,
                "error": "Gagal membaca gambar"
            })
        
        mode_thresh = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        _, binary = cv2.threshold(img, threshold, 255, mode_thresh)
        binary_norm = binary / 255.0
        
        # Hitung proyeksi
        horiz = np.sum(binary_norm, axis=0)
        vert = np.sum(binary_norm, axis=1)
        
        # Plot dan simpan proyeksi
        plt.figure()
        plt.plot(horiz)
        plt.title("Proyeksi Horizontal")
        horiz_path = f"static/results/horiz_{uuid4()}.png"
        plt.savefig(horiz_path)
        plt.close()
        
        plt.figure()
        plt.plot(vert)
        plt.title("Proyeksi Vertikal")
        vert_path = f"static/results/vert_{uuid4()}.png"
        plt.savefig(vert_path)
        plt.close()
        
        max_h, idx_h = float(np.max(horiz)), int(np.argmax(horiz))
        max_v, idx_v = float(np.max(vert)), int(np.argmax(vert))
        
        original_path = save_image(img, "original")
        binary_path = save_image(binary, "binary")
        
        return templates.TemplateResponse("image_analysis.html", {
            "request": request,
            "original_image_path": original_path,
            "binary_image_path": binary_path,
            "horiz_proj_path": f"/{horiz_path}",
            "vert_proj_path": f"/{vert_path}",
            "analysis": {
                "width": binary.shape[1],
                "height": binary.shape[0],
                "max_horiz": f"{max_h:.2f}",
                "max_horiz_idx": idx_h,
                "max_vert": f"{max_v:.2f}",
                "max_vert_idx": idx_v
            },
            "mode": "projection"
        })
    except Exception as e:
        return templates.TemplateResponse("image_analysis.html", {
            "request": request,
            "error": f"Error: {str(e)}"
        })

# Color Spaces Routes and Functions
@app.get("/color-spaces/", response_class=HTMLResponse)
async def color_spaces_page(request: Request):
    return templates.TemplateResponse("color_spaces.html", {
        "request": request
    })

@app.post("/color-spaces/convert/", response_class=HTMLResponse)
async def convert_color_space(
    request: Request,
    file: UploadFile = File(...),
    target_space: str = Form(...)
):
    try:
        image_data = await file.read()
        np_array = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        
        if img is None:
            return templates.TemplateResponse("color_spaces.html", {
                "request": request,
                "error": "Gagal membaca gambar"
            })

        # Konversi ruang warna
        if target_space == "xyz":
            converted = cv2.cvtColor(img, cv2.COLOR_BGR2XYZ)
        elif target_space == "lab":
            converted = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        elif target_space == "ycbcr":
            converted = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        elif target_space == "hsv":
            converted = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif target_space == "hsl":
            converted = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        elif target_space == "luv":
            converted = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
        elif target_space == "yiq":
            # YIQ = YUV * transformation matrix
            yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            y, u, v = cv2.split(yuv)
            i = 0.74 * (0.877 * u - 0.493 * v)
            q = 0.48 * (0.493 * u + 0.877 * v)
            converted = cv2.merge([y, i.astype(np.uint8), q.astype(np.uint8)])
        elif target_space == "yuv":
            converted = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        
        original_path = save_image(img, "original")
        converted_path = save_image(converted, "converted", is_processed=True)
        
        return templates.TemplateResponse("color_spaces.html", {
            "request": request,
            "original_image_path": original_path,
            "converted_image_path": converted_path,
            "target_space": target_space.upper(),
            "mode": "convert"
        })
    except Exception as e:
        return templates.TemplateResponse("color_spaces.html", {
            "request": request,
            "error": f"Error: {str(e)}"
        })

@app.post("/color-spaces/channels/", response_class=HTMLResponse)
async def analyze_channels(
    request: Request,
    file: UploadFile = File(...),
    color_space: str = Form(...)
):
    try:
        image_data = await file.read()
        np_array = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        
        if img is None:
            return templates.TemplateResponse("color_spaces.html", {
                "request": request,
                "error": "Gagal membaca gambar"
            })

        # Konversi dan pisahkan kanal
        if color_space == "rgb":
            channels = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            channel_names = ["Red", "Green", "Blue"]
        elif color_space == "lab":
            channels = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
            channel_names = ["Lightness", "a* (green-red)", "b* (blue-yellow)"]
        elif color_space == "hsv":
            channels = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
            channel_names = ["Hue", "Saturation", "Value"]
        elif color_space == "ycbcr":
            channels = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb))
            channel_names = ["Y (Luminance)", "Cr (Red-diff)", "Cb (Blue-diff)"]
        
        # Simpan gambar asli dan kanal-kanalnya
        original_path = save_image(img, "original")
        channel_paths = []
        
        for i, (channel, name) in enumerate(zip(channels, channel_names)):
            path = save_image(channel, f"channel_{i}", is_processed=True)
            channel_paths.append({
                "name": name,
                "path": path
            })
        
        return templates.TemplateResponse("color_spaces.html", {
            "request": request,
            "original_image_path": original_path,
            "channels": channel_paths,
            "mode": "channels"
        })
    except Exception as e:
        return templates.TemplateResponse("color_spaces.html", {
            "request": request,
            "error": f"Error: {str(e)}"
        })

@app.post("/color-spaces/luminance/", response_class=HTMLResponse)
async def analyze_luminance(
    request: Request,
    file: UploadFile = File(...)
):
    try:
        image_data = await file.read()
        np_array = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        
        if img is None:
            return templates.TemplateResponse("color_spaces.html", {
                "request": request,
                "error": "Gagal membaca gambar"
            })

        # Konversi ke YCrCb dan ambil komponen Y (luminance)
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        luminance = ycrcb[:,:,0]
        
        # Hitung statistik luminance
        mean_lum = np.mean(luminance)
        std_lum = np.std(luminance)
        min_lum = np.min(luminance)
        max_lum = np.max(luminance)
        
        original_path = save_image(img, "original")
        luminance_path = save_image(luminance, "luminance", is_processed=True)
        
        return templates.TemplateResponse("color_spaces.html", {
            "request": request,
            "original_image_path": original_path,
            "luminance_path": luminance_path,
            "analysis": {
                "mean_luminance": f"{mean_lum:.2f}",
                "std_luminance": f"{std_lum:.2f}",
                "min_luminance": f"{min_lum:.2f}",
                "max_luminance": f"{max_lum:.2f}"
            },
            "mode": "luminance"
        })
    except Exception as e:
        return templates.TemplateResponse("color_spaces.html", {
            "request": request,
            "error": f"Error: {str(e)}"
        })

# Texture Analysis Routes and Functions
@app.get("/texture/", response_class=HTMLResponse)
async def texture_page(request: Request):
    return templates.TemplateResponse("texture_analysis.html", {
        "request": request
    })

@app.post("/texture/statistical/", response_class=HTMLResponse)
async def analyze_texture_statistical(
    request: Request,
    file: UploadFile = File(...),
    method: str = Form(...),
    direction: Optional[int] = Form(None)
):
    try:
        image_data = await file.read()
        np_array = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_array, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            return templates.TemplateResponse("texture_analysis.html", {
                "request": request,
                "error": "Gagal membaca gambar"
            })

        if method == "first_order":
            # Hitung statistik first-order
            mean = np.mean(img)
            variance = np.var(img)
            skewness = np.mean((img - mean) ** 3) / (np.std(img) ** 3)
            kurtosis = np.mean((img - mean) ** 4) / (np.std(img) ** 4)
            
            # Hitung energy dan entropy
            hist = cv2.calcHist([img], [0], None, [256], [0, 256])
            hist_norm = hist / hist.sum()
            energy = np.sum(hist_norm ** 2)
            entropy = -np.sum(hist_norm * np.log2(hist_norm + np.finfo(float).eps))
            
            analysis = {
                "mean": f"{mean:.2f}",
                "variance": f"{variance:.2f}",
                "skewness": f"{skewness:.2f}",
                "kurtosis": f"{kurtosis:.2f}",
                "energy": f"{energy:.2f}",
                "entropy": f"{entropy:.2f}"
            }
            glcm_path = None
            
        elif method == "glcm":
            # Hitung GLCM
            distances = [1]
            angles = [np.pi/4 * direction] if direction is not None else [0, np.pi/4, np.pi/2, 3*np.pi/4]
            glcm = graycomatrix(img, distances, angles, 256, symmetric=True, normed=True)
            
            # Hitung fitur GLCM
            contrast = graycoprops(glcm, 'contrast')[0, 0]
            correlation = graycoprops(glcm, 'correlation')[0, 0]
            energy = graycoprops(glcm, 'energy')[0, 0]
            homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
            
            analysis = {
                "contrast": f"{contrast:.2f}",
                "correlation": f"{correlation:.2f}",
                "energy": f"{energy:.2f}",
                "homogeneity": f"{homogeneity:.2f}"
            }
            
            # Visualisasi GLCM
            plt.figure()
            plt.imshow(glcm[:,:,0,0], cmap='gray')
            plt.colorbar()
            plt.title('GLCM')
            glcm_path = f"static/results/glcm_{uuid4()}.png"
            plt.savefig(glcm_path)
            plt.close()
            glcm_path = f"/{glcm_path}"
            
        else:  # GLRLM
            # Implementasi GLRLM
            def calculate_glrlm(img, angle):
                if angle == 0:  # horizontal
                    return np.apply_along_axis(lambda x: np.bincount(len(x)), 1, img)
                elif angle == 90:  # vertical
                    return np.apply_along_axis(lambda x: np.bincount(len(x)), 0, img)
                
            glrlm = calculate_glrlm(img, 0)  # horizontal direction
            
            # Hitung fitur GLRLM
            total_runs = np.sum(glrlm)
            sre = np.sum(glrlm / (np.arange(glrlm.shape[0]) ** 2 + 1e-6)) / total_runs
            lre = np.sum(glrlm * (np.arange(glrlm.shape[0]) ** 2)) / total_runs
            gln = np.sum(np.sum(glrlm, axis=1) ** 2) / total_runs
            rln = np.sum(np.sum(glrlm, axis=0) ** 2) / total_runs
            
            analysis = {
                "sre": f"{sre:.2f}",
                "lre": f"{lre:.2f}",
                "gln": f"{gln:.2f}",
                "rln": f"{rln:.2f}"
            }
            
            # Visualisasi GLRLM
            plt.figure()
            plt.imshow(glrlm, cmap='gray')
            plt.colorbar()
            plt.title('GLRLM')
            glrlm_path = f"static/results/glrlm_{uuid4()}.png"
            plt.savefig(glrlm_path)
            plt.close()
            glcm_path = f"/{glrlm_path}"
        
        original_path = save_image(img, "original")
        
        return templates.TemplateResponse("texture_analysis.html", {
            "request": request,
            "original_image_path": original_path,
            "glcm_path": glcm_path,
            "glrlm_path": glcm_path if method == "glrlm" else None,
            "analysis": analysis,
            "method": method,
            "mode": "statistical"
        })
    except Exception as e:
        return templates.TemplateResponse("texture_analysis.html", {
            "request": request,
            "error": f"Error: {str(e)}"
        })

@app.post("/texture/spectral/", response_class=HTMLResponse)
async def analyze_texture_spectral(
    request: Request,
    file: UploadFile = File(...),
    method: str = Form(...),
    orientation: Optional[float] = Form(None),
    frequency: Optional[float] = Form(None)
):
    try:
        image_data = await file.read()
        np_array = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_array, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            return templates.TemplateResponse("texture_analysis.html", {
                "request": request,
                "error": "Gagal membaca gambar"
            })

        if method == "fourier":
            # Aplikasikan FFT
            f = np.fft.fft2(img)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = 20 * np.log(np.abs(fshift))
            
            # Normalisasi untuk visualisasi
            magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            spectral_path = save_image(magnitude_spectrum, "fourier", is_processed=True)
            gabor_filters = None
            
        else:  # Gabor
            # Buat kernel Gabor
            ksize = 31
            sigma = 5
            theta = np.radians(float(orientation)) if orientation is not None else 0
            lambd = 1/float(frequency) if frequency is not None else 10
            gamma = 0.5
            psi = 0
            
            kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi)
            
            # Aplikasikan filter Gabor
            filtered = cv2.filter2D(img, cv2.CV_8UC3, kernel)
            
            # Generate filter bank
            orientations = [0, 45, 90, 135]
            frequencies = [0.1, 0.5, 1.0]
            gabor_filters = []
            
            for theta in orientations:
                for freq in frequencies:
                    kernel = cv2.getGaborKernel(
                        (ksize, ksize), sigma, np.radians(theta), 1/freq, gamma, psi
                    )
                    kernel_norm = cv2.normalize(kernel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    path = save_image(kernel_norm, f"gabor_filter_{theta}_{freq}", is_processed=True)
                    gabor_filters.append({
                        "path": path,
                        "params": f"θ={theta}°, f={freq:.1f}"
                    })
            
            spectral_path = save_image(filtered, "gabor", is_processed=True)
        
        original_path = save_image(img, "original")
        
        return templates.TemplateResponse("texture_analysis.html", {
            "request": request,
            "original_image_path": original_path,
            "spectral_path": spectral_path,
            "gabor_filters": gabor_filters,
            "method": method,
            "mode": "spectral"
        })
    except Exception as e:
        return templates.TemplateResponse("texture_analysis.html", {
            "request": request,
            "error": f"Error: {str(e)}"
        })

@app.post("/texture/structural/", response_class=HTMLResponse)
async def analyze_texture_structural(
    request: Request,
    file: UploadFile = File(...),
    method: str = Form(...)
):
    try:
        image_data = await file.read()
        np_array = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_array, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            return templates.TemplateResponse("texture_analysis.html", {
                "request": request,
                "error": "Gagal membaca gambar"
            })

        if method == "lbp":
            # Implementasi LBP
            def calculate_lbp(image, points=8, radius=1):
                rows = image.shape[0]
                cols = image.shape[1]
                output = np.zeros((rows-2*radius, cols-2*radius), dtype=np.uint8)
                
                for i in range(radius, rows-radius):
                    for j in range(radius, cols-radius):
                        center = image[i,j]
                        code = 0
                        for k in range(points):
                            x = i + int(radius * np.cos(2*np.pi*k/points))
                            y = j - int(radius * np.sin(2*np.pi*k/points))
                            if image[x,y] >= center:
                                code |= (1 << k)
                        output[i-radius,j-radius] = code
                
                return output
            
            lbp = calculate_lbp(img)
            
            # Hitung histogram LBP
            hist = cv2.calcHist([lbp], [0], None, [256], [0, 256])
            plt.figure()
            plt.plot(hist)
            plt.title('LBP Histogram')
            hist_path = f"static/results/lbp_hist_{uuid4()}.png"
            plt.savefig(hist_path)
            plt.close()
            
            structural_path = save_image(lbp, "lbp", is_processed=True)
            lbp_histogram = f"/{hist_path}"
            
        else:  # Texton
            # Implementasi sederhana dari Texton menggunakan filter bank
            filters = []
            ksize = 49
            sigma = 1
            for theta in [0, 45, 90, 135]:
                kernel = cv2.getGaborKernel((ksize, ksize), sigma, np.radians(theta), 10.0, 0.5, 0)
                filters.append(kernel)
            
            # Aplikasikan filter bank
            responses = []
            for kernel in filters:
                responses.append(cv2.filter2D(img, cv2.CV_8UC3, kernel))
            
            # Gabungkan respons
            texton_map = np.mean(responses, axis=0)
            texton_map = cv2.normalize(texton_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            structural_path = save_image(texton_map, "texton", is_processed=True)
            lbp_histogram = None
        
        original_path = save_image(img, "original")
        
        return templates.TemplateResponse("texture_analysis.html", {
            "request": request,
            "original_image_path": original_path,
            "structural_path": structural_path,
            "lbp_histogram": lbp_histogram,
            "method": method,
            "mode": "structural"
        })
    except Exception as e:
        return templates.TemplateResponse("texture_analysis.html", {
            "request": request,
            "error": f"Error: {str(e)}"
        })

@app.get("/compilation/", response_class=HTMLResponse)
async def compilation_page(request: Request):
    return templates.TemplateResponse("compilation.html", {
        "request": request
    })

@app.post("/compilation/single/", response_class=HTMLResponse)
async def process_single_compilation(
    request: Request,
    file: UploadFile = File(...),
    operations: str = Form(None)
):
    try:
        # Parse operations string to list
        operation_list = operations.split(',') if operations else []
        
        if not operation_list:
            return templates.TemplateResponse("compilation.html", {
                "request": request,
                "error": "Pilih minimal satu operasi"
            })

        # Baca gambar
        image_data = await file.read()
        np_array = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        
        if img is None:
            return templates.TemplateResponse("compilation.html", {
                "request": request,
                "error": "Gagal membaca gambar"
            })

        # Get image information
        height, width = img.shape[:2]
        channels = img.shape[2] if len(img.shape) > 2 else 1
        file_size = len(image_data)
        
        # Determine image format from file content
        format = file.content_type.split('/')[-1].upper() if file.content_type else 'Unknown'
        
        # Determine color mode
        mode = 'RGB' if channels == 3 else 'RGBA' if channels == 4 else 'Grayscale'

        # Simpan gambar asli
        original_path = save_image(img, "original")
        
        # Calculate basic image statistics
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if channels > 1 else img
        min_val = int(np.min(gray))
        max_val = int(np.max(gray))
        mean_val = float(np.mean(gray))
        std_val = float(np.std(gray))
        
        results = {
            "original_image": original_path,
            "image_info": {
                "width": width,
                "height": height,
                "format": format,
                "mode": mode,
                "channels": channels
            },
            "image_stats": {
                "min_value": min_val,
                "max_value": max_val,
                "mean": mean_val,
                "std": std_val
            },
            "operations": []
        }

        # Proses setiap operasi yang dipilih
        for operation in operation_list:
            # Parse operation and parameter if exists
            if ':' in operation:
                op_name, op_value = operation.split(':')
            else:
                op_name = operation
                op_value = None

            if op_name == "grayscale":
                start_time = time.time()
                result = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                processing_time = time.time() - start_time
                path = save_image(result, "grayscale", is_processed=True)
                
                results["operations"].append({
                    "name": "Konversi Grayscale",
                    "type": "image",
                    "data": path
                })

            elif op_name == "histogram":
                start_time = time.time()
                if len(img.shape) == 3:  # Color image
                    hist_path = save_color_histogram(img)
                else:  # Grayscale image
                    hist_path = save_histogram(img, "histogram")
                processing_time = time.time() - start_time
                
                results["operations"].append({
                    "name": "Analisis Histogram",
                    "type": "plot",
                    "data": hist_path
                })

            elif op_name == "add":
                start_time = time.time()
                value = float(op_value)
                img_u8 = img.astype(np.uint8)
                arr = np.full(img.shape, value, dtype=np.uint8)
                result = cv2.add(img_u8, arr)
                processing_time = time.time() - start_time
                path = save_image(result, "add", is_processed=True)
                
                results["operations"].append({
                    "name": f"Penambahan (Nilai: {value})",
                    "type": "image",
                    "data": path
                })

            elif op_name == "subtract":
                start_time = time.time()
                value = float(op_value)
                img_u8 = img.astype(np.uint8)
                arr = np.full(img.shape, value, dtype=np.uint8)
                result = cv2.subtract(img_u8, arr)
                processing_time = time.time() - start_time
                path = save_image(result, "subtract", is_processed=True)
                
                results["operations"].append({
                    "name": f"Pengurangan (Nilai: {value})",
                    "type": "image",
                    "data": path
                })

            elif op_name == "multiply":
                start_time = time.time()
                value = float(op_value)
                img32 = img.astype(np.float32)
                arr = np.full(img.shape, value, dtype=np.float32)
                result = cv2.multiply(img32, arr)
                result = np.clip(result, 0, 255).astype(np.uint8)
                processing_time = time.time() - start_time
                path = save_image(result, "multiply", is_processed=True)
                
                results["operations"].append({
                    "name": f"Perkalian (Nilai: {value})",
                    "type": "image",
                    "data": path
                })

            elif op_name == "divide":
                start_time = time.time()
                value = float(op_value)
                img32 = img.astype(np.float32)
                arr = np.full(img.shape, value, dtype=np.float32)
                result = cv2.divide(img32, arr)
                result = np.clip(result, 0, 255).astype(np.uint8)
                processing_time = time.time() - start_time
                path = save_image(result, "divide", is_processed=True)
                
                results["operations"].append({
                    "name": f"Pembagian (Nilai: {value})",
                    "type": "image",
                    "data": path
                })

            elif op_name == "not":
                start_time = time.time()
                result = cv2.bitwise_not(img)
                processing_time = time.time() - start_time
                path = save_image(result, "not", is_processed=True)
                
                results["operations"].append({
                    "name": "Operasi NOT",
                    "type": "image",
                    "data": path
                })

            elif op_name == "equalize":
                mode = op_value or "grayscale"
                start_time = time.time()
                if mode == "grayscale":
                    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    result = cv2.equalizeHist(gray_img)
                    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
                else:
                    b, g, r = cv2.split(img)
                    b_eq = cv2.equalizeHist(b)
                    g_eq = cv2.equalizeHist(g)
                    r_eq = cv2.equalizeHist(r)
                    result = cv2.merge((b_eq, g_eq, r_eq))
                processing_time = time.time() - start_time
                path = save_image(result, "equalize", is_processed=True)
                results["operations"].append({
                    "name": f"Equalisasi Histogram ({mode})",
                    "type": "image",
                    "data": path
                })

            elif op_name == "padding":
                pad = int(op_value) if op_value else 20
                start_time = time.time()
                result = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[0,0,0])
                processing_time = time.time() - start_time
                path = save_image(result, "padding", is_processed=True)
                results["operations"].append({
                    "name": f"Padding ({pad}px)",
                    "type": "image",
                    "data": path
                })

            elif op_name == "filter":
                ftype, fkernel = (op_value.split('|') if op_value else ("gaussian", "5"))
                k = int(fkernel)
                start_time = time.time()
                if ftype == "gaussian":
                    result = cv2.GaussianBlur(img, (k, k), 0)
                elif ftype == "median":
                    result = cv2.medianBlur(img, k)
                elif ftype == "bilateral":
                    result = cv2.bilateralFilter(img, k, 75, 75)
                else:
                    result = img
                processing_time = time.time() - start_time
                path = save_image(result, "filter", is_processed=True)
                results["operations"].append({
                    "name": f"Filter ({ftype}, kernel={k})",
                    "type": "image",
                    "data": path
                })

            elif op_name == "fourier":
                ftype = op_value or "magnitude"
                start_time = time.time()
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                f = np.fft.fft2(gray_img)
                fshift = np.fft.fftshift(f)
                magnitude_spectrum = None
                phase_spectrum = None
                if ftype in ["magnitude", "both"]:
                    magnitude_spectrum = 20 * np.log(np.abs(fshift))
                    magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                if ftype in ["phase", "both"]:
                    phase_spectrum = np.angle(fshift)
                    phase_spectrum = cv2.normalize(phase_spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                if ftype == "both":
                    result = np.hstack((magnitude_spectrum, phase_spectrum))
                elif ftype == "magnitude":
                    result = magnitude_spectrum
                else:
                    result = phase_spectrum
                processing_time = time.time() - start_time
                path = save_image(result, "fourier", is_processed=True)
                results["operations"].append({
                    "name": f"Fourier ({ftype})",
                    "type": "image",
                    "data": path
                })

            elif op_name == "sharpen":
                smethod = op_value or "laplacian"
                start_time = time.time()
                if smethod == "laplacian":
                    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                    result = cv2.filter2D(img, -1, kernel)
                else:
                    gaussian = cv2.GaussianBlur(img, (5, 5), 0)
                    result = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)
                processing_time = time.time() - start_time
                path = save_image(result, "sharpen", is_processed=True)
                results["operations"].append({
                    "name": f"Penajaman ({smethod})",
                    "type": "image",
                    "data": path
                })

            elif op_name == "denoise":
                dmethod = op_value or "gaussian"
                start_time = time.time()
                if dmethod == "gaussian":
                    result = cv2.GaussianBlur(img, (5, 5), 0)
                elif dmethod == "median":
                    result = cv2.medianBlur(img, 5)
                elif dmethod == "bilateral":
                    result = cv2.bilateralFilter(img, 9, 75, 75)
                else:
                    result = img
                processing_time = time.time() - start_time
                path = save_image(result, "denoise", is_processed=True)
                results["operations"].append({
                    "name": f"Penghilangan Noise ({dmethod})",
                    "type": "image",
                    "data": path
                })

            elif op_name == "add_noise":
                amount = float(op_value) if op_value else 0.05
                start_time = time.time()
                output = np.copy(img)
                num_salt = np.ceil(amount * img.size * 0.5)
                coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
                output[coords[0], coords[1]] = 255
                num_pepper = np.ceil(amount * img.size * 0.5)
                coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
                output[coords[0], coords[1]] = 0
                result = output
                processing_time = time.time() - start_time
                path = save_image(result, "add_noise", is_processed=True)
                results["operations"].append({
                    "name": f"Tambah Noise Salt & Pepper ({amount})",
                    "type": "image",
                    "data": path
                })

            elif op_name == "reduce_periodic":
                # Format: d0|n|uk|vk
                d0, n, uk, vk = (op_value.split('|') if op_value else ("10", "4", "50", "50"))
                d0 = int(d0)
                n = int(n)
                uk = int(uk)
                vk = int(vk)
                start_time = time.time()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                f = np.fft.fft2(gray)
                fshift = np.fft.fftshift(f)
                rows, cols = gray.shape
                crow, ccol = rows//2, cols//2
                u = np.arange(rows) - crow
                v = np.arange(cols) - ccol
                v, u = np.meshgrid(v, u)
                d = np.sqrt(u**2 + v**2)
                h = 1 / (1 + (d0**2 / (d**2 + 1e-10))**n)
                fshift_filtered = fshift * h
                f_ishift = np.fft.ifftshift(fshift_filtered)
                img_back = np.fft.ifft2(f_ishift)
                img_back = np.abs(img_back)
                img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                result = img_back
                processing_time = time.time() - start_time
                path = save_image(result, "reduce_periodic", is_processed=True)
                results["operations"].append({
                    "name": f"Pengurangan Noise Periodik (D0={d0}, n={n}, uk={uk}, vk={vk})",
                    "type": "image",
                    "data": path
                })

            elif op_name == "freeman":
                thresh, invert = (op_value.split('|') if op_value else ("127", "False"))
                threshold = int(thresh)
                invert = (invert == 'true')
                
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                mode_thresh = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
                _, binary = cv2.threshold(gray, threshold, 255, mode_thresh)
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                
                result = img
                if contours:
                    largest = max(contours, key=cv2.contourArea)
                    result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                    cv2.drawContours(result, [largest], -1, (0, 255, 0), 2)

                path = save_image(result, "freeman", is_processed=True)
                results["operations"].append({
                    "name": f"Freeman Chain Code (Thresh: {threshold}, Invert: {invert})",
                    "type": "image",
                    "data": path
                })

            elif op_name == "crack_code":
                low, high, kernel = (op_value.split('|') if op_value else ("50", "150", "5"))
                low_threshold = int(low)
                high_threshold = int(high)
                kernel_size = int(kernel)
                
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
                result = cv2.Canny(blurred, low_threshold, high_threshold)
                
                path = save_image(result, "crack_code", is_processed=True)
                results["operations"].append({
                    "name": f"Deteksi Tepi (Low: {low}, High: {high}, Kernel: {kernel})",
                    "type": "image",
                    "data": path
                })

            elif op_name == "projection":
                thresh, invert = (op_value.split('|') if op_value else ("127", "False"))
                threshold = int(thresh)
                invert = (invert == 'true')
                
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                mode_thresh = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
                _, binary = cv2.threshold(gray, threshold, 255, mode_thresh)
                binary_norm = binary / 255.0
                
                horiz = np.sum(binary_norm, axis=1)
                vert = np.sum(binary_norm, axis=0)
                
                # Simpan Proyeksi Horizontal
                plt.figure()
                plt.plot(horiz)
                plt.title("Proyeksi Horizontal")
                horiz_filename = f"horiz_{uuid4()}.png"
                horiz_save_path = os.path.join("static", "results", horiz_filename)
                horiz_url_path = f"/static/results/{horiz_filename}"
                plt.savefig(horiz_save_path)
                plt.close()

                # Simpan Proyeksi Vertikal
                plt.figure()
                plt.plot(vert)
                plt.title("Proyeksi Vertikal")
                vert_filename = f"vert_{uuid4()}.png"
                vert_save_path = os.path.join("static", "results", vert_filename)
                vert_url_path = f"/static/results/{vert_filename}"
                plt.savefig(vert_save_path)
                plt.close()
                
                results["operations"].append({
                    "name": "Proyeksi Horizontal",
                    "type": "plot",
                    "data": horiz_url_path
                })
                results["operations"].append({
                    "name": "Proyeksi Vertikal",
                    "type": "plot",
                    "data": vert_url_path
                })

            elif op_name == "compression":
                method, value = (op_value.split('|'))
                
                # Simpan gambar original sementara untuk perbandingan ukuran
                temp_original_path = f"static/uploads/temp_{uuid4()}.png"
                cv2.imwrite(temp_original_path, img)
                original_size = os.path.getsize(temp_original_path)

                if method == "jpeg":
                    quality = int(value)
                    jpeg_path = f"static/results/compressed_{uuid4()}.jpg"
                    cv2.imwrite(jpeg_path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
                    compressed_size = os.path.getsize(jpeg_path)
                    compressed_img = cv2.imread(jpeg_path)
                    
                    psnr = cv2.PSNR(img, compressed_img)
                    mse = np.mean((img.astype(float) - compressed_img.astype(float)) ** 2)
                    ratio = original_size / compressed_size if compressed_size > 0 else float('inf')
                    
                    results["operations"].append({
                        "name": f"Kompresi JPEG (Kualitas: {quality})",
                        "type": "compression_analysis",
                        "data": {
                            "compressed_image_path": f"/{jpeg_path}",
                            "original_size": f"{original_size / 1024:.2f} KB",
                            "compressed_size": f"{compressed_size / 1024:.2f} KB",
                            "compression_ratio": f"{ratio:.2f}",
                            "psnr": f"{psnr:.2f}",
                            "mse": f"{mse:.2f}",
                            "method": "JPEG"
                        }
                    })
                else: # PNG
                    level = int(value)
                    png_path = f"static/results/compressed_{uuid4()}.png"
                    cv2.imwrite(png_path, img, [cv2.IMWRITE_PNG_COMPRESSION, level])
                    compressed_size = os.path.getsize(png_path)
                    compressed_img = cv2.imread(png_path)

                    is_identical = np.array_equal(img, compressed_img)
                    ratio = original_size / compressed_size if compressed_size > 0 else float('inf')

                    results["operations"].append({
                        "name": f"Kompresi PNG (Level: {level})",
                        "type": "compression_analysis",
                        "data": {
                            "compressed_image_path": f"/{png_path}",
                            "original_size": f"{original_size / 1024:.2f} KB",
                            "compressed_size": f"{compressed_size / 1024:.2f} KB",
                            "compression_ratio": f"{ratio:.2f}",
                            "is_identical": is_identical,
                            "method": "PNG"
                        }
                    })
                
                os.remove(temp_original_path) # Hapus file sementara
                
            elif op_name == "convert_space":
                target_space = op_value
                conversion_map = {
                    "xyz": cv2.COLOR_BGR2XYZ, "lab": cv2.COLOR_BGR2LAB,
                    "ycrcb": cv2.COLOR_BGR2YCrCb, "hsv": cv2.COLOR_BGR2HSV,
                    "hsl": cv2.COLOR_BGR2HLS, "luv": cv2.COLOR_BGR2LUV,
                    "yuv": cv2.COLOR_BGR2YUV
                }
                if target_space in conversion_map:
                    result = cv2.cvtColor(img, conversion_map[target_space])
                    path = save_image(result, f"convert_{target_space}", is_processed=True)
                    results["operations"].append({
                        "name": f"Konversi ke {target_space.upper()}",
                        "type": "image",
                        "data": path
                    })

            elif op_name == "analyze_channels":
                space = op_value
                channel_map = {
                    "rgb": (cv2.COLOR_BGR2RGB, ["Red", "Green", "Blue"]),
                    "lab": (cv2.COLOR_BGR2LAB, ["L*", "a*", "b*"]),
                    "hsv": (cv2.COLOR_BGR2HSV, ["Hue", "Saturation", "Value"]),
                    "ycrcb": (cv2.COLOR_BGR2YCrCb, ["Y", "Cr", "Cb"])
                }
                if space in channel_map:
                    conversion, names = channel_map[space]
                    converted_img = cv2.cvtColor(img, conversion)
                    channels = cv2.split(converted_img)
                    for i, channel in enumerate(channels):
                        path = save_image(channel, f"channel_{space}_{i}", is_processed=True)
                        results["operations"].append({
                            "name": f"Kanal {names[i]} ({space.upper()})",
                            "type": "image",
                            "data": path
                        })

            elif op_name == "analyze_luminance":
                ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
                luminance = ycrcb[:,:,0]
                
                # Calculate luminance statistics
                mean_lum = np.mean(luminance)
                std_lum = np.std(luminance)
                min_lum = np.min(luminance)
                max_lum = np.max(luminance)

                path = save_image(luminance, "luminance", is_processed=True)
                
                results["operations"].append({
                    "name": "Analisis Luminance (Kanal Y)",
                    "type": "luminance_analysis",
                    "data": {
                        "image_path": path,
                        "analysis": {
                            "mean_luminance": f"{mean_lum:.2f}",
                            "std_luminance": f"{std_lum:.2f}",
                            "min_luminance": f"{min_lum:.2f}",
                            "max_luminance": f"{max_lum:.2f}"
                        }
                    }
                })

            elif op_name == "texture_first_order":
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                mean = np.mean(gray)
                variance = np.var(gray)
                skewness = np.mean((gray - mean) ** 3) / (np.std(gray) ** 3)
                kurtosis = np.mean((gray - mean) ** 4) / (np.std(gray) ** 4)
                hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
                hist_norm = hist / hist.sum()
                energy = np.sum(hist_norm ** 2)
                entropy = -np.sum(hist_norm * np.log2(hist_norm + np.finfo(float).eps))
                
                results["operations"].append({
                    "name": "Tekstur: Statistik First-Order",
                    "type": "texture_analysis",
                    "data": {
                        "image_path": None,
                        "analysis": {
                            "Mean": f"{mean:.2f}", "Variance": f"{variance:.2f}",
                            "Skewness": f"{skewness:.2f}", "Kurtosis": f"{kurtosis:.2f}",
                            "Energy": f"{energy:.4f}", "Entropy": f"{entropy:.2f}"
                        }
                    }
                })

            elif op_name == "texture_glcm":
                direction = int(op_value)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                distances = [1]
                angles = [np.pi/4 * direction]
                glcm = graycomatrix(gray, distances, angles, 256, symmetric=True, normed=True)
                
                contrast = graycoprops(glcm, 'contrast')[0, 0]
                correlation = graycoprops(glcm, 'correlation')[0, 0]
                energy = graycoprops(glcm, 'energy')[0, 0]
                homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
                
                plt.figure()
                plt.imshow(glcm[:,:,0,0], cmap='gray')
                plt.colorbar()
                plt.title(f'GLCM (Arah: {direction*45}°)')
                glcm_filename = f"glcm_{uuid4()}.png"
                glcm_save_path = os.path.join("static", "results", glcm_filename)
                glcm_url_path = f"/static/results/{glcm_filename}"
                plt.savefig(glcm_save_path)
                plt.close()

                results["operations"].append({
                    "name": f"Tekstur: GLCM ({direction*45}°)",
                    "type": "texture_analysis",
                    "data": {
                        "image_path": glcm_url_path,
                        "analysis": {
                            "Contrast": f"{contrast:.2f}", "Correlation": f"{correlation:.2f}",
                            "Energy": f"{energy:.4f}", "Homogeneity": f"{homogeneity:.2f}"
                        }
                    }
                })
            
            elif op_name == "texture_gabor":
                orientation, frequency = map(float, op_value.split('|'))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                ksize, sigma, theta, lambd, gamma, psi = 31, 5, np.radians(orientation), 1/frequency, 0.5, 0
                kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi)
                result = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
                path = save_image(result, "gabor_filtered", is_processed=True)
                results["operations"].append({
                    "name": f"Tekstur: Filter Gabor (θ={orientation}°, f={frequency})",
                    "type": "image",
                    "data": path
                })

            elif op_name == "texture_lbp":
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                points, radius = 8, 1
                rows, cols = gray.shape
                lbp_img = np.zeros((rows-2*radius, cols-2*radius), dtype=np.uint8)
                for i in range(radius, rows-radius):
                    for j in range(radius, cols-radius):
                        center = gray[i,j]
                        code = 0
                        for k in range(points):
                            x = i + int(radius * np.cos(2*np.pi*k/points))
                            y = j - int(radius * np.sin(2*np.pi*k/points))
                            if gray[x,y] >= center:
                                code |= (1 << k)
                        lbp_img[i-radius,j-radius] = code
                path = save_image(lbp_img, "lbp", is_processed=True)
                results["operations"].append({
                    "name": "Tekstur: Local Binary Patterns (LBP)",
                    "type": "image",
                    "data": path
                })

            elif op_name == "texture_glrlm":
                direction = int(op_value)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                def calculate_glrlm_matrix(img_gray, angle_deg):
                    rows, cols = img_gray.shape
                    glrlm = np.zeros((256, max(rows, cols)), dtype=np.int32)
                    if angle_deg == 0: # Horizontal
                        for r in range(rows):
                            current_run_val, current_run_len = img_gray[r, 0], 1
                            for c in range(1, cols):
                                if img_gray[r, c] == current_run_val:
                                    current_run_len += 1
                                else:
                                    glrlm[current_run_val, current_run_len-1] += 1
                                    current_run_val, current_run_len = img_gray[r, c], 1
                            glrlm[current_run_val, current_run_len-1] += 1
                    else: # Vertical
                        for c in range(cols):
                            current_run_val, current_run_len = img_gray[0, c], 1
                            for r in range(1, rows):
                                if img_gray[r, c] == current_run_val:
                                    current_run_len += 1
                                else:
                                    glrlm[current_run_val, current_run_len-1] += 1
                                    current_run_val, current_run_len = img_gray[r, c], 1
                            glrlm[current_run_val, current_run_len-1] += 1
                    return glrlm

                glrlm_matrix = calculate_glrlm_matrix(gray, direction)
                total_runs = np.sum(glrlm_matrix)
                if total_runs > 0:
                    run_lengths = np.arange(1, glrlm_matrix.shape[1] + 1)
                    sre = np.sum(np.sum(glrlm_matrix, axis=0) / (run_lengths ** 2)) / total_runs
                    lre = np.sum(np.sum(glrlm_matrix, axis=0) * (run_lengths ** 2)) / total_runs
                    gln = np.sum(np.sum(glrlm_matrix, axis=1) ** 2) / total_runs
                    rln = np.sum(np.sum(glrlm_matrix, axis=0) ** 2) / total_runs
                else:
                    sre, lre, gln, rln = 0,0,0,0

                results["operations"].append({
                    "name": f"Tekstur: GLRLM ({direction}°)",
                    "type": "texture_analysis",
                    "data": {
                        "image_path": None,
                        "analysis": {
                            "Short Run Emphasis (SRE)": f"{sre:.2f}",
                            "Long Run Emphasis (LRE)": f"{lre:.2f}",
                            "Gray Level Non-uniformity (GLN)": f"{gln:.2f}",
                            "Run Length Non-uniformity (RLN)": f"{rln:.2f}"
                        }
                    }
                })

            elif op_name == "texture_texton":
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                filters, responses = [], []
                ksize, sigma = 31, 5
                for theta in [0, 45, 90, 135]:
                    kernel = cv2.getGaborKernel((ksize, ksize), sigma, np.radians(theta), 10.0, 0.5, 0)
                    filters.append(kernel)
                for kernel in filters:
                    responses.append(cv2.filter2D(gray, cv2.CV_8UC3, kernel))
                texton_map = np.mean(responses, axis=0)
                result = cv2.normalize(texton_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                path = save_image(result, "texton", is_processed=True)
                results["operations"].append({
                    "name": "Tekstur: Analisis Texton",
                    "type": "image",
                    "data": path
                })

        return templates.TemplateResponse("compilation.html", {
            "request": request,
            "results": results
        })
        
    except Exception as e:
        return templates.TemplateResponse("compilation.html", {
            "request": request,
            "error": f"Terjadi kesalahan: {str(e)}"
        })

@app.post("/compilation/double/", response_class=HTMLResponse)
async def process_double_compilation(
    request: Request,
    file1: UploadFile = File(...),
    file2: UploadFile = File(...),
    operations: str = Form(None)
):
    try:
        # Parse operations string to list
        operation_list = operations.split(',') if operations else []
        
        if not operation_list:
            return templates.TemplateResponse("compilation.html", {
                "request": request,
                "error": "Pilih minimal satu operasi"
            })

        # Baca gambar pertama
        image_data1 = await file1.read()
        np_array1 = np.frombuffer(image_data1, np.uint8)
        img1 = cv2.imdecode(np_array1, cv2.IMREAD_COLOR)
        
        # Baca gambar kedua
        image_data2 = await file2.read()
        np_array2 = np.frombuffer(image_data2, np.uint8)
        img2 = cv2.imdecode(np_array2, cv2.IMREAD_COLOR)
        
        if img1 is None or img2 is None:
            return templates.TemplateResponse("compilation.html", {
                "request": request,
                "error": "Gagal membaca gambar"
            })

        # Periksa dimensi gambar
        if img1.shape != img2.shape:
            return templates.TemplateResponse("compilation.html", {
                "request": request,
                "error": "Kedua gambar harus memiliki dimensi yang sama"
            })

        # Get image information for both images
        height1, width1 = img1.shape[:2]
        channels1 = img1.shape[2] if len(img1.shape) > 2 else 1
        format1 = file1.content_type.split('/')[-1].upper() if file1.content_type else 'Unknown'
        mode1 = 'RGB' if channels1 == 3 else 'RGBA' if channels1 == 4 else 'Grayscale'

        height2, width2 = img2.shape[:2]
        channels2 = img2.shape[2] if len(img2.shape) > 2 else 1
        format2 = file2.content_type.split('/')[-1].upper() if file2.content_type else 'Unknown'
        mode2 = 'RGB' if channels2 == 3 else 'RGBA' if channels2 == 4 else 'Grayscale'

        # Calculate basic image statistics for both images
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if channels1 > 1 else img1
        min_val1 = int(np.min(gray1))
        max_val1 = int(np.max(gray1))
        mean_val1 = float(np.mean(gray1))
        std_val1 = float(np.std(gray1))

        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if channels2 > 1 else img2
        min_val2 = int(np.min(gray2))
        max_val2 = int(np.max(gray2))
        mean_val2 = float(np.mean(gray2))
        std_val2 = float(np.std(gray2))

        # Simpan kedua gambar asli
        original_path1 = save_image(img1, "original1")
        original_path2 = save_image(img2, "original2")
        
        results = {
            "original_image": original_path1,
            "second_image": original_path2,
            "image_info": {
                "width": width1,
                "height": height1,
                "format": format1,
                "mode": mode1,
                "channels": channels1
            },
            "image_stats": {
                "min_value": min_val1,
                "max_value": max_val1,
                "mean": mean_val1,
                "std": std_val1
            },
            "second_image_info": {
                "width": width2,
                "height": height2,
                "format": format2,
                "mode": mode2,
                "channels": channels2
            },
            "second_image_stats": {
                "min_value": min_val2,
                "max_value": max_val2,
                "mean": mean_val2,
                "std": std_val2
            },
            "operations": []
        }

        # Proses setiap operasi yang dipilih
        for operation in operation_list:
            if operation == "logic_and":
                start_time = time.time()
                result = cv2.bitwise_and(img1, img2)
                processing_time = time.time() - start_time
                path = save_image(result, "logic_and", is_processed=True)
                
                results["operations"].append({
                    "name": "Operasi AND",
                    "type": "image",
                    "data": path
                })

            elif operation == "logic_or":
                start_time = time.time()
                result = cv2.bitwise_or(img1, img2)
                processing_time = time.time() - start_time
                path = save_image(result, "logic_or", is_processed=True)
                
                results["operations"].append({
                    "name": "Operasi OR",
                    "type": "image",
                    "data": path
                })

            elif operation == "logic_xor":
                start_time = time.time()
                result = cv2.bitwise_xor(img1, img2)
                processing_time = time.time() - start_time
                path = save_image(result, "logic_xor", is_processed=True)
                
                results["operations"].append({
                    "name": "Operasi XOR",
                    "type": "image",
                    "data": path
                })

            elif operation == "hist_spec_gray":
                start_time = time.time()
                # Convert both images to grayscale
                gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                
                # Calculate histograms
                hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
                hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
                
                # Calculate cumulative distribution functions (CDF)
                cdf1 = hist1.cumsum()
                cdf1_normalized = cdf1 / cdf1.max()
                
                cdf2 = hist2.cumsum()
                cdf2_normalized = cdf2 / cdf2.max()
                
                # Create lookup table
                lut = np.zeros(256)
                for i in range(256):
                    j = 0
                    while j < 256 and cdf2_normalized[j] < cdf1_normalized[i]:
                        j += 1
                    lut[i] = j
                
                # Apply specification
                result = cv2.LUT(gray1, lut.astype(np.uint8))
                
                # Save results
                processing_time = time.time() - start_time
                result_path = save_image(result, "hist_spec_gray", is_processed=True)
                
                # Generate histograms for visualization
                hist_before = save_histogram(gray1, "before_spec")
                hist_after = save_histogram(result, "after_spec")
                hist_ref = save_histogram(gray2, "reference")
                
                results["operations"].extend([
                    {
                        "name": "Spesifikasi Histogram (Grayscale)",
                        "type": "image",
                        "data": result_path,
                    "analysis": {
                        "metrics": {
                                "Processing Time": f"{processing_time:.3f} sec"
                            },
                            "description": "Hasil spesifikasi histogram menggunakan gambar kedua sebagai referensi."
                        }
                    },
                    {
                        "name": "Histogram Sebelum Spesifikasi",
                        "type": "plot",
                        "data": hist_before
                    },
                    {
                        "name": "Histogram Setelah Spesifikasi",
                        "type": "plot",
                        "data": hist_after
                    },
                    {
                        "name": "Histogram Referensi",
                        "type": "plot",
                        "data": hist_ref
                    }
                ])

            elif operation == "hist_spec_color":
                start_time = time.time()
                # Split channels
                b1, g1, r1 = cv2.split(img1)
                b2, g2, r2 = cv2.split(img2)
                
                # Process each channel separately
                channels = []
                hist_before = []
                hist_after = []
                hist_ref = []
                
                for src, ref, color in [(b1, b2, 'blue'), (g1, g2, 'green'), (r1, r2, 'red')]:
                    # Calculate histograms
                    hist1 = cv2.calcHist([src], [0], None, [256], [0, 256])
                    hist2 = cv2.calcHist([ref], [0], None, [256], [0, 256])
                    
                    # Calculate CDFs
                    cdf1 = hist1.cumsum()
                    cdf1_normalized = cdf1 / cdf1.max()
                    
                    cdf2 = hist2.cumsum()
                    cdf2_normalized = cdf2 / cdf2.max()
                    
                    # Create lookup table
                    lut = np.zeros(256)
                    for i in range(256):
                        j = 0
                        while j < 256 and cdf2_normalized[j] < cdf1_normalized[i]:
                            j += 1
                        lut[i] = j
                    
                    # Apply specification
                    result_channel = cv2.LUT(src, lut.astype(np.uint8))
                    channels.append(result_channel)
                    
                    # Save histograms
                    hist_before.append(save_histogram(src, f"before_spec_{color}"))
                    hist_after.append(save_histogram(result_channel, f"after_spec_{color}"))
                    hist_ref.append(save_histogram(ref, f"reference_{color}"))
                
                # Merge channels
                result = cv2.merge(channels)
                processing_time = time.time() - start_time
                
                # Save results
                result_path = save_image(result, "hist_spec_color", is_processed=True)
                
                # Add results
                results["operations"].append({
                    "name": "Spesifikasi Histogram (Warna)",
                    "type": "image",
                    "data": result_path,
                    "analysis": {
                        "metrics": {
                            "Processing Time": f"{processing_time:.3f} sec"
                        },
                        "description": "Hasil spesifikasi histogram untuk setiap channel warna menggunakan gambar kedua sebagai referensi."
                    }
                })
                
                # Add histograms for each channel
                for i, color in enumerate(['Blue', 'Green', 'Red']):
                    results["operations"].extend([
                        {
                            "name": f"Histogram {color} Sebelum",
                            "type": "plot",
                            "data": hist_before[i]
                        },
                        {
                            "name": f"Histogram {color} Setelah",
                            "type": "plot",
                            "data": hist_after[i]
                        },
                        {
                            "name": f"Histogram {color} Referensi",
                            "type": "plot",
                            "data": hist_ref[i]
                        }
                    ])

        return templates.TemplateResponse("compilation.html", {
            "request": request,
            "results": results
        })
        
    except Exception as e:
        return templates.TemplateResponse("compilation.html", {
            "request": request,
            "error": f"Terjadi kesalahan: {str(e)}"
        })

