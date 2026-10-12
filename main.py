from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import cv2
import numpy as np
import base64
from PIL import Image, ImageDraw, ImageFont
import io
import torch
import gc # Сборщик мусора для контроля памяти

torch.set_num_threads(1)

app = FastAPI(title="Vehicle Detection API")

# Глобальные переменные для хранения только одной активной модели
active_model = None
active_model_name = None

CLASS_NAMES = {
    2: "Легковое авто", 3: "Легковое авто", 4: "Фургон",         
    5: "Грузовик", 6: "Трейлер", 7: "Грузовик", 9: "Автобус"         
}

CLASS_MAPPING = {
    "Все": [2, 3, 4, 5, 6, 7, 9],
    "Легковые авто": [2, 3],
    "Фургоны": [4],
    "Грузовики и трейлеры": [5, 6, 7],
    "Автобусы": [5, 9]
}

COLORS = {
    "Легковое авто": (46, 204, 113), "Фургон": (241, 196, 15),         
    "Грузовик": (230, 126, 34), "Трейлер": (211, 84, 0),          
    "Автобус": (231, 76, 60), "Неизвестно": (149, 165, 166)     
}

@app.get("/")
def health_check():
    return {"status": "Сервер работает и ждет картинок!"}

@app.post("/predict")
async def predict(
    model_type: str = Form("fast"), 
    filter_class: str = Form("Все"),
    conf_threshold: float = Form(0.25),
    show_labels: bool = Form(True),
    file: UploadFile = File(...)
):
    global active_model, active_model_name
    
    print(f"ЛОГ: Запрос получен. Требуется модель: {model_type}")
    target_weights = "nano_ready.pt" if model_type == "fast" else "medium_ready.pt"
    
    # ЛЕНИВАЯ ЗАГРУЗКА И ОЧИСТКА ПАМЯТИ
    if active_model_name != target_weights:
        print(f"ЛОГ: Освобождаем память. Загружаем {target_weights}...")
        active_model = None
        gc.collect() # Принудительно чистим RAM
        active_model = YOLO(target_weights)
        active_model_name = target_weights

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    target_classes = CLASS_MAPPING.get(filter_class, CLASS_MAPPING["Все"])

    print("ЛОГ: Запускаем инференс YOLO...")
    results = active_model(image, classes=target_classes, conf=conf_threshold)
    
    print("ЛОГ: Рисуем боксы...")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_image)
    
    try:
        # <-- УМЕНЬШИЛИ ШРИФТ: было 0.02, стало 0.012 (и минимальный размер 12 вместо 16)
        font_size = max(12, int(pil_image.width * 0.012))
        font = ImageFont.truetype("Roboto-Regular.ttf", font_size)
    except:
        font = ImageFont.load_default()

    counts = {}

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        
        class_name = CLASS_NAMES.get(cls_id, "Неизвестно")
        counts[class_name] = counts.get(class_name, 0) + 1
        
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        color = COLORS.get(class_name, (255, 255, 255))
        
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        if show_labels:
            text = f"{class_name} ({conf:.2f})"
            text_bbox = draw.textbbox((x1, y1), text, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
            
            draw.rectangle([x1, y1, x1 + text_w + 8, y1 + text_h + 8], fill=color)
            draw.text((x1 + 4, y1 + 2), text, fill=(0, 0, 0), font=font)

    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    print("ЛОГ: Успех! Отправляем ответ.")
    return JSONResponse(content={"image_base64": img_str, "counts": counts})