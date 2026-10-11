import streamlit as st
import requests
from PIL import Image
import io
import base64
import os

st.set_page_config(
    page_title="Детекция автотранспорта AI",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Обязательно проверь, что тут твоя ссылка на Render!
API_URL = "https://vehicle-detection-api.onrender.com/predict"

# --- САЙДБАР ---
with st.sidebar:
    st.title("⚙️ Настройки")
    
    model_type = st.selectbox(
        "Архитектура модели:",
        ("Быстрая (YOLOv8 Nano)", "Точная (YOLOv8 Medium)")
    )
    
    st.divider()
    st.markdown("### Параметры поиска")
    
    filter_class = st.selectbox(
        "Что ищем на фото?",
        ("Все", "Легковые авто", "Фургоны", "Грузовики и трейлеры", "Автобусы")
    )
    
    conf_threshold = st.slider(
        "Порог уверенности (Confidence):", 
        min_value=0.10, max_value=0.95, value=0.30, step=0.05,
        help="Чем выше порог, тем меньше ложных срабатываний, но модель может пропустить неочевидные объекты."
    )
    
    st.divider()
    st.markdown("### Отображение")
    show_legend = st.checkbox("Показывать статистику", value=True)
    
    st.divider()
    st.markdown("### Исходные данные")
    uploaded_file = st.file_uploader("Загрузите фото (JPG/PNG)", type=["jpg", "jpeg", "png"])

# --- ЛОГИКА ДЕФОЛТНОЙ КАРТИНКИ ---
image_source = None
file_name = ""
file_bytes = None
file_mime = ""

if uploaded_file is not None:
    # Если пользователь сам загрузил фото
    image_source = uploaded_file
    file_name = uploaded_file.name
    file_bytes = uploaded_file.getvalue()
    file_mime = uploaded_file.type
elif os.path.exists("sample.jpg"):
    # Если фото нет, но есть дефолтная картинка в проекте
    image_source = "sample.jpg"
    file_name = "sample.jpg"
    with open("sample.jpg", "rb") as f:
        file_bytes = f.read()
    file_mime = "image/jpeg"

# --- ВКЛАДКИ (TABS) ---
tab_main, tab_about = st.tabs(["🔍 Детекция объектов", "ℹ️ О проекте (Стек)"])

with tab_main:
    st.title("CV-система детекции автотранспорта")
    model_param = "fast" if "Nano" in model_type else "accurate"

    # Если у нас есть хоть какая-то картинка (загруженная или дефолтная)
    if image_source is not None:
        analyze_button = st.button("🚀 Запустить анализ изображения", type="primary", use_container_width=True)
        st.divider()

        if analyze_button:
            with st.spinner("Инференс модели и отрисовка..."):
                try:
                    data = {
                        "model_type": model_param, 
                        "filter_class": filter_class,
                        "conf_threshold": conf_threshold
                    }
                    # Отправляем подготовленные байты
                    files = {"file": (file_name, file_bytes, file_mime)}
                    
                    response = requests.post(API_URL, data=data, files=files)
                    
                    if response.status_code == 200:
                        result = response.json()
                        image_data = base64.b64decode(result["image_base64"])
                        result_image = Image.open(io.BytesIO(image_data))
                        counts = result.get("counts", {})
                        
                        if show_legend:
                            st.subheader("📊 Статистика обнаружения")
                            if counts:
                                total_objects = sum(counts.values())
                                st.success(f"Успешно! Всего найдено объектов: **{total_objects}**")
                                
                                metric_cols = st.columns(len(counts))
                                for idx, (cls_name, count) in enumerate(counts.items()):
                                    metric_cols[idx].metric(label=cls_name, value=count)
                            else:
                                st.warning("Ни одного объекта не найдено. Попробуйте снизить порог уверенности.")
                            
                            st.markdown("<br>", unsafe_allow_html=True) 
                        
                        st.subheader("📷 Визуализация")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(image_source, caption="Исходное изображение", use_container_width=True)
                        with col2:
                            st.image(result_image, caption="Результат детекции", use_container_width=True)
                            
                    else:
                        st.error(f"Ошибка сервера: {response.status_code}")
                        
                except Exception as e:
                    st.error(f"Нет связи с Backend-сервером. Убедитесь, что сервер на Render активен. ({e})")
        else:
            # ДО НАЖАТИЯ КНОПКИ
            if uploaded_file is None:
                st.info("💡 Используется тестовое изображение по умолчанию. Вы можете загрузить своё фото в меню слева.")
            else:
                st.info("Нажмите кнопку «Запустить анализ изображения» выше, чтобы получить результат.")
                
            col_spacer1, col_center, col_spacer2 = st.columns([1, 2, 1])
            with col_center:
                st.image(image_source, caption="Предварительный просмотр", use_container_width=True)
    else:
        st.warning("⚠️ Для начала работы загрузите изображение в меню слева или поместите файл sample.jpg в папку с проектом.")

with tab_about:
    st.markdown("### Техническое описание проекта")
    st.markdown("""
    **Архитектура решения:**
    * **ML Framework:** PyTorch, Ultralytics YOLOv8
    * **Backend:** REST API на базе FastAPI (отвечает за инференс модели и отрисовку боксов).
    * **Frontend:** Streamlit.
    * **Данные:** Датасет VisDrone (изображения оптимизированы под детекцию малых объектов).
    
    **Особенности реализации:**
    1. **Выбор модели:** Доступен выбор между легковесной моделью (`Nano`) для быстрого инференса и тяжелой (`Medium`) для максимальной точности.
    2. **Динамическая фильтрация:** Реализована возможность выбора конкретных классов объектов для поиска (например, только грузовики).
    3. **Тюнинг уверенности:** Интерактивный ползунок позволяет изменять порог срабатывания (Confidence Threshold) без перезагрузки сервера.
    4. **Сбор статистики:** Backend агрегирует количество найденных объектов каждого класса и возвращает их в виде структурированного JSON.
    """)