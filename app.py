import streamlit as st
import requests
from PIL import Image
import io
import base64

st.set_page_config(
    page_title="Детекция автотранспорта AI",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# --- ВКЛАДКИ (TABS) ---
tab_main, tab_about = st.tabs(["🔍 Детекция объектов", "ℹ️ О проекте (Стек)"])

with tab_main:
    st.title("CV-система детекции автотранспорта")
    model_param = "fast" if "Nano" in model_type else "accurate"

    if uploaded_file is not None:
        # КНОПКА НА САМОМ ВЕРХУ
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
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    
                    response = requests.post(API_URL, data=data, files=files)
                    
                    if response.status_code == 200:
                        result = response.json()
                        image_data = base64.b64decode(result["image_base64"])
                        result_image = Image.open(io.BytesIO(image_data))
                        counts = result.get("counts", {})
                        
                        # 1. ВЫВОД СТАТИСТИКИ (Самое главное - сверху)
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
                            
                            st.markdown("<br>", unsafe_allow_html=True) # Небольшой отступ
                        
                        # 2. ВЫВОД КАРТИНОК (Под статистикой, для сравнения До/После)
                        st.subheader("Визуализация")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(uploaded_file, caption="Исходное изображение", use_container_width=True)
                        with col2:
                            st.image(result_image, caption="Результат детекции", use_container_width=True)
                            
                    else:
                        st.error(f"Ошибка сервера: {response.status_code}")
                        
                except Exception as e:
                    st.error(f"Нет связи с Backend-сервером. Убедитесь, что FastAPI запущен. ({e})")
        else:
            # ДО НАЖАТИЯ КНОПКИ: Показываем аккуратное превью по центру
            st.info("Нажмите кнопку «Запустить анализ изображения» выше, чтобы получить результат.")
            col_spacer1, col_center, col_spacer2 = st.columns([1, 2, 1])
            with col_center:
                st.image(uploaded_file, caption="Предварительный просмотр", use_container_width=True)
    else:
        st.info("Загрузите изображение в меню слева для начала работы.")

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