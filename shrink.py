from ultralytics.utils.torch_utils import strip_optimizer

print("Сжимаем модели встроенным методом YOLO...")
# Эта функция аккуратно удаляет градиенты, сохраняя архитектуру YOLO
strip_optimizer('nano_best.pt', 'nano_ready.pt')
strip_optimizer('medium_best.pt', 'medium_ready.pt')
print("✅ Успешно сжато!")
