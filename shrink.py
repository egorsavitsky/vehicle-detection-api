import torch

def shrink_model(input_name, output_name):
    print(f"Сжимаем {input_name}...")
    try:
        # ДОБАВЛЕН АРГУМЕНТ weights_only=False
        ckpt = torch.load(input_name, map_location='cpu', weights_only=False)
        
        # Удаляем тяжелые данные оптимизатора
        for key in ['optimizer', 'ema', 'updates']:
            if key in ckpt:
                ckpt[key] = None
                
        # Сохраняем очищенную версию
        torch.save(ckpt, output_name)
        print(f"Готово! Сохранен как {output_name}")
    except Exception as e:
        print(f"Ошибка с {input_name}: {e}")

# Применяем к нашим файлам
shrink_model('nano_best.pt', 'nano_ready.pt')
shrink_model('medium_best.pt', 'medium_ready.pt')