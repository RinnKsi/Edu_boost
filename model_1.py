from flask import Flask, request, jsonify, render_template
import numpy as np
import lightgbm as lgb
import joblib
import pandas as pd

app = Flask(__name__)

def load_model():
    try:
        return joblib.load('improved_model.joblib')
    except:
        print("Создание новой улучшенной модели...")
        np.random.seed(42)
        size = 15000
        
        # Генерация более реалистичных данных
        sleep = np.clip(np.random.normal(loc=7.5, scale=1.2, size=size), 4, 10)
        health = np.random.randint(2, 6, size=size)
        stress = np.random.randint(1, 6, size=size)
        mood = np.random.randint(1, 6, size=size)  # Шкала настроения 1-5
        adhd = np.random.choice([0, 1], size=size, p=[0.85, 0.15])
        
        # Улучшенная формула продуктивности с учетом настроения
        base_productivity = (
            np.sqrt(sleep) * 0.9 + 
            health * 0.7 - 
            np.sqrt(stress) * 1.1 +
            mood * 0.5 +  # Вклад настроения
            np.random.normal(1.5, 0.4, size=size)
        )
        
        # Применяем реалистичные ограничения
        productivity = np.where(
            adhd == 1,
            np.clip(base_productivity * 0.75, 1.5, 4.5),  # Диапазон для СДВГ
            np.clip(base_productivity, 2.0, 7.5)          # Диапазон без СДВГ
        )
        
        # Формируем DataFrame
        data = {
            'sleep': sleep,
            'health': health,
            'stress': stress,
            'mood': mood,  # Добавлен параметр настроения
            'adhd': adhd,
            'study': np.clip(np.random.normal(3.5, 1.2, size=size), 1, 6),
            'max_study': np.where(adhd == 1, 4.5, 7.5),
            'free_time': np.clip(np.random.normal(3, 1, size=size), 1, 6),
            'productive_time': productivity
        }
        
        df = pd.DataFrame(data)
        
        # Улучшенная модель
        model = lgb.LGBMRegressor(
            num_leaves=20,
            n_estimators=150,
            learning_rate=0.07,
            min_child_samples=25,
            random_state=42,
            n_jobs=-1,
            reg_alpha=0.1,
            reg_lambda=0.1
        )
        
        model.fit(
            df.drop('productive_time', axis=1),
            df['productive_time'],
            feature_name=['sleep', 'health', 'stress', 'mood', 'adhd', 'study', 'max_study', 'free_time'],
            categorical_feature=['adhd']
        )
        
        joblib.dump(model, 'improved_model.joblib')
        return model

model = load_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Обработка СДВГ
        has_adhd = data.get('adhd', 0)
        max_limit = 4.5 if has_adhd else min(data.get('max_study', 7.5), 7.5)
        
        # Подготовка данных с базовыми значениями (добавлено mood)
        input_data = np.array([[
            data.get('sleep', 7),
            data.get('health', 4),
            data.get('stress', 3),
            data.get('mood', 4),  # Новый параметр
            has_adhd,
            data.get('study', 3.5),
            max_limit,
            data.get('free_time', 3)
        ]], dtype=np.float32)
        
        # Получаем и корректируем предсказание
        raw_pred = model.predict(input_data)[0]
        
        # Динамическая коррекция
        if has_adhd:
            adjusted_pred = raw_pred * 0.82  # Коррекция для СДВГ
        else:
            adjusted_pred = raw_pred * 0.92  # Меньшая коррекция для обычных случаев
            
        # Обеспечиваем реалистичный диапазон
        final_pred = np.clip(adjusted_pred, 1.5 if has_adhd else 2.0, max_limit)
        
        # Округляем до 0.25 часа (15 минут)
        final_pred = round(final_pred * 4) / 4
        
        # Формируем рекомендации (добавлены проверки настроения)
        recommendations = []
        sleep = data.get('sleep', 7)
        stress = data.get('stress', 3)
        mood = data.get('mood', 4)
        
        if sleep < 6:
            recommendations.append("🔴 Недостаток сна! Стремитесь к 7-8 часам")
        elif sleep > 9:
            recommendations.append("🟡 Избыток сна может снижать продуктивность")
            
        if stress > 4:
            recommendations.append("🧠 Высокий стресс: попробуйте технику 5-5-5 (5 глубоких вдохов)")
            
        if mood < 3:
            recommendations.append("😔 Низкое настроение: возможно, стоит сделать перерыв")
        elif mood > 4:
            recommendations.append("😊 Отличное настроение: используйте этот заряд энергии!")
            
        if has_adhd:
            recommendations.append("✨ Для СДВГ: используйте таймер Pomodoro (25/5 мин)")
        
        return jsonify({
            'prediction': float(final_pred),
            'max_recommended': float(max_limit),
            'recommendations': recommendations,
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'failed'
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)