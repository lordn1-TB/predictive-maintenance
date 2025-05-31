
# 🏭 Predictive Maintenance: Прогнозирование отказов оборудования

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.45.1-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<div align="center">
  <img src="https://via.placeholder.com/800x300?text=Predictive+Maintenance+Dashboard" alt="Интерфейс приложения">
</div>

## 📝 Описание
Streamlit-приложение для предиктивного обслуживания промышленного оборудования. Система предсказывает вероятность отказа оборудования на основе данных датчиков.

**Ключевые особенности**:
- Загрузка данных из CSV или UCI репозитория
- Визуализация показателей оборудования
- Сравнение нескольких ML-моделей
- Интерактивный интерфейс для прогнозирования

## 🛠 Технологии
- **Frontend**: Streamlit
- **ML**: Scikit-learn, XGBoost
- **Анализ данных**: Pandas, NumPy
- **Визуализация**: Matplotlib, Seaborn
- **Развертывание**: Docker (опционально)

## 🚀 Быстрый старт

### Установка
```bash
# Клонировать репозиторий
git clone https://github.com/lordn1-TB/predictive-maintenance.git
cd predictive-maintenance

# Создать виртуальное окружение
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Установить зависимости
pip install -r requirements.txt
```

### Запуск
```bash
streamlit run app.py
```
Приложение будет доступно по адресу: [http://localhost:8501](http://localhost:8501)

## 🏗 Структура проекта
```
predictive-maintenance/
├── app.py                 # Главный скрипт Streamlit
├── analysis_and_model.py  # Анализ данных и ML модели
├── presentation.py        # Презентация проекта
├── requirements.txt       # Зависимости
├── data/                  # Пример данных
│   └── sample_data.csv
├── .gitignore
└── README.md
```

## 📈 Результаты моделирования
Модель | Accuracy | ROC-AUC | Precision | Recall
-------|----------|---------|-----------|-------
Random Forest | 0.95 | 0.98 | 0.93 | 0.91
XGBoost | 0.93 | 0.97 | 0.91 | 0.89
Logistic Regression | 0.88 | 0.92 | 0.85 | 0.82

## 📌 Использование
1. На главной странице выберите источник данных
2. Обучите модель (по умолчанию Random Forest)
3. Введите параметры оборудования для прогноза
4. Получите вероятность отказа

## 🌐 Развертывание
### Docker
```bash
docker build -t predictive-maintenance .
docker run -p 8501:8501 predictive-maintenance
```

## 🤝 Контакты
- **Автор**: [Ваше Имя]
- **Email**: ilnur.kasimov.92@mail.ru
- **GitHub**: [@lordn1-TB](https://github.com/lordn1-TB)
```