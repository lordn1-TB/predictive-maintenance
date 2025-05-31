import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, confusion_matrix, 
                           classification_report, roc_auc_score, roc_curve)

def main():
    st.title("Прогнозирование отказов оборудования")
    
    # Раздел загрузки данных
    st.header("1. Загрузка данных")
    data = load_data()
    
    if data is not None:
        # Раздел предобработки данных
        st.header("2. Предобработка данных")
        data = preprocess_data(data)
        st.write("Предпросмотр обработанных данных:", data.head())
        
        # Раздел разделения данных
        st.header("3. Разделение данных")
        X_train, X_test, y_train, y_test = split_data(data)
        
        # Раздел обучения модели
        st.header("4. Обучение модели")
        model = train_model(X_train, y_train)
        
        # Раздел оценки модели
        st.header("5. Оценка модели")
        evaluate_model(model, X_test, y_test)
        
        # Раздел прогнозирования
        st.header("6. Прогнозирование")
        make_predictions(model)

def load_data():
    """Загрузка данных из файла или репозитория UCI"""
    data = None
    option = st.radio("Выберите источник данных:", 
                     ("Использовать пример данных", "Загрузить свой CSV", "Загрузить из UCI"))
    
    if option == "Использовать пример данных":
        try:
            data = pd.read_csv("data/predictive_maintenance.csv")
            st.success("Пример данных успешно загружен!")
        except:
            st.error("Пример данных не найден. Пожалуйста, загрузите свои данные.")
    
    elif option == "Загрузить свой CSV":
        uploaded_file = st.file_uploader("Загрузите CSV файл", type="csv")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.success("Файл успешно загружен!")
    
    elif option == "Загрузить из UCI":
        try:
            from ucimlrepo import fetch_ucirepo
            ai4i_2020 = fetch_ucirepo(id=601)
            data = pd.concat([ai4i_2020.data.features, ai4i_2020.data.targets], axis=1)
            st.success("Данные успешно загружены из UCI!")
        except Exception as e:
            st.error(f"Ошибка при загрузке данных из UCI: {str(e)}")
    
    if data is not None:
        st.write("Размер данных:", data.shape)
        st.write("Предпросмотр данных:", data.head())
        
    return data

def preprocess_data(data):
    """Предобработка данных"""
    # 1. Удаление ненужных столбцов
    cols_to_drop = ['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    cols_to_drop = [col for col in cols_to_drop if col in data.columns]
    data = data.drop(columns=cols_to_drop)
    
    # 2. Переименование столбцов (удаление единиц измерения в квадратных скобках)
    rename_dict = {
        'Air temperature [K]': 'Air temperature',
        'Process temperature [K]': 'Process temperature',
        'Rotational speed [rpm]': 'Rotational speed',
        'Torque [Nm]': 'Torque',
        'Tool wear [min]': 'Tool wear'
    }
    data = data.rename(columns=rename_dict)
    
    # 3. Кодирование категориальных переменных
    if 'Type' in data.columns:
        le = LabelEncoder()
        data['Type'] = le.fit_transform(data['Type'])
    
    # 4. Проверка на пропущенные значения
    if data.isnull().sum().sum() > 0:
        st.warning("Обнаружены пропущенные значения. Заполняем медианными значениями.")
        data = data.fillna(data.median())
    
    # 5. Масштабирование числовых признаков (используем новые имена столбцов)
    numerical_features = ['Air temperature', 'Process temperature', 
                         'Rotational speed', 'Torque', 'Tool wear']
    numerical_features = [col for col in numerical_features if col in data.columns]
    
    if numerical_features:
        scaler = StandardScaler()
        data[numerical_features] = scaler.fit_transform(data[numerical_features])
    
    return data

def split_data(data):
    """Разделение данных на обучающую и тестовую выборки"""
    X = data.drop(columns=['Machine failure'])
    y = data['Machine failure']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    st.write(f"Размер обучающей выборки: {X_train.shape[0]} образцов")
    st.write(f"Размер тестовой выборки: {X_test.shape[0]} образцов")
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """Обучение и выбор лучшей модели"""
    st.subheader("Выбор модели")
    model_option = st.selectbox(
        "Выберите модель для обучения:",
        ("Логистическая регрессия", "Случайный лес", "XGBoost", "Метод опорных векторов")
    )
    
    if model_option == "Логистическая регрессия":
        model = LogisticRegression(random_state=42)
    elif model_option == "Случайный лес":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_option == "XGBoost":
        model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    elif model_option == "Метод опорных векторов":
        model = SVC(kernel='linear', probability=True, random_state=42)
    
    with st.spinner(f"Обучение {model_option}..."):
        model.fit(X_train, y_train)
    st.success(f"{model_option} успешно обучена!")
    
    return model

def evaluate_model(model, X_test, y_test):
    """Оценка производительности модели"""
    # Прогнозирование
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Расчет метрик
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Отображение метрик
    st.subheader("Производительность модели")
    st.write(f"Точность: {accuracy:.4f}")
    st.write(f"ROC AUC Score: {roc_auc:.4f}")
    
    # Матрица ошибок
    st.subheader("Матрица ошибок")
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Предсказанные')
    ax.set_ylabel('Фактические')
    st.pyplot(fig)
    
    # Отчет классификации
    st.subheader("Отчет классификации")
    st.text(class_report)
    
    # ROC-кривая
    st.subheader("ROC-кривая")
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"{model.__class__.__name__} (AUC = {roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Случайное угадывание')
    ax.set_xlabel('Ложноположительная частота')
    ax.set_ylabel('Истинноположительная частота')
    ax.set_title('ROC-кривая')
    ax.legend()
    st.pyplot(fig)

def make_predictions(model):
    st.subheader("Введите параметры оборудования")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            equipment_type = st.selectbox("Тип оборудования", ["L", "M", "H"])
            air_temp = st.number_input("Температура воздуха (K)", value=300.0)
            process_temp = st.number_input("Температура процесса (K)", value=310.0)
        
        with col2:
            rotational_speed = st.number_input("Скорость вращения (об/мин)", value=1500)
            torque = st.number_input("Крутящий момент (Нм)", value=40.0)
            tool_wear = st.number_input("Износ инструмента (мин)", value=0)
        
        submit_button = st.form_submit_button("Прогнозировать отказ")
    
    if submit_button:
        # Подготовка входных данных с правильными именами признаков
        input_data = pd.DataFrame({
            'Type': [0 if equipment_type == "L" else 1 if equipment_type == "M" else 2],
            'Air temperature': [air_temp],
            'Process temperature': [process_temp],
            'Rotational speed': [rotational_speed],
            'Torque': [torque],
            'Tool wear': [tool_wear]
        })
        
        # Убедимся, что порядок столбцов совпадает с обучающими данными
        input_data = input_data[model.feature_names_in_]
        
        # Прогнозирование
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)[:, 1]
        
        # Отображение результатов
        st.subheader("Результаты прогнозирования")
        if prediction[0] == 1:
            st.error(f"⚠️ Прогнозируется отказ с вероятностью {prediction_proba[0]*100:.2f}%")
        else:
            st.success(f"✅ Отказ не прогнозируется (вероятность отказа {prediction_proba[0]*100:.2f}%)")

if __name__ == "__main__":
    main()