import streamlit as st

def main():
    st.title("Презентация проекта")
    
    slides = [
        {
            "title": "Введение",
            "content": """
            - Цель: Прогнозирование отказов оборудования
            - Датасет: AI4I 2020 Predictive Maintenance
            - 10,000 записей с 14 признаками
            """
        },
        {
            "title": "Методология", 
            "content": """
            - Использованные модели:
              - Логистическая регрессия
              - Случайный лес
              - XGBoost
            - Метрики оценки:
              - Accuracy
              - ROC-AUC
            """
        },
        {
            "title": "Результаты",
            "content": """
            - Лучшая модель: Random Forest
            - Точность: 95%
            - ROC-AUC: 0.98
            """
        }
    ]
    
    if 'current_slide' not in st.session_state:
        st.session_state.current_slide = 0
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.session_state.current_slide > 0:
            if st.button("Назад"):
                st.session_state.current_slide -= 1
                st.rerun()
    
    with col2:
        if st.session_state.current_slide < len(slides)-1:
            if st.button("Вперед"):
                st.session_state.current_slide += 1
                st.rerun()
    
    slide = slides[st.session_state.current_slide]
    st.markdown(f"## {slide['title']}")
    st.markdown(slide['content'])

if __name__ == "__main__":
    main()