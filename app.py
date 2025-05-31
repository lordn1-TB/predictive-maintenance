import streamlit as st

# Настройка навигации
pages = {
    "Анализ и модель": "analysis_and_model.py",
    "Презентация": "presentation.py"
}

# Навигация в боковой панели
st.sidebar.title("Навигация")
selection = st.sidebar.radio("Перейти", list(pages.keys()))

# Загрузка выбранной страницы
page = pages[selection]
try:
    with open(page) as f:
        code = compile(f.read(), page, 'exec')
        exec(code, globals())
except Exception as e:
    st.error(f"Ошибка загрузки страницы {page}: {str(e)}")