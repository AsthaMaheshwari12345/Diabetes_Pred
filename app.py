import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv(r"./diabetes.csv")

# Updated CSS for the new layout
st.markdown("""
    <style>
        body {
            background-color: #f5f7fa;
        }
        .main-header {
            font-size: 42px;
            font-family: 'Verdana', sans-serif;
            text-align: center;
            color: #34495e;
            margin-top: 20px;
        }
        .sub-header {
            font-size: 20px;
            font-family: 'Verdana', sans-serif;
            color: #7f8c8d;
            text-align: center;
        }
        .divider {
            border: none;
            border-top: 3px solid #ecf0f1;
            margin: 20px 0;
        }
        .highlight {
            font-size: 18px;
            font-weight: bold;
            color: #2ecc71;
        }
        .sidebar-header {
            font-size: 22px;
            font-family: 'Verdana', sans-serif;
            color: #2c3e50;
        }
        .sidebar-description {
            font-size: 14px;
            color: #7f8c8d;
        }
        .btn-primary {
            background-color: #2c3e50;
            color: #ffffff;
            font-size: 16px;
            border-radius: 5px;
            padding: 10px;
            text-align: center;
        }
        .btn-primary:hover {
            background-color: #34495e;
            color: #ffffff;
        }
        .result-text {
            font-size: 20px;
            font-weight: bold;
            color: #e74c3c;
            text-align: center;
        }
        .result-success {
            font-size: 20px;
            font-weight: bold;
            color: #2ecc71;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>Diabetes Risk Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Use your health data to check your diabetes risk.</p>", unsafe_allow_html=True)
st.markdown("<hr class='divider'>", unsafe_allow_html=True)

st.sidebar.markdown("<h2 class='sidebar-header'>Health Information</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p class='sidebar-description'>Enter your health parameters below.</p>", unsafe_allow_html=True)

def get_user_input():
    pregnancies = st.sidebar.slider('Pregnancies', min_value=0, max_value=17, value=3, format="%d")
    bp = st.sidebar.slider('Blood Pressure (mm Hg)', min_value=0, max_value=122, value=70, format="%d")
    bmi = st.sidebar.slider('BMI (Body Mass Index)', min_value=0.0, max_value=67.0, value=20.0, format="%.1f")
    glucose = st.sidebar.slider('Glucose Level (mg/dL)', min_value=0, max_value=200, value=120, format="%d")
    skinthickness = st.sidebar.slider('Skin Thickness (mm)', min_value=0, max_value=100, value=20, format="%d")
    dpf = st.sidebar.slider('Diabetes Pedigree Function', min_value=0.0, max_value=2.4, value=0.47, format="%.2f")
    insulin = st.sidebar.slider('Insulin Level (IU/mL)', min_value=0, max_value=846, value=79, format="%d")
    age = st.sidebar.slider('Age (years)', min_value=21, max_value=88, value=33, format="%d")

    user_data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': bp,
        'SkinThickness': skinthickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }

    features = pd.DataFrame(user_data, index=[0])
    return features

user_data = get_user_input()

st.markdown("<h2 style='color: #34495e;'>Overview of Entered Data</h2>", unsafe_allow_html=True)
st.dataframe(user_data)

x = df.drop(['Outcome'], axis=1)
y = df['Outcome']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

rf = RandomForestClassifier()
rf.fit(x_train, y_train)

if st.button('Check Risk', key="primary"):
    st.markdown("<h3>Analyzing data...</h3>", unsafe_allow_html=True)
    progress = st.progress(0)
    for percent in range(100):
        progress.progress(percent + 1)

    prediction = rf.predict(user_data)
    
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("<h2>Prediction</h2>", unsafe_allow_html=True)
    if prediction[0] == 0:
        st.markdown("<p class='result-success'>You are not at risk of diabetes.</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p class='result-text'>You are at risk of diabetes.</p>", unsafe_allow_html=True)
    
    accuracy = accuracy_score(y_test, rf.predict(x_test)) * 100
    st.markdown(f"<p style='color: #34495e;'>Model Accuracy: {accuracy:.2f}%</p>", unsafe_allow_html=True)
else:
    st.markdown("<h3>Enter your details and click 'Check Risk'</h3>", unsafe_allow_html=True)
