import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv(r"./diabetes.csv")

# Completely revamped CSS for a new design
st.markdown("""
    <style>
        /* General body styling */
        body {
            background-color: #f0f4f8;
        }
        .main-header {
            font-size: 48px;
            font-family: 'Georgia', serif;
            text-align: center;
            color: #2c3e50;
            margin-top: 10px;
        }
        .sub-header {
            font-size: 20px;
            font-family: 'Arial', sans-serif;
            text-align: center;
            color: #7f8c8d;
        }
        .divider {
            border: none;
            border-top: 2px solid #dfe6e9;
            margin: 20px 0;
        }
        .highlight {
            font-size: 18px;
            font-weight: bold;
            color: #16a085;
        }
        /* Sidebar styles */
        .css-1aumxhk {
            background-color: #2c3e50;
        }
        .sidebar-title {
            font-size: 24px;
            font-family: 'Georgia', serif;
            color: #ecf0f1;
        }
        .sidebar-description {
            font-size: 14px;
            color: #bdc3c7;
        }
        /* Button styles */
        .stButton>button {
            background-color: #1abc9c;
            color: white;
            font-size: 16px;
            border-radius: 8px;
            padding: 10px;
            border: none;
        }
        .stButton>button:hover {
            background-color: #16a085;
            color: white;
        }
        /* Result text styling */
        .result-text {
            font-size: 22px;
            font-weight: bold;
            text-align: center;
        }
        .result-success {
            color: #2ecc71;
        }
        .result-danger {
            color: #e74c3c;
        }
    </style>
""", unsafe_allow_html=True)

# Main title and subtitle
st.markdown("<h1 class='main-header'>Diabetes Risk Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Analyze your health data to check your diabetes risk.</p>", unsafe_allow_html=True)
st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# Sidebar header and description
st.sidebar.markdown("<h2 class='sidebar-title'>Health Data Input</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p class='sidebar-description'>Please adjust the sliders to provide your health data.</p>", unsafe_allow_html=True)

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

# Displaying the entered data
st.markdown("<h2 style='color: #2c3e50;'>Your Entered Data</h2>", unsafe_allow_html=True)
st.dataframe(user_data)

# Data preparation and model training
x = df.drop(['Outcome'], axis=1)
y = df['Outcome']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

rf = RandomForestClassifier()
rf.fit(x_train, y_train)

# Button to trigger prediction
if st.button('Predict Risk'):
    st.markdown("<h3 style='text-align: center;'>Analyzing your data...</h3>", unsafe_allow_html=True)
    progress = st.progress(0)
    for percent in range(100):
        progress.progress(percent + 1)

    prediction = rf.predict(user_data)
    
    # Display prediction results
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    if prediction[0] == 0:
        st.markdown("<p class='result-text result-success'>You are not at risk of diabetes.</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p class='result-text result-danger'>You are at risk of diabetes.</p>", unsafe_allow_html=True)
    
    # Display model accuracy
    accuracy = accuracy_score(y_test, rf.predict(x_test)) * 100
    st.markdown(f"<p style='text-align: center; font-size: 18px; color: #7f8c8d;'>Model Accuracy: {accuracy:.2f}%</p>", unsafe_allow_html=True)
else:
    st.markdown("<h3 style='text-align: center;'>Enter your details and click 'Predict Risk' to see results.</h3>", unsafe_allow_html=True)
