import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv(r"./diabetes.csv")

# Add obfuscated styles
st.markdown("""
    <style>
    .custom-header {
        text-align: center;
        color: #FF9F40;
        font-size: 2rem;
        margin-bottom: 10px;
    }
    .custom-subtext {
        text-align: center;
        font-size: 1rem;
        color: #555;
    }
    .custom-divider {
        border: none;
        border-top: 1px solid #CCC;
        margin: 20px 0;
    }
    .custom-result {
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .custom-accuracy {
        color: #333;
        font-size: 1.2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# App title
st.markdown("<h1 class='custom-header'>Diabetes Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p class='custom-subtext'>This app predicts whether a patient is diabetic based on their health data.</p>", unsafe_allow_html=True)
st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

st.sidebar.header('Enter Patient Data')
st.sidebar.write("Please provide the following details for a diabetes checkup:")

# Function to collect user input
def calc():
    pregnancies = st.sidebar.number_input('Pregnancies', min_value=0, max_value=17, value=3)
    bp = st.sidebar.number_input('Blood Pressure', min_value=0, max_value=122, value=70)
    bmi = st.sidebar.number_input('BMI', min_value=0, max_value=67, value=20)
    glucose = st.sidebar.number_input('Glucose', min_value=0, max_value=200, value=120)
    skinthickness = st.sidebar.number_input('Skin Thickness', min_value=0, max_value=100, value=20)
    dpf = st.sidebar.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.4, value=0.47)
    insulin = st.sidebar.number_input('Insulin', min_value=0, max_value=846, value=79)
    age = st.sidebar.number_input('Age', min_value=21, max_value=88, value=33)

    output = {
        'pregnancies': pregnancies,
        'glucose': glucose,
        'bp': bp,
        'skinthickness': skinthickness,
        'insulin': insulin,
        'bmi': bmi,
        'dpf': dpf,
        'age': age
    }
    report_data = pd.DataFrame(output, index=[0])
    return report_data

# Get user data
user_data = calc()

# Display user data
st.subheader('Patient Data Summary')
st.write(user_data)

# Train Random Forest model
x = df.drop(['Outcome'], axis=1)
y = df['Outcome']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

progress = st.progress(0)
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
progress.progress(100)

# Prediction
result = rf.predict(user_data)

# Display prediction
st.subheader('Prediction Result:')
output = 'You are not Diabetic' if result[0] == 0 else 'You are Diabetic'
st.markdown(
    f"<p class='custom-result' style='color: {'#4CAF50' if result[0] == 0 else '#FF4136'};'>{output}</p>",
    unsafe_allow_html=True
)

# Display model accuracy
accuracy = accuracy_score(y_test, rf.predict(x_test)) * 100
st.subheader('Model Accuracy:')
st.markdown(f"<p class='custom-accuracy'>{accuracy:.2f}%</p>", unsafe_allow_html=True)
