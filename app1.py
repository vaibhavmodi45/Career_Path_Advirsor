import streamlit as st
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np


st.markdown("""
    <style>

        .navbar {
        margin-top: -50px;
        background-color: ;
        padding-left:2px;
        text-align: left;
        font-weight: bolder;
        font-size: 54px;
        color: black;
    }

    hr {
        margin-top: 8px;
        margin-bottom: 48px;
    }    
        
        .footer {
            text-align: center;
            padding: 10px;
            background-color: #2c2b62;
            color: white;
            margin-top: 100px;
            width: 100%;
            margin-bottom: 0;
    }
    
    </style>
    """, unsafe_allow_html=True)

# Sample data for demonstration
data = {
    'Interest': ['Technology', 'Arts', 'Science', 'Business', 'Technology', 'Arts', 'Science', 'Business'],
    'Skills': ['Programming', 'Design', 'Analysis', 'Communication', 'Data Analysis', 'Creative Writing', 'Research', 'Marketing'],
    'Career': ['Software Engineer', 'Graphic Designer', 'Research Scientist', 'Business Analyst', 
               'Data Scientist', 'Content Writer', 'Biologist', 'Marketing Manager']
}

# Create a DataFrame from the sample data
df = pd.DataFrame(data)

# Preprocessing: Convert skills to a numerical format
# This is a simplified example; you should ideally use one-hot encoding or similar techniques
def preprocess_data(df):
    df['Skills'] = df['Skills'].apply(lambda x: x.split(', '))
    return df

df = preprocess_data(df)

# Create a mapping for interests and skills
interest_mapping = {interest: i for i, interest in enumerate(df['Interest'].unique())}
skill_mapping = {skill: i for i, skill in enumerate(set(df['Skills'].sum()))}

# Prepare features and labels
X = []
for _, row in df.iterrows():
    interest_vector = np.zeros(len(interest_mapping))
    skill_vector = np.zeros(len(skill_mapping))
    interest_vector[interest_mapping[row['Interest']]] = 1
    for skill in row['Skills']:
        skill_vector[skill_mapping[skill]] = 1
    X.append(np.concatenate([interest_vector, skill_vector]))

X = np.array(X)
y = df['Career'].values

# Train a Decision Tree Classifier
model = DecisionTreeClassifier()
model.fit(X, y)

# Streamlit UI
# Navbar
st.markdown('<div class="navbar">CAREER PATH ADVISOR 🤖 <hr></div>', unsafe_allow_html=True)
st.header("Tell us about your interests and skills")

# User input
interest = st.selectbox("What is your main interest?", list(interest_mapping.keys()))
skills = st.multiselect("Select your skills:", list(skill_mapping.keys()))

# Process input for prediction
if st.button("Suggest Career"):
    # Create user input vector
    user_interest_vector = np.zeros(len(interest_mapping))
    user_skill_vector = np.zeros(len(skill_mapping))
    
    user_interest_vector[interest_mapping[interest]] = 1
    for skill in skills:
        user_skill_vector[skill_mapping[skill]] = 1
    
    user_input_vector = np.concatenate([user_interest_vector, user_skill_vector]).reshape(1, -1)
    
    # Predict career
    prediction = model.predict(user_input_vector)
    st.success(f"Suggested Career: {prediction[0]}")


# Additional information
st.markdown("""
### How It Works
This app uses a simple decision tree model to suggest careers based on your interests and skills. 
""")

st.markdown("1. Select your main interest from the dropdown.")
st.image('1.png')
st.markdown("2. Choose your skills from the list.")
st.image('2.png')
st.markdown("3. Click on 'Suggest Career' to see your recommended career path.")
st.image('3.png')

# Footer
st.markdown('<div class="footer">All Rights Reserved <br> Developed by Team @Career_Path_Advisor</div>', unsafe_allow_html=True)