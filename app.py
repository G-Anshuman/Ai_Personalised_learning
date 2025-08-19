import streamlit as st
import pandas as pd
import joblib
import time

# -------------------
# 1. Load assets
# -------------------
# Load datasets (replace with your repo paths or raw GitHub links)
datasets = {
    "Time & Work": "Time And work.csv",
    "Percentages": "Percentages.csv",
    "Time Speed Distance": "Time Speed Distance.csv"
}

# Load recommendation model + encoders
model = joblib.load("best_rf_model.pkl")
feature_encoders = joblib.load("feature_encoders.pkl")
target_encoder = joblib.load("target_encoder.pkl")

# -------------------
# 2. App layout
# -------------------
st.title("📚 Drona AI - Student Diagnostic Test")

# Student info
student_name = st.text_input("Enter your name")
subject = st.selectbox("Choose a subject", list(datasets.keys()))

# -------------------
# 3. Diagnostic Quiz
# -------------------
if subject:
    df = pd.read_csv(datasets[subject])

    # filter only diagnostic questions
    diagnostic_df = df[df["Question Type"] == "diagnostic"].head(10)  

    st.subheader(f"Diagnostic Quiz - {subject}")

    student_answers = {}
    start_time = time.time()

    with st.form("quiz_form"):
        for i, row in diagnostic_df.iterrows():
            st.write(f"**Q{i+1}: {row['Question']}**")
            options = [row['Option1'], row['Option2'], row['Option3'], row['Option4']]
            answer = st.radio("Select answer:", options, key=f"q{i}")
            student_answers[i] = answer

        submitted = st.form_submit_button("Submit Answers")

    # -------------------
    # 4. Results + Feedback
    # -------------------
    if submitted:
        end_time = time.time()
        time_taken = round(end_time - start_time, 2)

        score = 0
        correct, incorrect = [], []

        for i, row in diagnostic_df.iterrows():
            correct_ans = row['Correct Answer']
            if student_answers[i] == correct_ans:
                score += 1
                correct.append(row['Question'])
            else:
                incorrect.append(row['Question'])

        st.success(f"✅ You scored {score} / {len(diagnostic_df)}")
        st.write("Correct Answers:", correct)
        st.write("Incorrect Answers:", incorrect)
        st.write(f"⏱ Time Taken: {time_taken} seconds")

        # Feedback
        feedback = st.text_area("Give feedback on the test")

        # -------------------
        # 5. Store attempt data
        # -------------------
        attempt_data = {
            "Name": student_name,
            "Subject": subject,
            "Score": score,
            "TimeTaken": time_taken,
            "Feedback": feedback,
            "Attempt": 1
        }
        st.session_state["attempt_data"] = attempt_data  

        # -------------------
        # 6. Recommendation
        # -------------------
        if st.button("Get Recommendation"):
            df_student = pd.DataFrame([attempt_data])

            # Encode features like in your notebook
            for col, le in feature_encoders.items():
                if col in df_student:
                    df_student[col] = le.transform(df_student[col])

            y_pred = model.predict(df_student)
            recommendation = target_encoder.inverse_transform(y_pred)[0]

            st.info(f"📌 Recommendation: {recommendation}")
