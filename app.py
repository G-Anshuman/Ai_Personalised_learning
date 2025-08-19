import streamlit as st
import pandas as pd
import joblib
import os

# Load recommendation model (already trained and saved in repo)
@st.cache_resource
def load_model():
    return joblib.load("best_rf_model.pkl")

# Helper: Load subject dataset
def load_subject_data(subject):
    file_map = {
        "Percentages": "Percentages.csv",
        "Time and Work": "Time And work.csv",
        "Time Speed Distance": "Time Speed Distance.csv"
    }
    return pd.read_csv(file_map[subject])

# Streamlit UI
st.title("📚 Student Diagnostic App")

# Sidebar: choose subject
subject = st.sidebar.selectbox("Choose a subject", ["Percentages", "Time and Work", "Time Speed Distance"])

# Load dataset for chosen subject
df = load_subject_data(subject)

# Select 10 diagnostic questions
diagnostic_df = df[df["type"].str.lower() == "diagnostic"].head(10)

# Storage for quiz answers
if "responses" not in st.session_state:
    st.session_state.responses = []
if "score" not in st.session_state:
    st.session_state.score = 0
if "quiz_complete" not in st.session_state:
    st.session_state.quiz_complete = False

# Quiz Flow
st.header(f"Diagnostic Quiz - {subject}")

for idx, row in diagnostic_df.iterrows():
    question = row["question_text"]
    options = [row["OptionA"], row["OptionB"], row["OptionC"], row["OptionD"]]
    correct = row["Correct_Option"]

    choice = st.radio(f"Q{row['QuestionID']}: {question}", options, key=f"q_{row['QuestionID']}")

    if st.button(f"Submit Q{row['QuestionID']}", key=f"submit_{row['QuestionID']}"):
        st.session_state.responses.append({
            "QuestionID": row["QuestionID"],
            "question": question,
            "selected": choice,
            "correct": correct,
            "is_correct": (choice[0] == correct)  # compare option label A/B/C/D
        })
        if choice[0] == correct:
            st.session_state.score += 1

# After quiz submission
if st.button("Finish Quiz"):
    st.session_state.quiz_complete = True

if st.session_state.quiz_complete:
    st.subheader("✅ Quiz Completed!")
    st.write(f"Score: {st.session_state.score} / {len(diagnostic_df)}")

    # Show correct vs incorrect
    for r in st.session_state.responses:
        st.write(
            f"Q{r['QuestionID']}: {r['question']} | "
            f"Your Answer: {r['selected']} | Correct: {r['correct']} | "
            f"{'✅ Correct' if r['is_correct'] else '❌ Wrong'}"
        )

    # Feedback
    feedback = st.radio("How did you find the quiz?", ["Easy", "Okay", "Difficult"])
    time_taken = st.number_input("Time taken (minutes)", min_value=1, step=1)
    attempt = 1  # for now always 1st attempt

    # Save for recommendation
    student_data = pd.DataFrame([{
        "subject": subject,
        "score": st.session_state.score,
        "feedback": feedback,
        "time_taken": time_taken,
        "attempt": attempt
    }])

    # Load model
    model = load_model()

    # Encode + predict
    recommendation = model.predict(student_data)[0]
    st.subheader("🎯 Recommendation")
    st.write(recommendation)
