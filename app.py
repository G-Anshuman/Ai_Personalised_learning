import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle
import joblib
import os # Import the os module

# --- Configuration & Data Loading ---
CSV_LINKS = {
    "student_data": "https://raw.githubusercontent.com/G-Anshuman/Ai_Personalised_learning/refs/heads/main/Students_expanded_dataset_noisy.csv",
    "time_speed_distance": "https://raw.githubusercontent.com/G-Anshuman/Ai_Personalised_learning/refs/heads/main/Time%20Speed%20Distance.csv",
    "percentages": "https://raw.githubusercontent.com/G-Anshuman/Ai_Personalised_learning/refs/heads/main/Percentages.csv",
    "time_and_work": "https://raw.githubusercontent.com/G-Anshuman/Ai_Personalised_learning/refs/heads/main/Time%20And%20work.csv"
}

# --- Utility Functions ---
@st.cache_data
def load_data(url):
    """Loads CSV data from a URL with caching."""
    return pd.read_csv(url)

@st.cache_resource
def get_trained_model():
    """
    Trains the Random Forest model and saves the necessary files.
    This function is cached to run only once.
    """
    st.info("Training the model... This will only happen on the first run.")
    
    df_student = load_data(CSV_LINKS["student_data"])
    if "Student ID" in df_student.columns:
        df_student = df_student.drop(columns=["Student ID"])
    
    # Preprocessing
    categorical_cols = ["Topic", "Feedback"]
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        if col == "Topic":
            df_student[col] = df_student[col].str.lower().str.replace(' ', '').str.replace('and', '').str.replace('_', '')
        df_student[col] = le.fit_transform(df_student[col])
        label_encoders[col] = le

    df_student["Efficiency_Score"] = df_student["Score"] / (df_student["Time Taken (seconds)"] + 1)
    df_student["Norm_Time"] = df_student["Time Taken (seconds)"] / (df_student["Attempts"] + 1)
    df_student["Accuracy_per_Attempt"] = df_student["Score"] / (df_student["Attempts"] + 1)
    df_student["Time_per_Point"] = df_student["Time Taken (seconds)"] / df_student["Score"].replace(0, np.nan)
    df_student["Time_per_Point"] = df_student["Time_per_Point"].fillna(df_student["Time Taken (seconds)"])

    X = df_student.drop(columns=["Recommended Path"])
    y = df_student["Recommended Path"]
    
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)
    
    # Train the model
    rf = RandomForestClassifier(n_estimators=250, max_depth=30, min_samples_split=2, min_samples_leaf=2, max_features="sqrt", bootstrap=False, random_state=42)
    best_rf_model = rf.fit(X, y_encoded)
    
    # Return all the necessary objects
    return best_rf_model, label_encoders, target_encoder, X.columns.tolist()

def get_diagnostic_questions(topic_name):
    """Loads and returns diagnostic questions for a topic."""
    df = load_data(CSV_LINKS[topic_name])
    
    if 'Type' in df.columns:
        diagnostic_questions = df[df['Type'].str.lower() == "diagnostic"]
    elif 'type' in df.columns:
        diagnostic_questions = df[df['type'].str.lower() == "diagnostic"]
    else:
        diagnostic_questions = pd.DataFrame() 

    return diagnostic_questions.head(10).to_dict('records')


# --- Main App Logic ---
def main():
    st.title("Drona AI: Personalized Learning Path Recommender")
    st.markdown("This app uses a machine learning model to recommend a personalized learning path based on your performance in a diagnostic test.")

    # Initialize session state variables
    if 'student_name' not in st.session_state: st.session_state.student_name = ""
    if 'test_started' not in st.session_state: st.session_state.test_started = False
    if 'test_submitted' not in st.session_state: st.session_state.test_submitted = False
    if 'current_question_index' not in st.session_state: st.session_state.current_question_index = 0
    if 'answers' not in st.session_state: st.session_state.answers = []
    if 'start_time' not in st.session_state: st.session_state.start_time = None
    if 'diagnostic_questions' not in st.session_state: st.session_state.diagnostic_questions = []

    # --- Step 1: Student Login & Topic Selection ---
    if not st.session_state.test_started:
        if not st.session_state.student_name:
            student_name_input = st.text_input("Enter your name:")
            if student_name_input:
                st.session_state.student_name = student_name_input
                st.success(f"Welcome, {st.session_state.student_name}! 👋")
                st.rerun()
        else:
            st.sidebar.info(f"Welcome, {st.session_state.student_name}!")
            st.subheader("Choose a topic:")
            topics_display = [t.replace('_', ' ').title() for t in list(CSV_LINKS.keys())[1:]]
            selected_topic_display = st.selectbox("Select a topic:", topics_display)
            
            if st.button("Start Diagnostic Test"):
                st.session_state.selected_topic = selected_topic_display.replace(' ', '_').lower()
                st.session_state.test_started = True
                st.session_state.start_time = time.time()
                st.session_state.diagnostic_questions = get_diagnostic_questions(st.session_state.selected_topic)
                st.rerun()

    # --- Step 2: Running the Diagnostic Test (Question by Question) ---
    if st.session_state.test_started and not st.session_state.test_submitted:
        questions = st.session_state.diagnostic_questions
        total_questions = len(questions)

        if st.session_state.current_question_index < total_questions:
            q_num = st.session_state.current_question_index + 1
            question_data = questions[st.session_state.current_question_index]
            
            difficulty_col = 'DifficultyLevel' if 'DifficultyLevel' in question_data else 'difficulty'
            question_col = 'QuestionText' if 'QuestionText' in question_data else 'question_text'
            
            st.subheader(f"Question {q_num}/{total_questions}")
            st.markdown(f"**[{question_data.get(difficulty_col, 'N/A')}]** {question_data.get(question_col, 'Question not found.')}")

            options = {
                'A': question_data.get('OptionA', 'N/A'),
                'B': question_data.get('OptionB', 'N/A'),
                'C': question_data.get('OptionC', 'N/A'),
                'D': question_data.get('OptionD', 'N/A')
            }

            radio_options = [f"{letter}. {text}" for letter, text in options.items()]
            user_choice = st.radio("Choose your answer:", radio_options, key=f"q{q_num}")

            if st.button("Next Question"):
                # Store only the letter of the selected answer
                selected_answer_letter = user_choice.split('.')[0].strip()
                st.session_state.answers.append(selected_answer_letter)
                st.session_state.current_question_index += 1
                st.rerun()
        else:
            st.session_state.test_submitted = True
            st.rerun()

    # --- Step 3: Test Results & Recommendation ---
    if st.session_state.test_submitted:
        end_time = time.time()
        total_time = round(end_time - st.session_state.start_time, 2)

        # Calculate score and wrong answers
        score = 0
        wrong_answers = []
        total_marks = 0
        questions = st.session_state.diagnostic_questions

        for i, user_ans_letter in enumerate(st.session_state.answers):
            q_data = questions[i]
            correct_answer_raw = str(q_data.get('CorrectAnswer') or q_data.get('Correct_Option')).strip()
            marks = int(q_data.get('Marks', 4))
            total_marks += marks

            # Determine the correct answer letter
            correct_opt_letter = None
            if correct_answer_raw.upper() in ["A", "B", "C", "D"]:
                correct_opt_letter = correct_answer_raw
            else:
                for letter, opt_col in zip(["A", "B", "C", "D"], ["OptionA", "OptionB", "OptionC", "OptionD"]):
                    if str(q_data.get(opt_col)).strip().lower() == correct_answer_raw.lower():
                        correct_opt_letter = letter
                        break

            # Evaluate against the stored letter
            if user_ans_letter.upper() == correct_opt_letter.upper():
                score += marks
            else:
                wrong_answers.append({
                    "Q_No": i + 1,
                    "Your_Answer": user_ans_letter,
                    "Correct_Answer": correct_opt_letter
                })

        st.subheader("Diagnostic Test Results")
        st.markdown(f"📊 **Score:** {score}/{total_marks} marks")
        st.markdown(f"📝 **Total Questions:** {len(questions)}")
        st.markdown(f"⏳ **Time Taken:** {total_time} seconds")
        
        if wrong_answers:
            st.subheader("❌ Incorrect Answers:")
            for wa in wrong_answers:
                st.markdown(f"Q{wa['Q_No']}: Your Answer = `{wa['Your_Answer']}` | Correct Answer = `{wa['Correct_Answer']}`")
        else:
            st.balloons()
            st.success("🎉 All answers correct! Great job!")

        st.subheader("Select the area where you faced difficulty:")
        feedback_options = ["Speed Test", "Level 1", "Level 2"]
        feedback_str = st.selectbox("Your choice:", feedback_options)
        
        if st.button("Get Recommendation"):
            st.session_state.feedback = feedback_str
            
            with st.spinner("Generating your personalized path..."):
                # Call the cached function to get all objects
                best_rf_model, label_encoders, target_encoder, feature_order = get_trained_model()

                row = {
                    "Topic": st.session_state.selected_topic,
                    "Score": score,
                    "Time Taken (seconds)": total_time,
                    "Feedback": st.session_state.feedback,
                    "Attempts": 1
                }
                df_infer = pd.DataFrame([row])

                df_infer_processed = df_infer.copy()
                df_infer_processed["Topic"] = df_infer_processed["Topic"].str.lower().str.replace(" ", "").str.replace("and", "").str.replace("_", "")
                
                for col, encoder in label_encoders.items():
                    if col in df_infer_processed.columns:
                        try:
                            df_infer_processed[col] = encoder.transform(df_infer_processed[col])
                        except ValueError:
                            st.warning(f"Unseen label in column '{col}'. Using a default value.")
                            df_infer_processed[col] = -1

                df_infer_processed["Efficiency_Score"] = df_infer_processed["Score"] / (df_infer_processed["Time Taken (seconds)"] + 1)
                df_infer_processed["Norm_Time"] = df_infer_processed["Time Taken (seconds)"] / (df_infer_processed["Attempts"] + 1)
                df_infer_processed["Accuracy_per_Attempt"] = df_infer_processed["Score"] / (df_infer_processed["Attempts"] + 1)
                df_infer_processed["Time_per_Point"] = df_infer_processed["Time Taken (seconds)"] / df_infer_processed["Score"].replace(0, np.nan)
                df_infer_processed["Time_per_Point"] = df_infer_processed["Time_per_Point"].fillna(df_infer_processed["Time Taken (seconds)"])
                
                df_infer_processed = df_infer_processed[feature_order]
                
                predicted_path_encoded = best_rf_model.predict(df_infer_processed)
                predicted_path = target_encoder.inverse_transform(predicted_path_encoded)

                st.subheader("Your Recommended Learning Path:")
                st.success(predicted_path[0])

        if st.button("Start Over"):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()

if __name__ == "__main__":
    main()
