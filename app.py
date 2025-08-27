import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle
import joblib

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
def train_model():
    """
    Trains the Random Forest model and saves the necessary files.
    This function will be cached and only run once.
    """
    try:
        # Check if model files already exist to avoid re-training
        joblib.load('best_rf_model.pkl')
        with open('feature_encoders.pkl', 'rb') as f:
            pickle.load(f)
        with open('target_encoder.pkl', 'rb') as f:
            pickle.load(f)
        with open('feature_columns.pkl', 'rb') as f:
            pickle.load(f)
        st.info("Pre-trained model files found. Skipping training.")
        return
    except (FileNotFoundError, EOFError):
        st.warning("Pre-trained model files not found. Training the model now...")
    
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
    
    # Save the model and encoders
    with open('feature_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)
    with open('target_encoder.pkl', 'wb') as f:
        pickle.dump(target_encoder, f)
    with open('feature_columns.pkl', 'wb') as f:
        pickle.dump(X.columns.tolist(), f)
    joblib.dump(best_rf_model, 'best_rf_model.pkl')
    
    st.success("Model trained and saved successfully!")

def run_diagnostic_test(topic_name):
    """Simulates a diagnostic test and returns results."""
    df = load_data(CSV_LINKS[topic_name])
    
    diagnostic_questions = pd.DataFrame()
    if 'Type' in df.columns:
        diagnostic_questions = df[df['Type'].str.lower() == "diagnostic"]
    elif 'type' in df.columns:
        diagnostic_questions = df[df['type'].str.lower() == "diagnostic"]

    diagnostic_questions = diagnostic_questions.head(10)
    
    score = 0
    total_marks = 0
    wrong_answers = []
    
    start_time = time.time()
    
    st.subheader("Diagnostic Test Questions")
    
    if not diagnostic_questions.empty:
        for q_num, (_, row) in enumerate(diagnostic_questions.iterrows(), start=1):
            difficulty_col = 'DifficultyLevel' if 'DifficultyLevel' in diagnostic_questions.columns else 'difficulty'
            question_col = 'QuestionText' if 'QuestionText' in diagnostic_questions.columns else 'question_text'
            correct_answer_col = 'CorrectAnswer' if 'CorrectAnswer' in diagnostic_questions.columns else 'Correct_Option'

            if difficulty_col and question_col and correct_answer_col:
                marks_for_q = int(row.get('Marks', 4))
                total_marks += marks_for_q
                
                options = {
                    'A': row.get('OptionA'), 
                    'B': row.get('OptionB'), 
                    'C': row.get('OptionC'), 
                    'D': row.get('OptionD')
                }
                
                correct_answer_raw = str(row[correct_answer_col]).strip()
                # Normalize correct answer to be a single letter
                if correct_answer_raw.upper() not in ["A", "B", "C", "D"]:
                    for letter, opt_col in zip(["A", "B", "C", "D"], ["OptionA", "OptionB", "OptionC", "OptionD"]):
                        if opt_col in row and str(row[opt_col]).strip().lower() == correct_answer_raw.lower():
                            correct_answer = letter
                            break
                    else: # If loop completes without finding a match, use the raw answer
                        correct_answer = correct_answer_raw
                else:
                    correct_answer = correct_answer_raw

                st.markdown(f"**Q{q_num}** [{row[difficulty_col]}] {row[question_col]}")
                
                user_answer = st.radio("Your answer:", options=list(options.keys()), key=f"q{q_num}")

                if user_answer.upper() == correct_answer.upper():
                    score += marks_for_q
                else:
                    wrong_answers.append({
                        "Q_No": q_num,
                        "Your_Answer": user_answer.upper(),
                        "Correct_Answer": correct_answer.upper()
                    })
    
    end_time = time.time()
    total_time = round(end_time - start_time, 2)
    total_questions = len(diagnostic_questions)
    
    return score, total_time, wrong_answers, total_marks, total_questions

# --- Main App Logic ---
def main():
    st.title("Drona AI: Personalized Learning Path Recommender")
    st.markdown("This app uses a machine learning model to recommend a personalized learning path based on your performance in a diagnostic test.")

    # State management for the app flow
    if 'student_name' not in st.session_state:
        st.session_state['student_name'] = ""
    if 'test_started' not in st.session_state:
        st.session_state['test_started'] = False
    if 'test_submitted' not in st.session_state:
        st.session_state['test_submitted'] = False

    # --- Step 1: Student Login ---
    if not st.session_state['student_name']:
        student_name_input = st.text_input("Enter your name:")
        if student_name_input:
            st.session_state['student_name'] = student_name_input
            st.success(f"Welcome, {st.session_state['student_name']}! 👋")
            st.rerun()
    else:
        st.sidebar.info(f"Welcome, {st.session_state['student_name']}!")
        
    if not st.session_state['test_started']:
        # --- Step 2: Topic Selection ---
        st.subheader("Choose a topic:")
        topics_display = [t.replace('_', ' ').title() for t in list(CSV_LINKS.keys())[1:]]
        selected_topic_display = st.selectbox("Select a topic:", topics_display)
        
        if st.button("Start Diagnostic Test"):
            st.session_state['selected_topic'] = selected_topic_display.replace(' ', '_').lower()
            st.session_state['test_started'] = True
            st.rerun()

    if st.session_state['test_started'] and not st.session_state['test_submitted']:
        score, total_time, wrong_answers, total_marks, total_questions = run_diagnostic_test(st.session_state['selected_topic'])

        if st.button("Submit Test"):
            st.session_state['score'] = score
            st.session_state['total_time'] = total_time
            st.session_state['wrong_answers'] = wrong_answers
            st.session_state['total_marks'] = total_marks
            st.session_state['total_questions'] = total_questions
            st.session_state['test_submitted'] = True
            st.rerun()

    if st.session_state['test_submitted']:
        st.subheader("Diagnostic Test Results")
        st.markdown(f"📊 **Score:** {st.session_state['score']}/{st.session_state['total_marks']} marks")
        st.markdown(f"📝 **Total Questions:** {st.session_state['total_questions']}")
        st.markdown(f"⏳ **Time Taken:** {st.session_state['total_time']} seconds")
        
        if st.session_state['wrong_answers']:
            st.subheader("❌ Incorrect Answers:")
            for wa in st.session_state['wrong_answers']:
                st.markdown(f"Q{wa['Q_No']}: Your Answer = `{wa['Your_Answer']}` | Correct Answer = `{wa['Correct_Answer']}`")
        else:
            st.balloons()
            st.success("🎉 All answers correct! Great job!")

        # --- Step 3: Get Feedback ---
        st.subheader("Select the area where you faced difficulty:")
        feedback_options = ["Speed Test", "Level 1", "Level 2"]
        feedback_str = st.selectbox("Your choice:", feedback_options)
        
        if st.button("Get Recommendation"):
            st.session_state['feedback'] = feedback_str
            
            # --- Step 4: Prepare Data and Make Prediction ---
            try:
                # Load the pre-trained model and encoders
                best_rf_model = joblib.load('best_rf_model.pkl')
                with open('feature_encoders.pkl', 'rb') as f:
                    label_encoders = pickle.load(f)
                with open('target_encoder.pkl', 'rb') as f:
                    target_encoder = pickle.load(f)
                with open('feature_columns.pkl', 'rb') as f:
                    feature_order = pickle.load(f)

                row = {
                    "Topic": st.session_state['selected_topic'],
                    "Score": st.session_state['score'],
                    "Time Taken (seconds)": st.session_state['total_time'],
                    "Feedback": st.session_state['feedback'],
                    "Attempts": 1
                }
                df_infer = pd.DataFrame([row])

                # Preprocessing and Feature Engineering
                df_infer_processed = df_infer.copy()
                df_infer_processed["Topic"] = df_infer_processed["Topic"].str.lower().str.replace(" ", "").str.replace("and", "").str.replace("_", "")
                
                for col, encoder in label_encoders.items():
                    if col in df_infer_processed.columns:
                        try:
                            df_infer_processed[col] = encoder.transform(df_infer_processed[col])
                        except ValueError:
                            # Handle unseen labels by mapping them to a known category or default
                            st.warning(f"Unseen label in column '{col}'. Using a default value.")
                            df_infer_processed[col] = -1 # A generic default that may need more sophisticated handling in a real app

                df_infer_processed["Efficiency_Score"] = df_infer_processed["Score"] / (df_infer_processed["Time Taken (seconds)"] + 1)
                df_infer_processed["Norm_Time"] = df_infer_processed["Time Taken (seconds)"] / (df_infer_processed["Attempts"] + 1)
                df_infer_processed["Accuracy_per_Attempt"] = df_infer_processed["Score"] / (df_infer_processed["Attempts"] + 1)
                df_infer_processed["Time_per_Point"] = df_infer_processed["Time Taken (seconds)"] / df_infer_processed["Score"].replace(0, np.nan)
                df_infer_processed["Time_per_Point"] = df_infer_processed["Time_per_Point"].fillna(df_infer_processed["Time Taken (seconds)"])
                
                df_infer_processed = df_infer_processed[feature_order]

                # Make Prediction
                predicted_path_encoded = best_rf_model.predict(df_infer_processed)
                predicted_path = target_encoder.inverse_transform(predicted_path_encoded)

                # --- Step 5: Display Recommendation ---
                st.subheader("Your Recommended Learning Path:")
                st.success(predicted_path[0])

            except FileNotFoundError:
                st.error("Model files not found. Please ensure the model and encoders are saved in the repository.")
                
            st.session_state['show_recommendation'] = True
    
    if st.button("Start Over"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()

if __name__ == "__main__":
    main()
