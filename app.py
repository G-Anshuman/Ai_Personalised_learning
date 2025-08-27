import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle
import joblib
import os
import matplotlib.pyplot as plt
from streamlit_gsheets import GSheetsConnection

# --- Google Sheets Configuration ---
# You need to replace 'your_sheet_id' with your Google Sheet ID
SHEET_URL = f"https://docs.google.com/spreadsheets/d/{st.secrets['gcp_sheet_id']}/edit?usp=sharing"

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
    Trains the Random Forest model and caches the necessary objects.
    This function is cached to run only once.
    """
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
    
    return best_rf_model, label_encoders, target_encoder, X.columns.tolist()

# --- Load the model and encoders once at the start of the app ---
best_rf_model, label_encoders, target_encoder, feature_order = get_trained_model()


@st.cache_data(ttl=60) # Cache for 60 seconds to prevent rate limiting
def get_logged_in_users():
    """Reads student data from the Google Sheet."""
    conn = st.connection("gsheets", type=GSheetsConnection)
    df = conn.read(spreadsheet=SHEET_URL, usecols=list(range(6)), ttl=5)
    df.columns = ["student_name", "dob", "topic", "score", "time_taken", "recommended_path"]
    df.dropna(how="all", inplace=True)
    return df

def save_student_data(student_name, dob, topic, score, time_taken, recommended_path):
    """Saves student data to the Google Sheet."""
    conn = st.connection("gsheets", type=GSheetsConnection)
    new_data = pd.DataFrame([{
        "student_name": student_name,
        "dob": dob,
        "topic": topic,
        "score": score,
        "time_taken": time_taken,
        "recommended_path": recommended_path,
    }])
    conn.append(spreadsheet=SHEET_URL, data=new_data)
    st.info("Your results have been saved!")

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


def show_welcome_screen():
    st.title("Drona AI: Personalized Learning Path Recommender")
    st.header("The only Quantitative Aptitude teacher who teaches at your pace of learning.")
    st.markdown("---")
    
    st.write(
        "This app recommends the best learning path you need to follow for improving areas where you face difficulty. "
        "It starts with a diagnostic test with 3 levels of questions: Level 1, Level 2, and Speed Test. "
        "Based on the questions you find most difficult, along with your feedback, "
        "Drona AI will recommend a personalized learning path tailored just for you."
    )
    
    st.markdown("---")
    
    if st.button("Start Your Journey ✨", use_container_width=True):
        st.session_state['welcome_complete'] = True
        st.rerun()

def show_student_dashboard(student_data):
    st.title(f"Welcome back, {student_data['student_name'].iloc[0]}!")
    st.subheader("Your Last Test Performance")
    
    last_topic = student_data['topic'].iloc[0]
    last_score = student_data['score'].iloc[0]
    last_time = student_data['time_taken'].iloc[0]
    last_recommended_path = student_data['recommended_path'].iloc[0]

    st.markdown(f"**Topic:** {last_topic.replace('_', ' ').title()}")
    st.markdown(f"**Score:** {last_score}")
    st.markdown(f"**Time Taken:** {last_time} seconds")
    st.markdown(f"**Recommended Path:** {last_recommended_path}")

    total_questions = 10 
    correct_count = last_score // 4 
    incorrect_count = total_questions - correct_count

    result_df = pd.DataFrame({
        'Category': ['Correct', 'Incorrect'],
        'Count': [correct_count, incorrect_count]
    })
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(result_df['Category'], result_df['Count'], color=['green', 'red'])
    ax.set_title('Last Test Performance Summary')
    ax.set_xlabel('Question Status')
    ax.set_ylabel('Number of Questions')
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("Start a New Quiz")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Choose a New Topic"):
            st.session_state.test_started = False
            st.session_state.test_submitted = False
            st.session_state.current_question_index = 0
            st.session_state.answers = []
            st.session_state.is_returning_user = False
            st.rerun()
    with col2:
        if st.button("Re-take this Topic"):
            st.session_state.test_started = True
            st.session_state.test_submitted = False
            st.session_state.current_question_index = 0
            st.session_state.answers = []
            st.session_state.start_time = time.time()
            st.session_state.diagnostic_questions = get_diagnostic_questions(last_topic)
            st.session_state.is_returning_user = False
            st.rerun()

def show_main_app_flow():
    if 'is_returning_user' not in st.session_state:
        st.session_state.is_returning_user = None

    if st.session_state.is_returning_user is None:
        st.title("Ready to Start?")
        col_name, col_dob = st.columns(2)
        with col_name:
            student_name_input = st.text_input("Enter your name:")
        with col_dob:
            student_dob_input = st.text_input("Enter your D.O.B. (YYYY-MM-DD):")

        if st.button("Log In"):
            if student_name_input and student_dob_input:
                try:
                    logged_in_users_df = get_logged_in_users()
                    existing_student_data = logged_in_users_df[
                        (logged_in_users_df['student_name'].str.lower() == student_name_input.lower()) &
                        (logged_in_users_df['dob'] == student_dob_input)
                    ]
                    
                    if not existing_student_data.empty:
                        st.session_state.is_returning_user = True
                        st.session_state.student_name = existing_student_data['student_name'].iloc[0]
                        st.session_state.student_data = existing_student_data
                        st.rerun()
                    else:
                        st.session_state.is_returning_user = False
                        st.session_state.student_name = student_name_input
                        st.session_state.student_dob = student_dob_input
                        st.success(f"Welcome, {st.session_state.student_name}! 👋")
                        st.rerun()
                except Exception as e:
                    st.error(f"An error occurred: {e}. Please check your Google Sheets connection settings.")
            else:
                st.warning("Please enter both your name and D.O.B.")
        return

    if st.session_state.is_returning_user:
        show_student_dashboard(st.session_state.student_data)
        return

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

    if st.session_state.test_started and not st.session_state.test_submitted:
        questions = st.session_state.diagnostic_questions
        total_questions = len(questions)

        progress_percentage = (st.session_state.current_question_index) / total_questions
        st.progress(progress_percentage)

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

            button_text = "Submit Assessment" if q_num == total_questions else "Next Question"

            if st.button(button_text):
                selected_answer_letter = user_choice.split('.')[0].strip()
                st.session_state.answers.append(selected_answer_letter)
                st.session_state.current_question_index += 1
                st.rerun()
        else:
            st.session_state.test_submitted = True
            st.rerun()

    if st.session_state.test_submitted:
        st.progress(1.0)
        st.header("Test Completed!")
        
        end_time = time.time()
        total_time = round(end_time - st.session_state.start_time, 2)

        score = 0
        wrong_answers = []
        total_marks = 0
        questions = st.session_state.diagnostic_questions

        for i, user_ans_letter in enumerate(st.session_state.answers):
            q_data = questions[i]
            correct_answer_raw = str(q_data.get('CorrectAnswer') or q_data.get('Correct_Option')).strip()
            marks = int(q_data.get('Marks', 4))
            total_marks += marks

            correct_opt_letter = None
            if correct_answer_raw.upper() in ["A", "B", "C", "D"]:
                correct_opt_letter = correct_answer_raw
            else:
                for letter, opt_col in zip(["A", "B", "C", "D"], ["OptionA", "OptionB", "OptionC", "OptionD"]):
                    if str(q_data.get(opt_col)).strip().lower() == correct_answer_raw.lower():
                        correct_opt_letter = letter
                        break

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
        
        correct_count = len(questions) - len(wrong_answers)
        incorrect_count = len(wrong_answers)

        result_df = pd.DataFrame({
            'Category': ['Correct', 'Incorrect'],
            'Count': [correct_count, incorrect_count]
        })

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(result_df['Category'], result_df['Count'], color=['green', 'red'])
        ax.set_title('Test Performance Summary')
        ax.set_xlabel('Question Status')
        ax.set_ylabel('Number of Questions')
        st.pyplot(fig)
        
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
                recommended_path_str = predicted_path[0]

                st.subheader("Your Recommended Learning Path:")
                st.success(recommended_path_str)

                save_student_data(st.session_state.student_name, st.session_state.student_dob, st.session_state.selected_topic, score, total_time, recommended_path_str)

        if st.button("Start Over"):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()

def main():
    if 'welcome_complete' not in st.session_state:
        st.session_state.welcome_complete = False
    
    if st.session_state.welcome_complete:
        show_main_app_flow()
    else:
        show_welcome_screen()

if __name__ == "__main__":
    main()
