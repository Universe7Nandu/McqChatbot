import streamlit as st
import os
import json
import ast
from datetime import datetime
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space

# Load environment variables
load_dotenv()

# Initialize session state variables
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    st.session_state.questions = []
    st.session_state.current_question = 0
    st.session_state.score = 0
    st.session_state.answers = []
    st.session_state.total = 0
    st.session_state.once = True
    st.session_state.done = False
    st.session_state.topic = ""
    st.session_state.difficulty_level = "Medium"
    st.session_state.num_questions = 5
    st.session_state.user_data = []
    st.session_state.explanations = []
    st.session_state.feedback = []

# Initialize Groq AI model
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    groq_api_key = st.secrets["GROQ_API_KEY"] if "GROQ_API_KEY" in st.secrets else None

if groq_api_key:
    llm = ChatGroq(
        model_name="llama3-70b-8192",
        temperature=0.5,
        groq_api_key=groq_api_key
    )
else:
    st.error("Groq API key not found. Please set it in the .env file or Streamlit secrets.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Adaptive MCQ Generator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom styling for a more attractive UI
st.markdown("""
<style>
    /* Main page styling */
    .main {
        background-color: #f8f9fa;
        color: #333333;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* App container */
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    
    # p{
    # color: mediumvioletred;
    # }
    
    /* Question card styling */
    .question-card {
        background-color: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
        margin-bottom: 25px;
        border-left: 5px solid #6c5ce7;
        transition: transform 0.2s ease;
    }
    .question-card:hover {
        transform: translateY(-3px);
    }
    
    /* Option buttons styling */
    .option-button {
        width: 100%;
        text-align: left;
        margin: 8px 0;
        padding: 12px 18px;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        background-color: white;
        color:black;
        transition: all 0.2s ease;
        font-size: 16px;
        cursor: pointer;
    }
    .option-button:hover {
        background-color: #f0f2f6;
        border-color: #6c5ce7;
    }
    
    /* Answer feedback styling */
    .selected-correct {
        background-color: #d4edda;
        border-color: #28a745;
        border-left: 5px solid #28a745;
    }
    .selected-incorrect {
        background-color: #f8d7da;
        border-color: #dc3545;
        border-left: 5px solid #dc3545;
    }
    .correct-answer {
        background-color: #d4edda;
        border-color: #28a745;
        border-left: 5px solid #28a745;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s ease;
        border: none;
        padding: 10px 24px;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Explanation box styling */
    .explanation-box {
        background-color: #f8f9fa;
        color:black;
        padding: 18px;
        border-radius: 8px;
        margin-top: 18px;
        border-left: 3px solid #6c5ce7;
    }
    
    /* Title styling */
    .title-container {
        text-align: center;
        padding: 30px 0;
        background: linear-gradient(135deg, #6c5ce7, #a29bfe);
        border-radius: 12px;
        margin-bottom: 25px;
        color: white;
    }
    .title-text {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    .subtitle-text {
        font-size: 1.1rem;
        opacity: 0.9;
        margin-top: 8px;
    }
    
    /* Analytics card styling */
    .analytics-card {
        background-color: white;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
        margin-bottom: 25px;
        border-top: 4px solid #6c5ce7;
    }
    
    /* Metric styling */
    .metric-card {
        background: linear-gradient(135deg, #6c5ce7, #a29bfe);
        padding: 20px;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin: 10px 0;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* Progress bar styling */
    .stProgress > div > div {
        background-color: #6c5ce7;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    
    /* Widget labels */
    .stTextInput > label, .stSelectbox > label, .stNumberInput > label {
        font-weight: 500;
        font-size: 16px;
        color: #333;
    }
    
    /* Remove fullscreen button */
    .modebar-container {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# Remove old UI section and replace with only empty title placeholder
# We'll use the main() function for the actual UI
st.markdown("<div id='title-placeholder'></div>", unsafe_allow_html=True)

# Function to generate MCQs based on the topic and difficulty
def generate_mcqs(topic, difficulty_level, num_questions, performance_history=None):
    """
    Generate MCQs using Groq LLM based on topic, difficulty, and past performance
    """
    
    # Create a system prompt based on user performance and requirements
    performance_context = ""
    if performance_history:
        # Format performance data for the AI to understand patterns
        performance_context = f"User performance history: {performance_history}\n"
        
    system_prompt = f"""
    You are an expert educational assessment generator specialized in creating high-quality multiple-choice questions (MCQs) for adaptive learning systems.

    TOPIC: {topic}
    DIFFICULTY: {difficulty_level}
    NUMBER OF QUESTIONS: {num_questions}
    {performance_context}

    Create exactly {num_questions} multiple-choice questions on the topic "{topic}" with {difficulty_level} difficulty.
    
    Follow these requirements strictly:
    1. Each question must be clear, concise, and directly relevant to the topic
    2. Match the difficulty level accurately: 
       - Easy: Basic understanding and recall questions
       - Medium: Application and comprehension questions
       - Hard: Analysis and evaluation questions
    3. Provide exactly 4 answer choices labeled A, B, C, D for each question
    4. Only ONE answer should be correct
    5. The other answers must be plausible distractors that seem reasonable but are incorrect
    6. Include a detailed explanation that teaches why the correct answer is right
    
    Format your response as a valid Python list of lists ONLY, where each inner list contains EXACTLY:
    [question_text, option_A, option_B, option_C, option_D, correct_answer_letter, explanation]

    Example format:
    [
        ["What is the capital of France?", "London", "Berlin", "Paris", "Madrid", "C", "Paris is the capital city of France."],
        ["Which planet is closest to the sun?", "Earth", "Mercury", "Venus", "Mars", "B", "Mercury is the closest planet to the sun in our solar system."]
    ]

    The output MUST be a valid Python list that can be parsed with ast.literal_eval() - nothing else.
    DO NOT include any text, explanations, or markdown formatting before or after the list.
    """

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Generate {num_questions} multiple-choice questions about {topic} with {difficulty_level} difficulty level.")
    ]

    try:
        response = llm.invoke(messages)
        # Clean the response to ensure it's a valid Python list
        content = response.content.strip()
        
        # Handle different response formats
        if content.startswith("```python") and content.endswith("```"):
            content = content[content.find("["):content.rfind("]")+1]
        elif content.startswith("```") and content.endswith("```"):
            content = content[content.find("["):content.rfind("]")+1]
        elif content.startswith("[") and content.endswith("]"):
            content = content
        else:
            # Try to extract the list if it's embedded in text
            start_idx = content.find("[")
            end_idx = content.rfind("]")
            if start_idx != -1 and end_idx != -1:
                content = content[start_idx:end_idx+1]
            
        # Parse the list
        questions = ast.literal_eval(content)
        
        # Ensure we have the right number of questions
        questions = questions[:num_questions]
        
        # Validate question format
        for q in questions:
            if len(q) != 7:
                st.error(f"Question format is incorrect: {q}")
                return []
                
        return questions
    except Exception as e:
        st.error(f"Error generating questions: {str(e)}")
        st.error("Please try again with a different topic or check your API key.")
        return []

# Function to determine next question difficulty based on performance
def determine_difficulty(current_difficulty, correct_answers, total_questions):
    """
    Adapt difficulty based on user performance
    """
    if total_questions == 0:
        return "Medium"
        
    accuracy = correct_answers / total_questions
    
    if current_difficulty == "Easy":
        if accuracy > 0.7:
            return "Medium"
        else:
            return "Easy"
    elif current_difficulty == "Medium":
        if accuracy > 0.7:
            return "Hard"
        elif accuracy < 0.4:
            return "Easy"
        else:
            return "Medium"
    else:  # Hard
        if accuracy < 0.5:
            return "Medium"
        else:
            return "Hard"

# Function to save user performance data
def save_performance_data(topic, score, total, difficulty, questions, answers):
    """
    Save the user's performance data for analytics
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Calculate metrics for each question
    question_metrics = []
    for i, (question, user_answer) in enumerate(zip(questions, answers)):
        is_correct = user_answer == question[5]  # Index 5 contains the correct answer
        question_metrics.append({
            "question_number": i + 1,
            "question_text": question[0],
            "correct_answer": question[5],
            "user_answer": user_answer,
            "is_correct": is_correct
        })
    
    performance_data = {
        "timestamp": timestamp,
        "topic": topic,
        "difficulty": difficulty,
        "score": score,
        "total": total,
        "accuracy": score / total if total > 0 else 0,
        "question_details": question_metrics
    }
    
    # Append to session state user data
    st.session_state.user_data.append(performance_data)
    
    # Return a summary for adaptive difficulty
    return {
        "topic": topic,
        "accuracy": performance_data["accuracy"],
        "difficulty": difficulty
    }

# Function to display analytics
def display_analytics():
    """
    Display user performance analytics with enhanced visuals
    """
    if not st.session_state.user_data:
        st.info("📊 No performance data available yet. Complete a quiz to see analytics.")
        
        # Show sample analytics UI
        st.markdown(
            """
            <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin: 20px 0; border-left: 4px solid #6c5ce7;'>
                <h4 style='color: #6c5ce7; margin-top: 0;'>Analytics Preview</h4>
                <p style='color:mediumvioletred;'>After completing quizzes, you'll see detailed analytics including:</p>
                <ul style='color: mediumvioletred;'>
                    <li>Performance trends over time</li>
                    <li>Topic strengths and weaknesses</li>
                    <li>Difficulty progression</li>
                    <li>Question-level analysis</li>
                </ul>
                <p>Take a quiz to start building your personalized learning profile!</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        return
    
    # Create dataframes for analysis
    df = pd.DataFrame([
        {
            "Topic": data["topic"],
            "Difficulty": data["difficulty"],
            "Score": data["score"],
            "Total": data["total"],
            "Accuracy": data["accuracy"],
            "Timestamp": data["timestamp"]
        } for data in st.session_state.user_data
    ])
    
    # Overall stats with card styling
    st.markdown("<h3 style='color: #6c5ce7; margin-bottom: 20px;'>📈 Overall Performance</h3>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-label'>Total Quizzes</div>
                <div class='metric-value'>{len(df)}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-label'>Average Accuracy</div>
                <div class='metric-value'>{df['Accuracy'].mean():.0%}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-label'>Topics Covered</div>
                <div class='metric-value'>{df['Topic'].nunique()}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Topic performance
    st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: #6c5ce7; margin-bottom: 20px;'>📚 Performance by Topic</h3>", unsafe_allow_html=True)
    
    # Create an analytics card for topic performance
    st.markdown(
        """
        <div class='analytics-card'>
            <p style='color: #666; margin-bottom: 20px;'>
                This chart shows your average accuracy for each topic you've studied.
                Higher bars indicate better understanding of those topics.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    topic_df = df.groupby("Topic").agg({
        "Accuracy": "mean",
        "Score": "sum",
        "Total": "sum"
    }).reset_index()
    
    fig = px.bar(
        topic_df,
        x="Topic",
        y="Accuracy",
        color="Accuracy",
        text_auto=".0%",
        title="Average Accuracy by Topic",
        color_continuous_scale="Viridis",
        height=400
    )
    
    fig.update_layout(
        yaxis_tickformat=".0%",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Segoe UI, Arial, sans-serif", size=12),
        margin=dict(l=20, r=20, t=40, b=20),
        yaxis=dict(
            title="Average Accuracy",
            gridcolor="#eaeaea",
            zerolinecolor="#eaeaea",
        ),
        xaxis=dict(
            title="",
            gridcolor="#eaeaea",
        ),
        coloraxis_colorbar=dict(
            title="Accuracy",
            tickformat=".0%",
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show detailed topic statistics
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    with st.expander("View Detailed Topic Statistics"):
        # Create a formatted table
        for i, row in topic_df.iterrows():
            topic_color = "#6c5ce7" if row["Accuracy"] >= 0.7 else "#e74c3c" if row["Accuracy"] < 0.5 else "#3498db"
            
            st.markdown(
                f"""
                <div style='padding: 15px; margin-bottom: 10px; border-radius: 8px; border-left: 4px solid {topic_color}; background-color: white;'>
                    <h4 style='margin: 0 0 10px 0; color: {topic_color};'>{row["Topic"]}</h4>
                    <div style='display: flex; justify-content: space-between;'>
                        <div>
                            <strong>Questions Answered:</strong> {int(row["Total"])}
                        </div>
                        <div>
                            <strong>Correct Answers:</strong> {int(row["Score"])}
                        </div>
                        <div>
                            <strong>Accuracy:</strong> {row["Accuracy"]:.0%}
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    # Difficulty progression
    if len(df) > 1:
        st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
        st.markdown("<h3 style='color: #6c5ce7; margin-bottom: 20px;'>🔄 Learning Progression</h3>", unsafe_allow_html=True)
        
        # Create an analytics card for learning progression
        st.markdown(
            """
            <div class='analytics-card'>
                <p style='color: #666; margin-bottom: 20px;'>
                    This chart shows how your performance has changed over time.
                    Upward trends indicate improvement in your understanding.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        fig = px.line(
            df,
            x="Timestamp",
            y="Accuracy",
            color="Topic",
            markers=True,
            title="Accuracy Over Time",
            height=400
        )
        
        fig.update_layout(
            yaxis_tickformat=".0%",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Segoe UI, Arial, sans-serif", size=12),
            margin=dict(l=20, r=20, t=40, b=20),
            yaxis=dict(
                title="Accuracy",
                gridcolor="#eaeaea",
                zerolinecolor="#eaeaea",
            ),
            xaxis=dict(
                title="",
                gridcolor="#eaeaea",
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add difficulty progression visualization
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        
        # Create difficulty level counts
        difficulty_counts = df.groupby(["Topic", "Difficulty"]).size().reset_index(name="Count")
        
        fig = px.bar(
            difficulty_counts,
            x="Topic",
            y="Count",
            color="Difficulty",
            title="Difficulty Levels by Topic",
            barmode="stack",
            color_discrete_map={"Easy": "#55efc4", "Medium": "#74b9ff", "Hard": "#a29bfe"},
            height=350
        )
        
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Segoe UI, Arial, sans-serif", size=12),
            margin=dict(l=20, r=20, t=40, b=20),
            yaxis=dict(
                title="Number of Quizzes",
                gridcolor="#eaeaea",
                zerolinecolor="#eaeaea",
            ),
            xaxis=dict(
                title="",
                gridcolor="#eaeaea",
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Question-level analysis (most recent quiz)
    if st.session_state.user_data:
        st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
        st.markdown("<h3 style='color: #6c5ce7; margin-bottom: 20px;'>🔍 Recent Quiz Analysis</h3>", unsafe_allow_html=True)
        
        # Create an analytics card
        last_quiz = st.session_state.user_data[-1]
        questions_df = pd.DataFrame(last_quiz["question_details"])
        
        # Add a summary of the most recent quiz
        st.markdown(
            f"""
            <div class='analytics-card'>
                <h4 style='color: #6c5ce7; margin: 0 0 15px 0;'>Latest Quiz: {last_quiz['topic']} ({last_quiz['difficulty']})</h4>
                <div style='display: flex; justify-content: space-between; margin-bottom: 20px;'>
                    <div style='text-align: center; padding: 10px 15px; background-color: #f8f9fa; border-radius: 8px;'>
                        <div style='font-size: 16px;'>Score</div>
                        <div style='font-size: 24px; font-weight: 500;'>{last_quiz['score']} / {last_quiz['total']}</div>
                    </div>
                    <div style='text-align: center; padding: 10px 15px; background-color: #f8f9fa; border-radius: 8px;'>
                        <div style='font-size: 16px;'>Accuracy</div>
                        <div style='font-size: 24px; font-weight: 500;'>{last_quiz['accuracy']:.0%}</div>
                    </div>
                    <div style='text-align: center; padding: 10px 15px; background-color: #f8f9fa; border-radius: 8px;'>
                        <div style='font-size: 16px;'>Date</div>
                        <div style='font-size: 18px;'>{last_quiz['timestamp'].split()[0]}</div>
                    </div>
                </div>
                <p style='color: #666;'>
                    This chart shows your performance on each question in your most recent quiz.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        fig = px.bar(
            questions_df,
            x="question_number",
            y="is_correct",
            color="is_correct",
            title=f"Question Performance - {last_quiz['topic']} ({last_quiz['difficulty']})",
            labels={"question_number": "Question Number", "is_correct": "Correct"},
            color_discrete_map={True: "#55efc4", False: "#ff7675"},
            height=350
        )
        
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Segoe UI, Arial, sans-serif", size=12),
            margin=dict(l=20, r=20, t=40, b=20),
            yaxis=dict(
                title="Result",
                tickvals=[0, 1],
                ticktext=["Incorrect", "Correct"],
                gridcolor="#eaeaea",
                zerolinecolor="#eaeaea",
            ),
            xaxis=dict(
                title="Question Number",
                gridcolor="#eaeaea",
            ),
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display strengths and weaknesses
        correct_count = sum(questions_df["is_correct"])
        incorrect_count = len(questions_df) - correct_count
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(
                """
                <div style='background-color: #d4edda; padding: 15px; border-radius: 8px; height: 100%;'>
                    <h4 style='color: #28a745; margin-top: 0;'>💪 Strengths</h4>
                """,
                unsafe_allow_html=True
            )
            
            if correct_count > 0:
                correct_questions = questions_df[questions_df["is_correct"]]
                for i, row in correct_questions.iterrows():
                    st.markdown(f"✓ {row['question_text'][:80]}...")
            else:
                st.write("No correct answers in this quiz.")
        
        with col2:
            st.markdown(
                """
                <div style='background-color: #f8d7da; padding: 15px; border-radius: 8px; height: 100%;'>
                    <h4 style='color: #dc3545; margin-top: 0;'>🎯 Areas to Improve</h4>
                """,
                unsafe_allow_html=True
            )
            
            if incorrect_count > 0:
                incorrect_questions = questions_df[~questions_df["is_correct"]]
                for i, row in incorrect_questions.iterrows():
                    st.markdown(f"✗ {row['question_text'][:80]}...")
            else:
                st.write("Perfect score! No areas to improve.")
        
        # Recommendations based on performance
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin: 20px 0; border-left: 5px solid #6c5ce7;'>
                <h4 style='color: #6c5ce7; margin-top: 0;'>🎓 Learning Recommendations</h4>
                <p style='color: #6c5ce7';>Based on your performance in the "{last_quiz['topic']}" quiz:</p>
                <ul>
                    {"<li>Consider moving to a <strong>harder difficulty</strong> for this topic.</li>" if last_quiz['accuracy'] > 0.8 else ""}
                    {"<li>You're doing well! Continue practicing at the current difficulty.</li>" if 0.6 <= last_quiz['accuracy'] <= 0.8 else ""}
                    {"<li>Try studying this topic more or attempt an <strong>easier difficulty</strong>.</li>" if last_quiz['accuracy'] < 0.6 else ""}
                    {"<li>Focus on reviewing the questions you answered incorrectly.</li>" if incorrect_count > 0 else ""}
                    {"<li>Try a new related topic to expand your knowledge.</li>" if last_quiz['accuracy'] > 0.7 else ""}
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

# Main application UI
def main():
    """
    Main Streamlit application
    """
    # Sidebar navigation and information
    with st.sidebar:
        st.image("mcqimage.jpg", width=80)
        st.markdown("<h1 style='text-align: center; color: #6c5ce7;'>Adaptive MCQ Generator</h1>", unsafe_allow_html=True)
        
        st.divider()
        
        # Navigation
        page = st.radio("📋 Navigation", ["🧠 Generate MCQs", "📊 Analytics"], index=0)
        
        st.divider()
        
        # About section
        st.markdown("<h3 style='color: #6c5ce7;'>📌 About</h3>", unsafe_allow_html=True)
        st.info(
            "This intelligent MCQ generator creates adaptive assessment questions that adjust difficulty based on your performance. "
            "Powered by Groq LLM, it generates high-quality questions for educational assessment."
        )
        
        # Usage tips
        st.markdown("<h3 style='color: #6c5ce7;'>💡 Tips</h3>", unsafe_allow_html=True)
        st.success(
            "✓ Enter any educational topic\n"
            "✓ Start with Medium difficulty\n"
            "✓ Answer questions to build your learning profile\n"
            "✓ The system adapts to your performance"
        )
        
        st.divider()
        
        # GitHub link
        st.markdown("<h3 style='color: #6c5ce7;'>🔗 Links</h3>", unsafe_allow_html=True)
        st.markdown("[GitHub Repository](https://github.com/Universe7Nandu/McqChatbot)")
        st.markdown("[Report Issues](https://github.com/Universe7Nandu/McqChatbot/issues)")
    
    # Main content area
    if page == "🧠 Generate MCQs":
        # Title section with custom styling
        st.markdown(
            """
            <div class='title-container'>
                <h1 class='title-text'>🧠 Intelligent MCQ Generator</h1>
                <p class='subtitle-text'>Create adaptive assessment questions on any educational topic</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # Input form with improved styling
        with st.container():
            st.markdown("<h3 style='color: #6c5ce7;'>Enter Topic Details</h3>", unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                topic = st.text_input(
                    "Topic:",
                    key="topic_input",
                    value=st.session_state.topic,
                    placeholder="Enter any educational topic (e.g., Photosynthesis, World War II, Calculus...)"
                )
            with col2:
                difficulty = st.selectbox(
                    "Difficulty Level:",
                    ["Easy", "Medium", "Hard"],
                    index=["Easy", "Medium", "Hard"].index(st.session_state.difficulty_level)
                )
            with col3:
                num_questions = st.number_input(
                    "Number of Questions:",
                    min_value=3,
                    max_value=10,
                    value=st.session_state.num_questions
                )
        
        # Generate button with attractive styling
        generate_button = st.button(
            "🔮 Generate MCQs",
            key="generate_button",
            use_container_width=True,
            type="primary"
        )
        
        # Help text
        st.markdown(
            "<div style='text-align: center; color: #666; font-size: 14px; margin-top: 10px;'>"
            "The system will generate questions and adapt to your performance over time."
            "</div>",
            unsafe_allow_html=True
        )
        
        # Update session state
        st.session_state.topic = topic
        st.session_state.difficulty_level = difficulty
        st.session_state.num_questions = num_questions
        
        # Generate questions when button is clicked
        if generate_button and topic:
            with st.spinner(f"🔮 Generating questions about {topic}..."):
                # Determine appropriate difficulty based on past performance
                if st.session_state.user_data:
                    # Filter data for the same topic if it exists
                    topic_data = [d for d in st.session_state.user_data if d["topic"].lower() == topic.lower()]
                    if topic_data:
                        # Use the most recent performance data for this topic
                        recent_data = topic_data[-1]
                        adaptive_difficulty = determine_difficulty(
                            difficulty,
                            recent_data["score"],
                            recent_data["total"]
                        )
                        if adaptive_difficulty != difficulty:
                            st.info(f"🔄 Based on your previous performance, the difficulty has been adjusted to **{adaptive_difficulty}**.")
                            difficulty = adaptive_difficulty
                
                # Generate questions
                questions = generate_mcqs(topic, difficulty, num_questions, st.session_state.user_data)
                
                if questions:
                    # Reset quiz state
                    st.session_state.questions = questions
                    st.session_state.total = len(questions)
                    st.session_state.current_question = 0
                    st.session_state.score = 0
                    st.session_state.answers = []
                    st.session_state.explanations = []
                    st.session_state.feedback = []
                    st.session_state.once = False
                    st.session_state.done = False
                    
                    st.rerun()  # Refresh to show the first question
        
        # Display questions
        if not st.session_state.once and not st.session_state.done and st.session_state.questions:
            # Get current question data
            current_idx = st.session_state.current_question
            question_data = st.session_state.questions[current_idx]
            
            question = question_data[0]
            options = question_data[1:5]  # A, B, C, D
            correct_answer = question_data[5]
            explanation = question_data[6]
            
            # Progress indicator with custom styling
            st.markdown(
                f"""
                <div style='margin-bottom: 15px;'>
                    <p style='color: #6c5ce7; font-weight: 500; margin-bottom: 5px;'>
                        Question {current_idx + 1} of {st.session_state.total}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            progress = (current_idx + 1) / st.session_state.total
            st.progress(progress)
            
            # Question card with improved styling
            with st.container():
                st.markdown(
                    f"""
                    <div class='question-card'>
                        <h3 style='color: #333; margin-bottom: 20px;'>{question}</h3>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Answer options
                option_labels = ["A", "B", "C", "D"]
                selected_option = None
                
                # Check if this question has already been answered
                already_answered = len(st.session_state.answers) > current_idx
                
                st.markdown("<h4 style='color: #6c5ce7; margin-bottom: 15px;'>Select your answer:</h4>", unsafe_allow_html=True)
                
                # Create columns for better layout of options
                col1, col2 = st.columns(2)
                
                for i, (label, option) in enumerate(zip(option_labels, options)):
                    # Determine which column to use
                    col = col1 if i % 2 == 0 else col2
                    
                    # Determine button style based on whether the question was already answered
                    with col:
                        if already_answered:
                            user_answer = st.session_state.answers[current_idx]
                            is_correct_answer = label == correct_answer
                            is_user_selection = label == user_answer
                            
                            if is_user_selection and is_correct_answer:
                                button_style = "selected-correct"
                                icon = "✓"
                            elif is_user_selection and not is_correct_answer:
                                button_style = "selected-incorrect"
                                icon = "✗"
                            elif is_correct_answer:
                                button_style = "correct-answer"
                                icon = "✓"
                            else:
                                button_style = ""
                                icon = ""
                            
                            # Display the option as text with appropriate styling
                            st.markdown(
                                f"<div class='option-button {button_style}'><strong>{label}.</strong> {option} {icon}</div>",
                                unsafe_allow_html=True
                            )
                        else:
                            # Display clickable buttons for unanswered questions
                            if st.button(f"{label}. {option}", key=f"option_{current_idx}_{label}"):
                                selected_option = label
                
                # Process selected answer
                if selected_option:
                    # Record the answer
                    st.session_state.answers.append(selected_option)
                    
                    # Update score
                    if selected_option == correct_answer:
                        st.session_state.score += 1
                        st.session_state.feedback.append(True)
                    else:
                        st.session_state.feedback.append(False)
                    
                    # Save explanation
                    st.session_state.explanations.append(explanation)
                    
                    # Move to next question or end quiz
                    if current_idx < st.session_state.total - 1:
                        st.session_state.current_question += 1
                    else:
                        # Save performance data
                        save_performance_data(
                            topic,
                            st.session_state.score,
                            st.session_state.total,
                            difficulty,
                            st.session_state.questions,
                            st.session_state.answers
                        )
                        st.session_state.done = True
                    
                    st.rerun()  # Refresh to show next question or results
                
                # Show explanation if the question has been answered
                if already_answered:
                    with st.expander("📚 View Explanation", expanded=True):
                        st.markdown(
                            f"""
                            <div class='explanation-box'>
                                <h4 style='color: #6c5ce7; margin-top: 0;'>Explanation</h4>
                                <p>{st.session_state.explanations[current_idx]}</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    
                    # Navigation buttons with improved styling
                    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
                    col1, col2 = st.columns(2)
                    with col1:
                        if current_idx > 0:
                            if st.button("⬅️ Previous Question", use_container_width=True):
                                st.session_state.current_question -= 1
                                st.rerun()
                    with col2:
                        if current_idx < st.session_state.total - 1:
                            if st.button("Next Question ➡️", use_container_width=True):
                                st.session_state.current_question += 1
                                st.rerun()
                        elif not st.session_state.done:
                            if st.button("Finish Quiz 🏁", use_container_width=True):
                                # Save performance data if not already saved
                                save_performance_data(
                                    topic,
                                    st.session_state.score,
                                    st.session_state.total,
                                    difficulty,
                                    st.session_state.questions,
                                    st.session_state.answers
                                )
                                st.session_state.done = True
                                st.rerun()
        
        # Quiz results with attractive styling
        if st.session_state.done and st.session_state.questions:
            st.balloons()
            
            colored_header(
                label="🏆 Quiz Results",
                description=f"Topic: {st.session_state.topic} | Difficulty: {st.session_state.difficulty_level}",
                color_name="blue-70"
            )
            
            # Score display with improved visualization
            score_percentage = (st.session_state.score / st.session_state.total) * 100
            
            # Create score card
            st.markdown(
                f"""
                <div style='background: linear-gradient(135deg, #6c5ce7, #a29bfe); border-radius: 15px; padding: 20px; text-align: center; color: white; margin-bottom: 30px;'>
                    <h2 style='margin-bottom: 10px;'>Your Score</h2>
                    <div style='font-size: 48px; font-weight: 700; margin: 20px 0;'>
                        {st.session_state.score} / {st.session_state.total}
                    </div>
                    <div style='font-size: 24px; font-weight: 500;'>
                        {score_percentage:.1f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Performance gauge chart
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                fig, ax = plt.subplots(figsize=(8, 3))
                ax.set_xlim(0, 100)
                ax.set_ylim(0, 1)
                ax.set_yticks([])
                ax.set_xticks([0, 25, 50, 75, 100])
                ax.axvspan(0, 50, alpha=0.2, color='#ff7675')
                ax.axvspan(50, 75, alpha=0.2, color='#fdcb6e')
                ax.axvspan(75, 100, alpha=0.2, color='#55efc4')
                ax.axvline(score_percentage, color='#6c5ce7', linewidth=6)
                ax.set_title('Performance Meter', fontsize=16, color='#333')
                st.pyplot(fig)
            
            # Performance feedback with icons and styling
            if score_percentage >= 75:
                st.success("🏆 Excellent! You have a strong understanding of this topic. Keep up the great work!")
            elif score_percentage >= 50:
                st.info("👍 Good job! You have a decent grasp of the subject but there's room for improvement.")
            else:
                st.warning("💪 You might need more practice on this topic. Don't give up! Try again to improve your score.")
            
            # Learning recommendation based on performance
            st.markdown(
                f"""
                <div style='background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin: 20px 0; border-left: 5px solid #6c5ce7;'>
                    <h4 style='color: #6c5ce7; margin-top: 0;'>Learning Recommendation</h4>
                    <p>Based on your performance, you should consider:</p>
                    <ul>
                        {"<li>Reviewing the basics of this topic before attempting harder questions</li>" if score_percentage < 50 else ""}
                        {"<li>Focusing on the concepts you answered incorrectly</li>" if score_percentage < 100 else ""}
                        {"<li>Moving to a higher difficulty level</li>" if score_percentage >= 80 else ""}
                        {"<li>Practicing with more questions on this topic</li>"}
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Question review with improved styling
            st.subheader("📝 Question Review")
            for i, (question, user_answer, is_correct, explanation) in enumerate(
                zip(st.session_state.questions, st.session_state.answers, st.session_state.feedback, st.session_state.explanations)
            ):
                with st.expander(f"Question {i+1}: {question[0][:60]}...", expanded=False):
                    # Question card
                    st.markdown(
                        f"""
                        <div style='padding: 15px 0;'>
                            <h4 style='color: #333;'>{question[0]}</h4>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    # Display options with improved highlighting
                    for j, opt in enumerate(["A", "B", "C", "D"]):
                        if opt == question[5] and opt == user_answer:  # Correct and selected
                            st.markdown(
                                f"""
                                <div class='option-button selected-correct'>
                                    <strong>{opt}.</strong> {question[j+1]} ✓ <span style='color: green;'>(Your answer - Correct)</span>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        elif opt == question[5]:  # Correct but not selected
                            st.markdown(
                                f"""
                                <div class='option-button correct-answer'>
                                    <strong>{opt}.</strong> {question[j+1]} ✓ <span style='color: green;'>(Correct answer)</span>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        elif opt == user_answer:  # Incorrect and selected
                            st.markdown(
                                f"""
                                <div class='option-button selected-incorrect'>
                                    <strong>{opt}.</strong> {question[j+1]} ✗ <span style='color: red;'>(Your answer - Incorrect)</span>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                        else:  # Not selected
                            st.markdown(
                                f"""
                                <div class='option-button'>
                                    <strong>{opt}.</strong> {question[j+1]}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                    
                    # Explanation with improved styling
                    st.markdown(
                        f"""
                        <div class='explanation-box'>
                            <h4 style='color: #6c5ce7; margin-top: 0;'>Explanation</h4>
                            <p>{explanation}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            
            # Call to action buttons with improved styling
            st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🆕 Try Another Topic", use_container_width=True):
                    # Reset state for a new topic
                    st.session_state.once = True
                    st.session_state.done = False
                    st.session_state.topic = ""
                    st.rerun()
            with col2:
                if st.button("🔄 Retry This Topic", use_container_width=True):
                    # Keep the topic but reset other state for regenerating questions
                    st.session_state.once = True
                    st.session_state.done = False
                    st.rerun()
            
            # Add button to view analytics
            st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
            if st.button("📊 View Detailed Analytics", use_container_width=True, type="secondary"):
                # Switch to analytics page
                st.session_state.page = "Analytics"
                st.rerun()
                
    elif page == "📊 Analytics":
        # Analytics page with improved styling
        st.markdown(
            """
            <div class='title-container'>
                <h1 class='title-text'>📊 Performance Analytics</h1>
                <p class='subtitle-text'>Track your progress and identify areas for improvement</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # Display the analytics
        display_analytics()
        
        # Button to return to quiz generation
        st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
        if st.button("🧠 Back to Quiz Generator", use_container_width=True):
            st.session_state.page = "Generate MCQs"
            st.rerun()

# Run the application
if __name__ == "__main__":
    main()
