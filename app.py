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
from streamlit_lottie import st_lottie
import requests
from streamlit_card import card
from streamlit_extras.stylable_container import stylable_container
import random

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
    st.session_state.show_welcome = True

# Function to load Lottie animations
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load Lottie animations
lottie_quiz = load_lottieurl("https://lottie.host/1fa28c0d-2fa1-41d0-b3f8-3e31a4401e87/fX7Z7kDMcT.json")
lottie_brain = load_lottieurl("https://lottie.host/ea5e3e26-f141-4a68-a880-1f13371d637e/rqh9YnBbpI.json")
lottie_correct = load_lottieurl("https://lottie.host/63de93c4-2f87-45de-b3d6-af7b254723ea/CTANuYO3Yu.json")
lottie_wrong = load_lottieurl("https://lottie.host/ae11e8e9-3803-4ecf-9d0e-5b41e8fe4d22/YO9j0yw5hJ.json")
lottie_trophy = load_lottieurl("https://lottie.host/a83ad1b2-3e33-402e-af0c-1fa7aa7cb52d/dN4iQOi9sC.json")

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
    /* Google Fonts Integration */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700&display=swap');
    
    /* Main page styling */
    .main {
        background-color: #f8f9fa;
        color: #333333;
        font-family: 'Poppins', sans-serif;
    }
    
    /* App container */
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    
    p, ul, li {
        color: #555555;
        font-family: 'Poppins', sans-serif;
        line-height: 1.6;
    }
    
    /* Question card styling with glass morphism */
    .question-card {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        padding: 25px;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.15);
        margin-bottom: 25px;
        border-left: 5px solid #6c5ce7;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        animation: fadeIn 0.5s ease-out;
    }
    .question-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 36px rgba(31, 38, 135, 0.2);
    }
    
    /* Option buttons styling with hover effects */
    .option-button {
        width: 100%;
        text-align: left;
        margin: 10px 0;
        padding: 16px 20px;
        border-radius: 12px;
        border: 1px solid #e0e0e0;
        background-color: white;
        color: #333;
        transition: all 0.3s ease;
        font-size: 16px;
        cursor: pointer;
        position: relative;
        overflow: hidden;
        font-family: 'Poppins', sans-serif;
    }
    .option-button:before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 0%;
        height: 100%;
        background-color: rgba(108, 92, 231, 0.1);
        transition: all 0.3s ease;
        z-index: 0;
    }
    .option-button:hover {
        background-color: #f7f7ff;
        border-color: #6c5ce7;
        transform: translateX(5px);
    }
    .option-button:hover:before {
        width: 100%;
    }
    
    /* Answer feedback styling with animations */
    .selected-correct {
        background-color: #d4edda;
        border-color: #28a745;
        border-left: 5px solid #28a745;
        animation: pulseGreen 1s;
    }
    .selected-incorrect {
        background-color: #f8d7da;
        border-color: #dc3545;
        border-left: 5px solid #dc3545;
        animation: pulseRed 1s;
    }
    .correct-answer {
        background-color: #d4edda;
        border-color: #28a745;
        border-left: 5px solid #28a745;
    }
    
    /* Button styling with hover effects */
    .stButton > button {
        width: 100%;
        border-radius: 12px;
        font-weight: 500;
        transition: all 0.3s ease;
        border: none;
        padding: 12px 24px;
        font-family: 'Poppins', sans-serif;
        position: relative;
        overflow: hidden;
        z-index: 1;
    }
    .stButton > button:before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, rgba(255,255,255,0.1), rgba(255,255,255,0.3));
        transition: all 0.6s;
        z-index: -1;
    }
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 7px 14px rgba(50, 50, 93, 0.1), 0 3px 6px rgba(0, 0, 0, 0.08);
    }
    .stButton > button:hover:before {
        left: 100%;
    }
    .stButton > button:active {
        transform: translateY(1px);
    }
    
    /* Explanation box styling with glass effect */
    .explanation-box {
        background: rgba(248, 249, 250, 0.8);
        backdrop-filter: blur(7px);
        -webkit-backdrop-filter: blur(7px);
        color: #333;
        padding: 20px;
        border-radius: 12px;
        margin-top: 18px;
        border-left: 3px solid #6c5ce7;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        animation: slideUp 0.5s ease-out;
    }
    
    /* Title styling with gradient and animation */
    .title-container {
        text-align: center;
        padding: 35px 0;
        background: linear-gradient(135deg, #6c5ce7, #a29bfe);
        border-radius: 20px;
        margin-bottom: 30px;
        color: white;
        box-shadow: 0 10px 25px rgba(108, 92, 231, 0.2);
        position: relative;
        overflow: hidden;
        animation: gradientShift 10s ease infinite;
    }
    .title-container:before {
        content: '';
        position: absolute;
        top: -10%;
        left: -10%;
        width: 120%;
        height: 120%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0) 70%);
        transform: rotate(30deg);
    }
    .title-text {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        font-family: 'Montserrat', sans-serif;
        position: relative;
        z-index: 1;
    }
    .subtitle-text {
        font-size: 1.2rem;
        color: rgba(255, 255, 255, 0.9);
        opacity: 0.9;
        margin-top: 10px;
        font-family: 'Poppins', sans-serif;
        position: relative;
        z-index: 1;
    }
    
    /* Analytics card styling with glass morphism */
    .analytics-card {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        padding: 25px;
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.1);
        margin-bottom: 25px;
        border-top: 4px solid #6c5ce7;
        transition: all 0.3s ease;
    }
    .analytics-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 40px rgba(31, 38, 135, 0.15);
    }
    
    /* Metric styling with modern design */
    .metric-card {
        background: linear-gradient(135deg, #6c5ce7, #a29bfe);
        padding: 20px;
        border-radius: 16px;
        color: white;
        box-shadow: 0 7px 20px rgba(108, 92, 231, 0.25);
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    .metric-card:before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0) 70%);
        transform: rotate(30deg);
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(108, 92, 231, 0.3);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 10px 0;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
        position: relative;
        z-index: 1;
        font-family: 'Montserrat', sans-serif;
    }
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
        font-family: 'Poppins', sans-serif;
        position: relative;
        z-index: 1;
    }
    
    /* Progress bar styling */
    .stProgress > div > div {
        background: linear-gradient(to right, #6c5ce7, #a29bfe);
        border-radius: 10px;
        box-shadow: 0 3px 8px rgba(108, 92, 231, 0.2);
        transition: width 0.5s ease;
    }
    .stProgress {
        height: 10px;
    }
    
    /* Sidebar styling with glass morphism */
    .sidebar .sidebar-content {
        background: rgba(248, 249, 250, 0.9);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
    }
    
    /* Widget labels */
    .stTextInput > label, .stSelectbox > label, .stNumberInput > label {
        font-weight: 500;
        font-size: 16px;
        color: #333;
        font-family: 'Poppins', sans-serif;
        margin-bottom: 8px;
    }
    
    /* Input fields styling */
    .stTextInput input, .stSelectbox > div > div[data-baseweb="select"] > div, .stNumberInput input {
        border-radius: 10px !important;
        border: 1px solid #e0e0e0 !important;
        padding: 10px !important;
        font-family: 'Poppins', sans-serif !important;
        transition: all 0.3s ease !important;
    }
    .stTextInput input:focus, .stSelectbox > div > div[data-baseweb="select"] > div:focus, .stNumberInput input:focus {
        border-color: #6c5ce7 !important;
        box-shadow: 0 0 0 2px rgba(108, 92, 231, 0.2) !important;
    }
    
    /* Remove fullscreen button */
    .modebar-container {
        display: none !important;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes pulseGreen {
        0% { box-shadow: 0 0 0 0 rgba(40, 167, 69, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(40, 167, 69, 0); }
        100% { box-shadow: 0 0 0 0 rgba(40, 167, 69, 0); }
    }
    
    @keyframes pulseRed {
        0% { box-shadow: 0 0 0 0 rgba(220, 53, 69, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(220, 53, 69, 0); }
        100% { box-shadow: 0 0 0 0 rgba(220, 53, 69, 0); }
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Welcome modal */
    .welcome-modal {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.1);
        position: relative;
        z-index: 1000;
        max-width: 800px;
        margin: 0 auto;
        animation: fadeIn 0.5s ease-out;
    }
    
    /* Topic cards */
    .topic-card {
        background: white;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
        cursor: pointer;
        border-left: 3px solid #6c5ce7;
    }
    .topic-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #6c5ce7;
    }
    
    /* Question counter badge */
    .question-counter {
        background: rgba(108, 92, 231, 0.1);
        color: #6c5ce7;
        font-weight: 600;
        padding: 5px 15px;
        border-radius: 20px;
        display: inline-block;
        margin-bottom: 10px;
        font-family: 'Poppins', sans-serif;
    }
    
    /* Custom tooltips */
    .tooltip {
        position: relative;
        display: inline-block;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #333;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 8px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 14px;
    }
    .tooltip .tooltiptext::after {
        content: "";
        position: absolute;
        top: 100%;
        left: 50%;
        margin-left: -5px;
        border-width: 5px;
        border-style: solid;
        border-color: #333 transparent transparent transparent;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    /* Notification badge */
    .notification-badge {
        background: #6c5ce7;
        color: white;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 12px;
        position: absolute;
        top: -10px;
        right: -10px;
        box-shadow: 0 3px 8px rgba(108, 92, 231, 0.3);
    }
    
    /* Loading spinner */
    .loading-spinner {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100px;
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb {
        background: #6c5ce7;
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #a29bfe;
    }
    
    /* Dark mode toggle */
    .dark-mode-toggle {
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 1000;
        background: white;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        display: flex;
        justify-content: center;
        align-items: center;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .dark-mode-toggle:hover {
        transform: rotate(30deg);
    }
    
    /* Interactive elements */
    .interact {
        transition: all 0.3s ease;
    }
    .interact:hover {
        transform: scale(1.05);
    }
    
    /* Score display animation */
    .score-display {
        animation: bounceIn 0.8s;
    }
    @keyframes bounceIn {
        0% { transform: scale(0.3); opacity: 0; }
        50% { transform: scale(1.1); opacity: 1; }
        70% { transform: scale(0.9); }
        100% { transform: scale(1); }
    }
</style>
""", unsafe_allow_html=True)

# Add a dynamic background
st.markdown("""
<style>
body {
    background: linear-gradient(-45deg, #f8f9fa, #e9ecef, #dee2e6, #f1f3f5);
    background-size: 400% 400%;
    animation: gradient 15s ease infinite;
}

@keyframes gradient {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
</style>
""", unsafe_allow_html=True)

# Add confetti effect for celebrations
st.markdown("""
<script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.5.1/dist/confetti.browser.min.js"></script>
<script>
function runConfetti() {
    confetti({
        particleCount: 100,
        spread: 70,
        origin: { y: 0.6 }
    });
}
</script>
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
                <p>After completing quizzes, you'll see detailed analytics including:</p>
                <ul>
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
                <p>Based on your performance in the "{last_quiz['topic']}" quiz:</p>
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
    # Add a copyright notice in footer
    st.markdown(
        """
        <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background: linear-gradient(90deg, rgba(108, 92, 231, 0.1), rgba(162, 155, 254, 0.1));
            backdrop-filter: blur(5px);
            color: #6c5ce7;
            text-align: center;
            padding: 10px;
            font-size: 13px;
            font-family: 'Poppins', sans-serif;
            z-index: 999;
            border-top: 1px solid rgba(108, 92, 231, 0.2);
        }
        .footer a {
            color: #6c5ce7;
            text-decoration: none;
            font-weight: 600;
        }
        .footer a:hover {
            text-decoration: underline;
        }
        </style>
        <div class="footer">
            © 2024 Adaptive MCQ Generator | Created by <a href="mailto:nandeshkalshetti1@gmail.com">Nandesh Kalashetti</a> | 
            If you like this project, please give it a ⭐ and send feedback to nandeshkalshetti1@gmail.com
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Enhanced background with particle animation
    st.markdown(
        """
        <style>
        @keyframes move {
            100% {
                transform: translate3d(0, 0, 1px) rotate(360deg);
            }
        }
        
        .background {
            position: fixed;
            width: 100vw;
            height: 100vh;
            top: 0;
            left: 0;
            background: #f8f9fa;
            overflow: hidden;
            z-index: -1;
        }
        
        .background span {
            width: 20vmin;
            height: 20vmin;
            border-radius: 20vmin;
            backface-visibility: hidden;
            position: absolute;
            animation: move;
            animation-duration: 36s;
            animation-timing-function: linear;
            animation-iteration-count: infinite;
        }
        
        .background span:nth-child(0) {
            color: rgba(108, 92, 231, 0.1);
            top: 56%;
            left: 13%;
            animation-duration: 21s;
            animation-delay: -31s;
            transform-origin: -19vw -3vh;
            box-shadow: -40vmin 0 5.2vmin currentColor;
        }
        
        .background span:nth-child(1) {
            color: rgba(108, 92, 231, 0.1);
            top: 22%;
            left: 13%;
            animation-duration: 21s;
            animation-delay: -20s;
            transform-origin: 19vw -11vh;
            box-shadow: 40vmin 0 5.3vmin currentColor;
        }
        
        .background span:nth-child(2) {
            color: rgba(162, 155, 254, 0.1);
            top: 77%;
            left: 82%;
            animation-duration: 34s;
            animation-delay: -17s;
            transform-origin: -13vw 24vh;
            box-shadow: 40vmin 0 5.8vmin currentColor;
        }
        
        .background span:nth-child(3) {
            color: rgba(108, 92, 231, 0.1);
            top: 93%;
            left: 76%;
            animation-duration: 33s;
            animation-delay: -36s;
            transform-origin: 9vw 7vh;
            box-shadow: 40vmin 0 5.5vmin currentColor;
        }
        
        .background span:nth-child(4) {
            color: rgba(162, 155, 254, 0.1);
            top: 3%;
            left: 29%;
            animation-duration: 66s;
            animation-delay: -30s;
            transform-origin: -20vw -1vh;
            box-shadow: -40vmin 0 5.3vmin currentColor;
        }
        
        .background span:nth-child(5) {
            color: rgba(108, 92, 231, 0.1);
            top: 86%;
            left: 84%;
            animation-duration: 42s;
            animation-delay: -20s;
            transform-origin: 15vw -21vh;
            box-shadow: 40vmin 0 5.7vmin currentColor;
        }
        </style>
        
        <div class="background">
           <span></span>
           <span></span>
           <span></span>
           <span></span>
           <span></span>
           <span></span>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Welcome screen (only shown on first load)
    if st.session_state.show_welcome:
        with st.container():
            st.markdown(
                """
                <div class="welcome-modal">
                    <h1 style="color: #6c5ce7; text-align: center; font-family: 'Montserrat', sans-serif; margin-bottom: 20px;">
                        Welcome to the Adaptive MCQ Generator
                    </h1>
                    <div style="display: flex; justify-content: center; margin-bottom: 20px;">
                """,
                unsafe_allow_html=True
            )
            
            col1, col2 = st.columns([1, 1])
            with col1:
                if lottie_brain:
                    st_lottie(lottie_brain, height=200, key="welcome_lottie")
            
            with col2:
                st.markdown(
                    """
                    <div style="padding: 20px;">
                        <h3 style="color: #333; font-family: 'Poppins', sans-serif;">🧠 Test Your Knowledge</h3>
                        <p style="color: #555;">
                            This intelligent quiz system adapts to your performance level and 
                            helps you learn more effectively with personalized questions.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Featured topics
            st.markdown(
                """
                <h3 style="color: #6c5ce7; font-family: 'Poppins', sans-serif; margin: 15px 0;">
                    Popular Topics to Try
                </h3>
                """,
                unsafe_allow_html=True
            )
            
            # Display topic cards
            topic_col1, topic_col2, topic_col3 = st.columns(3)
            popular_topics = [
                {"name": "Quantum Physics", "icon": "⚛️", "description": "Explore the strange world of quantum mechanics"},
                {"name": "World History", "icon": "🌍", "description": "Journey through the key events that shaped our world"},
                {"name": "Machine Learning", "icon": "🤖", "description": "Dive into AI algorithms and neural networks"}
            ]
            
            for i, (col, topic) in enumerate(zip([topic_col1, topic_col2, topic_col3], popular_topics)):
                with col:
                    with stylable_container(
                        key=f"topic_{i}",
                        css_styles="""
                            {
                                background: white;
                                border-radius: 15px;
                                padding: 20px;
                                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
                                transition: all 0.3s ease;
                                border-left: 3px solid #6c5ce7;
                                margin-bottom: 15px;
                            }
                            :hover {
                                transform: translateY(-5px);
                                box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
                                border-left: 5px solid #6c5ce7;
                            }
                        """
                    ):
                        st.markdown(f"<h2 style='margin: 0; font-size: 28px;'>{topic['icon']}</h2>", unsafe_allow_html=True)
                        st.markdown(f"<h4 style='margin: 10px 0; color: #333;'>{topic['name']}</h4>", unsafe_allow_html=True)
                        st.markdown(f"<p style='color: #666; font-size: 14px;'>{topic['description']}</p>", unsafe_allow_html=True)
                        topic_btn = st.button(f"Start Quiz", key=f"topic_btn_{i}")
                        if topic_btn:
                            st.session_state.topic = topic["name"]
                            st.session_state.show_welcome = False
                            st.rerun()
            
            # Get started button
            st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("Get Started with Your Own Topic", use_container_width=True, type="primary"):
                    st.session_state.show_welcome = False
                    st.rerun()
                
            st.markdown("</div>", unsafe_allow_html=True)
    
    # Sidebar navigation and information
    with st.sidebar:
        col1, col2 = st.columns([1, 3])
        with col1:
            st_lottie(lottie_quiz, height=80, key="sidebar_lottie")
        with col2:
            st.markdown("<h1 style='text-align: left; color: #6c5ce7; font-family: \"Montserrat\", sans-serif; font-size: 24px;'>Adaptive MCQ Generator</h1>", unsafe_allow_html=True)
        
        st.divider()
        
        # Navigation with stylish tabs
        with stylable_container(
            key="tabs_container",
            css_styles="""
                {
                    background: rgba(255, 255, 255, 0.8);
                    padding: 10px;
                    border-radius: 10px;
                    margin-bottom: 15px;
                    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
                }
            """
        ):
            page = st.radio("📋 Navigation", ["🧠 Generate MCQs", "📊 Analytics"], index=0)
        
        st.divider()
        
        # About section with card
        with stylable_container(
            key="about_container",
            css_styles="""
                {
                    background: linear-gradient(135deg, #f8f9fa, #e9ecef);
                    border-radius: 15px;
                    padding: 15px;
                    margin-bottom: 15px;
                    border-left: 3px solid #6c5ce7;
                    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
                }
            """
        ):
            st.markdown("<h3 style='color: #6c5ce7; font-family: \"Poppins\", sans-serif;'>📌 About</h3>", unsafe_allow_html=True)
            st.info(
                "This intelligent MCQ generator creates adaptive assessment questions that adjust difficulty based on your performance. "
                "Powered by Groq LLM, it generates high-quality questions for educational assessment."
            )
        
        # Usage tips with better styling
        with stylable_container(
            key="tips_container",
            css_styles="""
                {
                    background: linear-gradient(135deg, #f1f3f5, #e9ecef);
                    border-radius: 15px;
                    padding: 15px;
                    margin-bottom: 15px;
                    border-left: 3px solid #6c5ce7;
                    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
                }
            """
        ):
            st.markdown("<h3 style='color: #6c5ce7; font-family: \"Poppins\", sans-serif;'>💡 Tips</h3>", unsafe_allow_html=True)
            
            tip1, tip2 = st.columns(2)
            with tip1:
                st.markdown(
                    """
                    <div style="padding: 10px; background: white; border-radius: 10px; margin-bottom: 10px; transition: all 0.3s ease;" onmouseover="this.style.transform='translateY(-5px)'" onmouseout="this.style.transform='translateY(0px)'">
                        <p style="margin: 0; color: #333;"><strong>✓</strong> Enter any educational topic</p>
                    </div>
                    <div style="padding: 10px; background: white; border-radius: 10px; margin-bottom: 10px;">
                        <p style="margin: 0; color: #333;"><strong>✓</strong> Start with Medium difficulty</p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            with tip2:
                st.markdown(
                    """
                    <div style="padding: 10px; background: white; border-radius: 10px; margin-bottom: 10px;">
                        <p style="margin: 0; color: #333;"><strong>✓</strong> Answer questions to build your profile</p>
                    </div>
                    <div style="padding: 10px; background: white; border-radius: 10px; margin-bottom: 10px;">
                        <p style="margin: 0; color: #333;"><strong>✓</strong> System adapts to your performance</p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
        
        st.divider()
        
        # GitHub link with better styling
        with stylable_container(
            key="links_container",
            css_styles="""
                {
                    background: linear-gradient(135deg, #f8f9fa, #e9ecef);
                    border-radius: 15px;
                    padding: 15px;
                    border-left: 3px solid #6c5ce7;
                    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
                }
            """
        ):
            st.markdown("<h3 style='color: #6c5ce7; font-family: \"Poppins\", sans-serif;'>🔗 Links</h3>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(
                    """
                    <a href="https://github.com/Universe7Nandu/McqChatbot" target="_blank" style="text-decoration: none;">
                        <div style="padding: 10px; background: white; border-radius: 10px; text-align: center; transition: all 0.3s ease;" onmouseover="this.style.transform='scale(1.05)'" onmouseout="this.style.transform='scale(1)'">
                            <span style="font-size: 24px;">📂</span>
                            <p style="margin: 5px 0 0 0; color: #333; font-size: 14px;">GitHub Repo</p>
                        </div>
                    </a>
                    """,
                    unsafe_allow_html=True
                )
            with col2:
                st.markdown(
                    """
                    <a href="https://github.com/Universe7Nandu/McqChatbot/issues" target="_blank" style="text-decoration: none;">
                        <div style="padding: 10px; background: white; border-radius: 10px; text-align: center; transition: all 0.3s ease;" onmouseover="this.style.transform='scale(1.05)'" onmouseout="this.style.transform='scale(1)'">
                            <span style="font-size: 24px;">🔔</span>
                            <p style="margin: 5px 0 0 0; color: #333; font-size: 14px;">Report Issues</p>
                        </div>
                    </a>
                    """,
                    unsafe_allow_html=True
                )
        
    # Main content area
    if not st.session_state.show_welcome:
        if page == "🧠 Generate MCQs":
            # Title section with custom styling and animation
            st.markdown(
                """
                <div class='title-container' style="background-size: 300% 300%;">
                    <h1 class='title-text'>🧠 Intelligent MCQ Generator</h1>
                    <p class='subtitle-text'>Create adaptive assessment questions on any educational topic</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            # Input form with improved styling
            with st.container():
                st.markdown("<h3 style='color: #6c5ce7; font-family: \"Poppins\", sans-serif;'>Enter Topic Details</h3>", unsafe_allow_html=True)
                
                # Use a card-like container for inputs
                with stylable_container(
                    key="input_container",
                    css_styles="""
                        {
                            background: rgba(255, 255, 255, 0.8);
                            backdrop-filter: blur(10px);
                            -webkit-backdrop-filter: blur(10px);
                            padding: 25px;
                            border-radius: 15px;
                            box-shadow: 0 8px 32px rgba(31, 38, 135, 0.1);
                            margin-bottom: 25px;
                            border-left: 4px solid #6c5ce7;
                            animation: fadeIn 0.5s ease-out;
                        }
                    """
                ):
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
                generate_col1, generate_col2, generate_col3 = st.columns([1, 2, 1])
                with generate_col2:
                    generate_button = st.button(
                        "🔮 Generate MCQs",
                        key="generate_button",
                        use_container_width=True,
                        type="primary"
                    )
                    
                    # Help text
                    st.markdown(
                        "<div style='text-align: center; color: #666; font-size: 14px; margin-top: 10px; font-family: \"Poppins\", sans-serif;'>"
                        "The system will generate questions and adapt to your performance over time."
                        "</div>",
                        unsafe_allow_html=True
                    )
            
            # Quick topic suggestions
            if not st.session_state.questions and not generate_button:
                st.markdown(
                    """
                    <h4 style='color: #6c5ce7; font-family: "Poppins", sans-serif; margin: 25px 0 15px 0;'>
                        📚 Suggested Topics to Try
                    </h4>
                    """,
                    unsafe_allow_html=True
                )
                
                # Create a scrollable list of suggested topics
                suggested_topics = [
                    "Quantum Physics", "Machine Learning", "World History", 
                    "Human Anatomy", "Climate Change", "Financial Markets",
                    "Organic Chemistry", "Astronomy", "Classical Literature",
                    "Artificial Intelligence", "Genetics", "Game Theory"
                ]
                
                # Display as a grid of cards
                cols = st.columns(4)
                for i, topic_name in enumerate(suggested_topics):
                    with cols[i % 4]:
                        with stylable_container(
                            key=f"topic_suggestion_{i}",
                            css_styles="""
                                {
                                    background: white;
                                    border-radius: 12px;
                                    padding: 15px;
                                    margin-bottom: 15px;
                                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05);
                                    text-align: center;
                                    cursor: pointer;
                                    transition: all 0.3s ease;
                                    border-bottom: 3px solid #6c5ce7;
                                }
                                :hover {
                                    transform: translateY(-5px);
                                    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
                                }
                            """
                        ):
                            st.markdown(f"<p style='margin: 0; color: #333; font-weight: 500;'>{topic_name}</p>", unsafe_allow_html=True)
                            if st.button(f"Select", key=f"select_topic_{i}", use_container_width=True):
                                st.session_state.topic = topic_name
                                st.rerun()
            
            # Update session state
            st.session_state.topic = topic
            st.session_state.difficulty_level = difficulty
            st.session_state.num_questions = num_questions
            
            # Generate questions when button is clicked
            if generate_button and topic:
                with st.spinner(""):
                    # Custom loading animation
                    st.markdown(
                        """
                        <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 30px;">
                            <div style="width: 50px; height: 50px; border: 5px solid #f3f3f3; border-top: 5px solid #6c5ce7; border-radius: 50%; animation: spin 1s linear infinite;"></div>
                            <p style="margin-top: 20px; color: #6c5ce7; font-weight: 500; font-family: 'Poppins', sans-serif;">Generating intelligent questions...</p>
                            <style>
                                @keyframes spin {
                                    0% { transform: rotate(0deg); }
                                    100% { transform: rotate(360deg); }
                                }
                            </style>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
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
                    <div style='margin-bottom: 15px; display: flex; align-items: center;'>
                        <div class='question-counter'>
                            Question {current_idx + 1} of {st.session_state.total}
                        </div>
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
                            <h3 style='color: #333; margin-bottom: 20px; font-family: "Poppins", sans-serif;'>{question}</h3>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    # Answer options
                    option_labels = ["A", "B", "C", "D"]
                    selected_option = None
                    
                    # Check if this question has already been answered
                    already_answered = len(st.session_state.answers) > current_idx
                    
                    st.markdown("<h4 style='color: #6c5ce7; margin-bottom: 15px; font-family: \"Poppins\", sans-serif;'>Select your answer:</h4>", unsafe_allow_html=True)
                    
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
                                
                                # Display animations for correct/incorrect
                                if is_user_selection:
                                    if is_correct_answer:
                                        st_lottie(lottie_correct, height=80, key=f"correct_lottie_{current_idx}")
                                    else:
                                        st_lottie(lottie_wrong, height=80, key=f"wrong_lottie_{current_idx}")
                            else:
                                # Display clickable buttons for unanswered questions
                                # Using custom container for better hover effects
                                with stylable_container(
                                    key=f"option_container_{current_idx}_{label}",
                                    css_styles="""
                                        {
                                            margin-bottom: 10px;
                                        }
                                        button {
                                            width: 100%;
                                            text-align: left;
                                            padding: 16px 20px;
                                            border-radius: 12px;
                                            border: 1px solid #e0e0e0;
                                            background-color: white;
                                            color: #333;
                                            transition: all 0.3s ease;
                                            font-size: 16px;
                                            position: relative;
                                            overflow: hidden;
                                            font-family: 'Poppins', sans-serif;
                                        }
                                        button:hover {
                                            background-color: #f7f7ff;
                                            border-color: #6c5ce7;
                                            transform: translateX(5px);
                                            box-shadow: 0 4px 10px rgba(108, 92, 231, 0.1);
                                        }
                                    """
                                ):
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
                                <h4 style='color: #6c5ce7; margin-top: 0; font-family: "Poppins", sans-serif;'>Explanation</h4>
                                <p style='color: #333;'>{st.session_state.explanations[current_idx]}</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    
                    # Navigation buttons with improved styling
                    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        with stylable_container(
                            key=f"prev_btn_container",
                            css_styles="""
                                button {
                                    background: linear-gradient(to right, #a29bfe, #6c5ce7);
                                    color: white;
                                    border: none;
                                    border-radius: 12px;
                                    padding: 12px 24px;
                                    font-weight: 500;
                                    transition: all 0.3s ease;
                                    font-family: 'Poppins', sans-serif;
                                    position: relative;
                                    overflow: hidden;
                                }
                                button:hover {
                                    transform: translateY(-3px);
                                    box-shadow: 0 7px 14px rgba(108, 92, 231, 0.2);
                                }
                                button:active {
                                    transform: translateY(1px);
                                }
                            """
                        ):
                            if current_idx > 0:
                                if st.button("⬅️ Previous Question", use_container_width=True):
                                    st.session_state.current_question -= 1
                                    st.rerun()
                    
                    with col2:
                        with stylable_container(
                            key=f"next_btn_container",
                            css_styles="""
                                button {
                                    background: linear-gradient(to right, #6c5ce7, #a29bfe);
                                    color: white;
                                    border: none;
                                    border-radius: 12px;
                                    padding: 12px 24px;
                                    font-weight: 500;
                                    transition: all 0.3s ease;
                                    font-family: 'Poppins', sans-serif;
                                    position: relative;
                                    overflow: hidden;
                                }
                                button:hover {
                                    transform: translateY(-3px);
                                    box-shadow: 0 7px 14px rgba(108, 92, 231, 0.2);
                                }
                                button:active {
                                    transform: translateY(1px);
                                }
                            """
                        ):
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
                # JavaScript for confetti effect
                st.markdown(
                    """
                    <script>
                    function launchConfetti() {
                        confetti({
                            particleCount: 150,
                            spread: 70,
                            origin: { y: 0.6 }
                        });
                    }
                    document.addEventListener("DOMContentLoaded", function() {
                        launchConfetti();
                    });
                    </script>
                    """,
                    unsafe_allow_html=True
                )
                
                # Show celebration animation
                col1, col2, col3 = st.columns([1, 1, 1])
                with col2:
                    st_lottie(lottie_trophy, height=200, key="trophy_animation")
                
                colored_header(
                    label="🏆 Quiz Results",
                    description=f"Topic: {st.session_state.topic} | Difficulty: {st.session_state.difficulty_level}",
                    color_name="blue-70"
                )
                
                # Score display with improved visualization
                score_percentage = (st.session_state.score / st.session_state.total) * 100
                
                # Create score card with glass morphism effect
                st.markdown(
                    f"""
                    <div style='background: linear-gradient(135deg, rgba(108, 92, 231, 0.9), rgba(162, 155, 254, 0.9)); 
                         backdrop-filter: blur(10px); 
                         -webkit-backdrop-filter: blur(10px);
                         border-radius: 20px; 
                         padding: 30px; 
                         text-align: center; 
                         color: white; 
                         margin-bottom: 30px;
                         box-shadow: 0 10px 30px rgba(108, 92, 231, 0.3);
                         position: relative;
                         overflow: hidden;
                         animation: fadeIn 0.8s ease-out, pulse 2s infinite alternate;'>
                        <div style='position: absolute; top: -50%; left: -50%; width: 200%; height: 200%; background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0) 70%); transform: rotate(30deg);'></div>
                        <h2 style='margin-bottom: 10px; position: relative; z-index: 1; font-family: "Montserrat", sans-serif;'>Your Score</h2>
                        <div class='score-display' style='font-size: 60px; font-weight: 700; margin: 20px 0; position: relative; z-index: 1; font-family: "Montserrat", sans-serif; text-shadow: 2px 2px 4px rgba(0,0,0,0.2);'>
                            {st.session_state.score} / {st.session_state.total}
                        </div>
                        <div style='font-size: 28px; font-weight: 500; position: relative; z-index: 1; font-family: "Poppins", sans-serif;'>
                            {score_percentage:.1f}%
                        </div>
                        <style>
                            @keyframes pulse {
                                0% { box-shadow: 0 10px 30px rgba(108, 92, 231, 0.3); }
                                100% { box-shadow: 0 15px 40px rgba(108, 92, 231, 0.5); }
                            }
                        </style>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Performance gauge chart with animated progress
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    with stylable_container(
                        key="gauge_container",
                        css_styles="""
                            {
                                background: rgba(255, 255, 255, 0.8);
                                backdrop-filter: blur(10px);
                                -webkit-backdrop-filter: blur(10px);
                                border-radius: 15px;
                                padding: 20px;
                                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05);
                                margin-bottom: 20px;
                                animation: fadeIn 0.5s ease-out;
                            }
                        """
                    ):
                        fig, ax = plt.subplots(figsize=(8, 3))
                        ax.set_xlim(0, 100)
                        ax.set_ylim(0, 1)
                        ax.set_yticks([])
                        ax.set_xticks([0, 25, 50, 75, 100])
                        ax.axvspan(0, 50, alpha=0.2, color='#ff7675')
                        ax.axvspan(50, 75, alpha=0.2, color='#fdcb6e')
                        ax.axvspan(75, 100, alpha=0.2, color='#55efc4')
                        ax.axvline(score_percentage, color='#6c5ce7', linewidth=6)
                        ax.set_title('Performance Meter', fontsize=16, color='#333', fontfamily='Poppins')
                        st.pyplot(fig)
                        
                        # Add score labels
                        st.markdown(
                            f"""
                            <div style="display: flex; justify-content: space-between; margin-top: 5px; font-family: 'Poppins', sans-serif;">
                                <div style="color: #ff7675; font-weight: 500;">Needs Improvement</div>
                                <div style="color: #fdcb6e; font-weight: 500;">Good</div>
                                <div style="color: #55efc4; font-weight: 500;">Excellent</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                
                # Performance feedback with enhanced styling
                with stylable_container(
                    key="feedback_container",
                    css_styles="""
                        {
                            background: rgba(255, 255, 255, 0.8);
                            backdrop-filter: blur(10px);
                            -webkit-backdrop-filter: blur(10px);
                            border-radius: 15px;
                            padding: 20px;
                            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05);
                            margin-bottom: 20px;
                            border-left: 4px solid #6c5ce7;
                            animation: fadeIn 0.5s ease-out;
                        }
                    """
                ):
                    if score_percentage >= 75:
                        st.markdown(
                            """
                            <div style="display: flex; align-items: center;">
                                <div style="font-size: 40px; margin-right: 15px;">🏆</div>
                                <div>
                                    <h3 style="color: #6c5ce7; margin: 0; font-family: 'Poppins', sans-serif;">Excellent!</h3>
                                    <p style="color: #333; margin-top: 5px;">You have a strong understanding of this topic. Keep up the great work!</p>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    elif score_percentage >= 50:
                        st.markdown(
                            """
                            <div style="display: flex; align-items: center;">
                                <div style="font-size: 40px; margin-right: 15px;">👍</div>
                                <div>
                                    <h3 style="color: #6c5ce7; margin: 0; font-family: 'Poppins', sans-serif;">Good job!</h3>
                                    <p style="color: #333; margin-top: 5px;">You have a decent grasp of the subject but there's room for improvement.</p>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            """
                            <div style="display: flex; align-items: center;">
                                <div style="font-size: 40px; margin-right: 15px;">💪</div>
                                <div>
                                    <h3 style="color: #6c5ce7; margin: 0; font-family: 'Poppins', sans-serif;">Keep practicing!</h3>
                                    <p style="color: #333; margin-top: 5px;">You might need more practice on this topic. Don't give up! Try again to improve your score.</p>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                
                # Learning recommendation based on performance
                with stylable_container(
                    key="recommendation_container",
                    css_styles="""
                        {
                            background: linear-gradient(135deg, #f8f9fa, #e9ecef);
                            border-radius: 15px;
                            padding: 20px;
                            margin-bottom: 20px;
                            border-left: 5px solid #6c5ce7;
                            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05);
                            animation: fadeIn 0.5s ease-out;
                            position: relative;
                            overflow: hidden;
                        }
                        :before {
                            content: '';
                            position: absolute;
                            top: -10%;
                            left: -10%;
                            width: 120%;
                            height: 120%;
                            background: radial-gradient(circle, rgba(108, 92, 231, 0.05) 0%, rgba(255, 255, 255, 0) 70%);
                        }
                    """
                ):
                    st.markdown(
                        f"""
                        <h4 style='color: #6c5ce7; margin-top: 0; font-family: "Poppins", sans-serif; position: relative; z-index: 1;'>🎓 Learning Recommendations</h4>
                        <p style='color: #333; position: relative; z-index: 1;'>Based on your performance in the <strong>"{st.session_state.topic}"</strong> quiz:</p>
                        <ul style='color: #333; position: relative; z-index: 1;'>
                            {"<li>Consider moving to a <strong>harder difficulty</strong> for this topic.</li>" if score_percentage > 80 else ""}
                            {"<li>You're doing well! Continue practicing at the current difficulty.</li>" if 60 <= score_percentage <= 80 else ""}
                            {"<li>Try studying this topic more or attempt an <strong>easier difficulty</strong>.</li>" if score_percentage < 60 else ""}
                            {"<li>Focus on reviewing the questions you answered incorrectly.</li>" if st.session_state.total - st.session_state.score > 0 else ""}
                            {"<li>Try a new related topic to expand your knowledge.</li>" if score_percentage > 70 else ""}
                        </ul>
                        """,
                        unsafe_allow_html=True
                    )
                
                # Question review with improved styling
                st.subheader("📝 Question Review")
                
                # Add toggle button for question review
                show_review = st.toggle("Show detailed question review", value=True)
                
                if show_review:
                    for i, (question, user_answer, is_correct, explanation) in enumerate(
                        zip(st.session_state.questions, st.session_state.answers, st.session_state.feedback, st.session_state.explanations)
                    ):
                        with st.expander(f"Question {i+1}: {question[0][:60]}...", expanded=False):
                            # Question card with animation
                            st.markdown(
                                f"""
                                <div style='padding: 15px 0; animation: fadeIn 0.5s ease-out;'>
                                    <h4 style='color: #333; font-family: "Poppins", sans-serif;'>{question[0]}</h4>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                            
                            # Display options with improved highlighting and animations
                            for j, opt in enumerate(["A", "B", "C", "D"]):
                                if opt == question[5] and opt == user_answer:  # Correct and selected
                                    st.markdown(
                                        f"""
                                        <div class='option-button selected-correct' style='animation: pulseGreen 1s;'>
                                            <strong>{opt}.</strong> {question[j+1]} ✓ <span style='color: green; font-weight: 500;'>(Your answer - Correct)</span>
                                        </div>
                                        """,
                                        unsafe_allow_html=True
                                    )
                                elif opt == question[5]:  # Correct but not selected
                                    st.markdown(
                                        f"""
                                        <div class='option-button correct-answer'>
                                            <strong>{opt}.</strong> {question[j+1]} ✓ <span style='color: green; font-weight: 500;'>(Correct answer)</span>
                                        </div>
                                        """,
                                        unsafe_allow_html=True
                                    )
                                elif opt == user_answer:  # Incorrect and selected
                                    st.markdown(
                                        f"""
                                        <div class='option-button selected-incorrect' style='animation: pulseRed 1s;'>
                                            <strong>{opt}.</strong> {question[j+1]} ✗ <span style='color: red; font-weight: 500;'>(Your answer - Incorrect)</span>
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
                            
                            # Explanation with improved styling and animation
                            st.markdown(
                                f"""
                                <div class='explanation-box' style='animation: slideUp 0.5s ease-out;'>
                                    <h4 style='color: #6c5ce7; margin-top: 0; font-family: "Poppins", sans-serif;'>Explanation</h4>
                                    <p style='color: #333;'>{explanation}</p>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                
                # Add gap
                st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
                
                # Call to action buttons with improved styling
                col1, col2 = st.columns(2)
                with col1:
                    with stylable_container(
                        key="new_topic_btn",
                        css_styles="""
                            button {
                                background: linear-gradient(to right, #a29bfe, #6c5ce7);
                                color: white;
                                border: none;
                                border-radius: 12px;
                                padding: 14px 24px;
                                font-weight: 500;
                                transition: all 0.3s ease;
                                font-family: 'Poppins', sans-serif;
                                position: relative;
                                overflow: hidden;
                            }
                            button:hover {
                                transform: translateY(-3px);
                                box-shadow: 0 7px 14px rgba(108, 92, 231, 0.2);
                            }
                            button:before {
                                content: '';
                                position: absolute;
                                top: 0;
                                left: -100%;
                                width: 100%;
                                height: 100%;
                                background: linear-gradient(90deg, rgba(255,255,255,0.1), rgba(255,255,255,0.3));
                                transition: all 0.6s;
                                z-index: 0;
                            }
                            button:hover:before {
                                left: 100%;
                            }
                        """
                    ):
                        if st.button("🆕 Try Another Topic", use_container_width=True):
                            # Reset state for a new topic
                            st.session_state.once = True
                            st.session_state.done = False
                            st.session_state.topic = ""
                            st.rerun()
                
                with col2:
                    with stylable_container(
                        key="retry_btn",
                        css_styles="""
                            button {
                                background: linear-gradient(to right, #6c5ce7, #a29bfe);
                                color: white;
                                border: none;
                                border-radius: 12px;
                                padding: 14px 24px;
                                font-weight: 500;
                                transition: all 0.3s ease;
                                font-family: 'Poppins', sans-serif;
                                position: relative;
                                overflow: hidden;
                            }
                            button:hover {
                                transform: translateY(-3px);
                                box-shadow: 0 7px 14px rgba(108, 92, 231, 0.2);
                            }
                            button:before {
                                content: '';
                                position: absolute;
                                top: 0;
                                left: -100%;
                                width: 100%;
                                height: 100%;
                                background: linear-gradient(90deg, rgba(255,255,255,0.1), rgba(255,255,255,0.3));
                                transition: all 0.6s;
                                z-index: 0;
                            }
                            button:hover:before {
                                left: 100%;
                            }
                        """
                    ):
                        if st.button("🔄 Retry This Topic", use_container_width=True):
                            # Keep the topic but reset other state for regenerating questions
                            st.session_state.once = True
                            st.session_state.done = False
                            st.rerun()
                
                # Add button to view analytics
                st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
                
                with stylable_container(
                    key="analytics_btn",
                    css_styles="""
                        button {
                            background: linear-gradient(to right, #55efc4, #00b894);
                            color: white;
                            border: none;
                            border-radius: 12px;
                            padding: 16px 24px;
                            font-weight: 500;
                            transition: all 0.3s ease;
                            font-family: 'Poppins', sans-serif;
                            position: relative;
                            overflow: hidden;
                        }
                        button:hover {
                            transform: translateY(-3px);
                            box-shadow: 0 7px 14px rgba(0, 184, 148, 0.2);
                        }
                        button:before {
                            content: '';
                            position: absolute;
                            top: 0;
                            left: -100%;
                            width: 100%;
                            height: 100%;
                            background: linear-gradient(90deg, rgba(255,255,255,0.1), rgba(255,255,255,0.3));
                            transition: all 0.6s;
                            z-index: 0;
                        }
                        button:hover:before {
                            left: 100%;
                        }
                    """
                ):
                    if st.button("📊 View Detailed Analytics", use_container_width=True):
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
