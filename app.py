# Career Path Advisor/app.py
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the trained model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("./CARRER_PATH_ADVISOR")
tokenizer = AutoTokenizer.from_pretrained("./CARRER_PATH_ADVISOR")

st.title("Career Path Advisor")
st.write("Answer a few questions, and we’ll help you discover potential career paths!")

# Function to get predictions
def get_career_path(input_text):
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=1)
    # Mapping back to category labels
    labels = ["college", "computer-science", "computer-software", "business", "doctor", "engineering", "career", "medicine", "science", "engineer", "teaching", "nursing", "psychology", "teacher", "medical", "finance", "healthcare", "college-major", "professor", "computer", "law", "nurse", "biology", "technology", "jobs", "education", "management", "any", "professional", "sports", "accounting", "university", "lawyer", "marketing", "art", "career-counseling", "internships", "music", "mechanical-engineering", "money", "health", "stem", "school", "career-choice", "job", "student", "entrepreneurship", "career-path", "business-administration", "pre-med", "graduate-school", "math", "fashion", "educator", "mathematics", "graduate", "programming", "software", "college-admissions", "job-search", "business-management", "leadership", "research", "ixzz4ofjuqwa1", "major", "police", "registered-nurses", "communications", "help", "pediatrics", "chemistry", "economics", "tech", "acting", "journalism", "criminal-justice", "design", "career-paths", "veterinarian", "higher-education", "aviation", "human-resources", "biomedical", "scholarships", "civil-engineering", "hospital", "pediatrician", "interviews", "resume", "airline-industry", "careers", "computer-engineering", "architecture", "physician", "degree", "experience", "physics", "electrical-engineering", "english", "medical-school", "sat", "social-work", "veterinary", "military", "biomedical-engineering", "dentistry", "high-school", "writing", "colleges", "volunteering", "information-technology", "law-enforcement", "physical-therapist", "students", "college-majors", "forensic", "general", "work", "advice", "architect", "gpa", "graphic-design", "video-games", "counselor", "mechanical", "investing", "surgery", "travel", "engineers", "nurse-practitioner", "pharmacy", "financial-services", "mba", "photography", "scientist", "undergraduate", "act", "artist", "doctorate-degree", "surgeon", "government", "investment-management", "software-engineering", "office-management", "pediatric-nursing", "social", "therapy", "networking", "sports-management", "aerospace-engineering", "clinical-psychology", "college-jobs", "computer-programming", "counseling", "mechanical-engineer", "animation", "bioengineering", "aerospace", "chemical-engineering", "internship", "phd", "chef", "veterinary-medicine", "arts", "business-development", "physical-therapy", "salary", "masters", "sociology", "college-selection", "entertainment", "international", "marine-biology", "classes", "film", "financial-aid", "psychiatry", "college-minor", "verizon", "animals", "software-development", "theatre", "verizoninnovativelearning", "accountant", "it", "life", "nursing-education", "actor", "dance", "history", "job-coaching", "majors", "pharmacists", "pilot", "police-officer", "recruiting", "neuroscience", "study-abroad", "time-management", "criminology", "medical-education", "advertising", "cooking", "cosmetology", "media", "career-development", "dental", "robotics", "astronomy", "basketball", "hospital-and-health-care", "biotechnology", "administration", "biochemistry", "college-student", "communication", "entrepreneur", "career-advice", "environmental", "environmental-science", "football", "mentoring", "political-science", "women-in-stem", "dentist", "gaming", "pharmaceuticals", "social-media", "texas", "cna", "college-applications", "dental-hygienist", "double-major", "language", "politics", "professors", "soccer", "student-loans", "studies", "forensics", "in", "neurology", "scholarship", "athletic-training", "culinary", "television", "athlete", "banking", "children", "college-advice", "hiring", "interior-design", "manager", "physical", "academic-advising", "coaching", "coding", "creative", "employment", "industrial-engineering", "intern", "law-school", "linkedin", "singer", "theater", "aps", "fbi", "future", "people", "project-management", "studying-tips", "aeronautics", "anthropology", "athletics", "collegelife", "criminal", "first-job", "friends", "neonatal", "organization", "public-relations", "resume-writing", "umatter", "attorney", "broadcast-media", "computer-games", "computer-hardware", "courses", "extracurriculars", "fashion-design", "graphics", "ias", "skills", "buisness", "computers", "culinary-arts", "degrees", "federal-government", "high-school-classes", "job-market", "security", "consulting", "counselling", "guidance-counselor", "marketing-and-advertising", "nutrition", "singing", "spanish"]
    return labels[predictions.item()]

# Create a questionnaire form
with st.form(key='questionnaire_form'):
    question_title = st.text_input("What is your question title?")
    question_body = st.text_area("Describe your question in detail.")
    submit_button = st.form_submit_button(label='Get Career Path')

# Display recommendation
if submit_button:
    user_input = question_title + " " + question_body
    recommendation = get_career_path(user_input)
    st.success(f"Suggested Career Path: {recommendation}")