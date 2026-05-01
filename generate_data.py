"""
generate_data.py
Run this once: python generate_data.py
Generates mbti_dataset.csv, careers.json, skills.json
"""

import json
import numpy as np
import pandas as pd
import os

np.random.seed(42)

# ── 1. MBTI TYPE DEFINITIONS ──────────────────────────────────────────────────
# Each type encoded as (E/I, N/S, T/F, J/P) where 1=first letter, 0=second
# Scores on Q1-Q4 (E/I), Q5-Q8 (N/S), Q9-Q11 (T/F), Q12-Q15 (J/P)
# Higher score → first letter of dimension

MBTI_PROFILES = {
    # type : [E/I_mean, N/S_mean, T/F_mean, J/P_mean]
    # Scale 1-5: high (4.0) = first letter of dimension, low (2.0) = second letter
    # Means are spread further apart (1.8 vs 4.2) to give clearer signal,
    # but std=1.1 creates realistic overlap between adjacent types.
    "INTJ": [1.8, 4.2, 4.2, 4.2],
    "INTP": [1.8, 4.2, 4.2, 1.8],
    "INFJ": [1.8, 4.2, 1.8, 4.2],
    "INFP": [1.8, 4.2, 1.8, 1.8],
    "ISTJ": [1.8, 1.8, 4.2, 4.2],
    "ISTP": [1.8, 1.8, 4.2, 1.8],
    "ISFJ": [1.8, 1.8, 1.8, 4.2],
    "ISFP": [1.8, 1.8, 1.8, 1.8],
    "ENTJ": [4.2, 4.2, 4.2, 4.2],
    "ENTP": [4.2, 4.2, 4.2, 1.8],
    "ENFJ": [4.2, 4.2, 1.8, 4.2],
    "ENFP": [4.2, 4.2, 1.8, 1.8],
    "ESTJ": [4.2, 1.8, 4.2, 4.2],
    "ESTP": [4.2, 1.8, 4.2, 1.8],
    "ESFJ": [4.2, 1.8, 1.8, 4.2],
    "ESFP": [4.2, 1.8, 1.8, 1.8],
}

TEXT_SAMPLES = {
    "INTJ": [
        "I prefer working independently on complex strategic problems and long term planning.",
        "I enjoy designing systems and thinking critically about theoretical frameworks.",
        "I like analysing data and building efficient solutions to difficult problems.",
        "I am drawn to long range strategy and enjoy working alone on intellectually demanding tasks.",
        "I find satisfaction in mastering complex subjects and applying them to build better systems.",
        "I prefer solitude when thinking and tend to plan everything carefully before acting.",
        "I enjoy setting ambitious goals and working methodically toward them without distraction.",
        "I like to understand the underlying logic of systems and improve them through careful analysis.",
    ],
    "INTP": [
        "I love exploring abstract ideas and questioning assumptions about how things work.",
        "I enjoy logic puzzles theoretical debates and understanding underlying principles.",
        "I like to analyse systems and find innovative conceptual solutions.",
        "I am fascinated by theories and enjoy spending hours thinking through complex problems alone.",
        "I prefer to understand the why behind everything rather than just following instructions.",
        "I enjoy debating ideas and exploring multiple logical frameworks before reaching a conclusion.",
        "I like working independently on open ended intellectual problems with no fixed answer.",
        "I find routine boring and prefer exploring new concepts and unconventional approaches.",
    ],
    "INFJ": [
        "I am deeply empathetic and enjoy helping others achieve their personal growth goals.",
        "I like meaningful conversations about values and making a positive impact on society.",
        "I am intuitive about people feelings and enjoy counselling and guiding others.",
        "I prefer deep one on one conversations over large social gatherings.",
        "I am driven by a sense of purpose and want my work to contribute to something meaningful.",
        "I often sense what others are feeling before they say it and try to support them quietly.",
        "I enjoy writing and reflecting on ideas that connect personal values to broader human themes.",
        "I like helping individuals find clarity and direction in their lives through thoughtful guidance.",
    ],
    "INFP": [
        "I value creativity self expression and authentic connections with people around me.",
        "I enjoy writing art and exploring emotions through creative and imaginative projects.",
        "I care deeply about helping others and expressing my inner values through my work.",
        "I am idealistic and often imagine how the world could be better and more compassionate.",
        "I prefer working on projects that feel personally meaningful rather than purely practical.",
        "I enjoy quiet reflection and expressing my feelings through creative writing or art.",
        "I am sensitive to the emotions of others and try to create harmony in my relationships.",
        "I like exploring philosophical questions about identity meaning and human connection.",
    ],
    "ISTJ": [
        "I am reliable organised and prefer clear procedures and structured environments.",
        "I like following established methods maintaining accurate records and meeting deadlines.",
        "I value responsibility consistency and working systematically through detailed tasks.",
        "I prefer predictable routines and feel most productive when I have a clear plan to follow.",
        "I take my commitments seriously and always follow through on what I promise.",
        "I like working with facts and data and prefer proven methods over experimental approaches.",
        "I enjoy maintaining order and ensuring that processes run smoothly and efficiently.",
        "I am thorough and detail oriented and I rarely make mistakes when I follow a structured process.",
    ],
    "ISTP": [
        "I enjoy hands on problem solving working with tools and understanding how things work.",
        "I like troubleshooting technical systems and finding practical efficient solutions.",
        "I prefer working independently with flexible spontaneous approach to challenges.",
        "I am calm under pressure and enjoy figuring out how mechanical or technical systems operate.",
        "I like to observe situations carefully before acting and prefer practical solutions over theory.",
        "I enjoy working with my hands and find satisfaction in fixing or building physical things.",
        "I prefer to stay flexible and respond to situations as they arise rather than planning ahead.",
        "I am independent and self reliant and I prefer to solve problems on my own terms.",
    ],
    "ISFJ": [
        "I am caring supportive and dedicated to helping others in practical everyday ways.",
        "I enjoy providing care nurturing environments and maintaining harmony in teams.",
        "I like routine and dependable work that allows me to support and protect others.",
        "I am attentive to the needs of others and enjoy making people feel comfortable and cared for.",
        "I prefer stable predictable environments where I can focus on helping those around me.",
        "I take pride in being dependable and always showing up for the people who count on me.",
        "I enjoy organising and maintaining systems that help others function smoothly.",
        "I am patient and thorough and I like to make sure everyone feels included and supported.",
    ],
    "ISFP": [
        "I enjoy artistic expression working at my own pace and experiencing the present moment.",
        "I like creative hands on activities and expressing my feelings through visual art.",
        "I value kindness beauty and helping others in quiet behind the scenes ways.",
        "I am gentle and easy going and I prefer to avoid conflict and keep things harmonious.",
        "I enjoy spending time in nature and find inspiration in the beauty of everyday experiences.",
        "I like working on creative projects that allow me to express my personal aesthetic.",
        "I prefer to act on my feelings rather than follow rigid plans or schedules.",
        "I am observant and sensitive and I notice small details that others often overlook.",
    ],
    "ENTJ": [
        "I excel at leading teams setting ambitious goals and driving strategic execution.",
        "I enjoy managing complex projects making decisive choices and inspiring leadership.",
        "I like organising people resources and processes to achieve maximum efficiency.",
        "I am confident and decisive and I enjoy taking charge of situations that require strong leadership.",
        "I like setting high standards and pushing teams to achieve results that exceed expectations.",
        "I enjoy strategic planning and find satisfaction in turning ambitious visions into reality.",
        "I am direct and assertive and I prefer to make decisions quickly based on logic and data.",
        "I thrive in competitive environments and enjoy the challenge of leading large scale initiatives.",
    ],
    "ENTP": [
        "I love debating new ideas challenging conventional thinking and exploring possibilities.",
        "I enjoy brainstorming creative solutions and arguing multiple perspectives on problems.",
        "I like entrepreneurial ventures innovation and rapidly generating novel concepts.",
        "I am energised by intellectual debate and enjoy challenging assumptions with counterarguments.",
        "I like exploring unconventional ideas and finding creative ways to solve complex problems.",
        "I enjoy working with others to brainstorm and prototype new concepts quickly.",
        "I am adaptable and thrive in fast changing environments where I can experiment freely.",
        "I like to question the status quo and find better more innovative ways of doing things.",
    ],
    "ENFJ": [
        "I am passionate about inspiring others mentoring teams and facilitating group growth.",
        "I love building relationships supporting personal development and leading collaboratively.",
        "I enjoy teaching coaching and helping communities achieve shared meaningful goals.",
        "I am energised by connecting with people and helping them reach their full potential.",
        "I enjoy leading groups and creating environments where everyone feels valued and motivated.",
        "I like facilitating discussions and helping teams work through challenges collaboratively.",
        "I am empathetic and charismatic and I naturally take on the role of mentor or guide.",
        "I find deep satisfaction in seeing others grow and succeed because of my support.",
    ],
    "ENFP": [
        "I am enthusiastic creative and love connecting people with exciting new opportunities.",
        "I enjoy exploring possibilities meeting diverse people and inspiring others with ideas.",
        "I like brainstorming campaigns advocacy and energising teams with optimistic vision.",
        "I am curious and open minded and I love learning about new ideas and meeting new people.",
        "I enjoy working on creative projects that allow me to express my imagination and enthusiasm.",
        "I am energised by social interaction and love inspiring others with my ideas and energy.",
        "I like exploring many different possibilities and I often pursue multiple interests at once.",
        "I am spontaneous and adaptable and I thrive in environments that allow for creative freedom.",
    ],
    "ESTJ": [
        "I am organised decisive and excellent at managing teams to meet operational goals.",
        "I like establishing clear processes enforcing standards and delivering reliable results.",
        "I enjoy leading projects coordinating logistics and maintaining productive structured teams.",
        "I prefer clear hierarchies and well defined roles and I hold myself and others accountable.",
        "I am practical and results oriented and I like to get things done efficiently and on time.",
        "I enjoy managing operations and ensuring that systems and processes run smoothly.",
        "I like setting clear expectations and following through with consistent disciplined execution.",
        "I am direct and confident and I prefer straightforward communication over ambiguity.",
    ],
    "ESTP": [
        "I thrive in fast paced environments and excel at quick pragmatic problem solving.",
        "I enjoy action oriented challenges negotiation and persuading others with direct energy.",
        "I like taking risks adapting quickly and solving immediate real world problems.",
        "I am bold and energetic and I enjoy jumping into action without overthinking.",
        "I like working in dynamic environments where I can respond quickly to changing situations.",
        "I enjoy persuading others and thrive in competitive high stakes situations.",
        "I am observant and resourceful and I find practical solutions to problems on the spot.",
        "I prefer hands on experience over theory and I learn best by doing.",
    ],
    "ESFJ": [
        "I care about community harmony and enjoy organising social events and supporting others.",
        "I like working with people creating welcoming environments and fulfilling social duties.",
        "I enjoy caring professions where I can support and nurture others in practical ways.",
        "I am warm and sociable and I enjoy making others feel welcome and appreciated.",
        "I like organising group activities and ensuring that everyone feels included and valued.",
        "I am attentive to the needs of others and enjoy providing practical support and care.",
        "I prefer structured environments where I can contribute to the wellbeing of a community.",
        "I find satisfaction in maintaining harmony and helping groups work together effectively.",
    ],
    "ESFP": [
        "I am spontaneous energetic and love entertaining and engaging with people around me.",
        "I enjoy performing collaborating in lively teams and bringing fun to every situation.",
        "I like people centred work that lets me express enthusiasm and bring joy to others.",
        "I am outgoing and playful and I love being the centre of attention in social settings.",
        "I enjoy living in the moment and bringing energy and excitement to everything I do.",
        "I like working with people and find satisfaction in making others laugh and feel good.",
        "I am adaptable and spontaneous and I prefer variety and excitement over routine.",
        "I thrive in lively social environments and enjoy collaborating with enthusiastic teams.",
    ],
}


def generate_features(mbti_type: str, n_samples: int) -> np.ndarray:
    """Generate n Likert-scale (1-5) feature rows for a given MBTI type.

    std=1.1 creates realistic overlap between adjacent types (e.g. INTJ vs INTP
    differ only on J/P, so their Q12-Q15 distributions will overlap meaningfully).
    This forces the model to learn genuine decision boundaries rather than
    memorising perfectly separated clusters.
    """
    means = MBTI_PROFILES[mbti_type]
    rows = []
    for _ in range(n_samples):
        row = []
        # Q1-Q4: E/I dimension
        for _ in range(4):
            val = np.random.normal(means[0], 1.1)
            row.append(int(np.clip(round(val), 1, 5)))
        # Q5-Q8: N/S dimension
        for _ in range(4):
            val = np.random.normal(means[1], 1.1)
            row.append(int(np.clip(round(val), 1, 5)))
        # Q9-Q11: T/F dimension
        for _ in range(3):
            val = np.random.normal(means[2], 1.1)
            row.append(int(np.clip(round(val), 1, 5)))
        # Q12-Q15: J/P dimension
        for _ in range(4):
            val = np.random.normal(means[3], 1.1)
            row.append(int(np.clip(round(val), 1, 5)))
        rows.append(row)
    return np.array(rows)


def build_dataset(samples_per_class: int = 1000) -> pd.DataFrame:
    records = []
    for mbti_type in MBTI_PROFILES:
        features = generate_features(mbti_type, samples_per_class)
        texts = TEXT_SAMPLES[mbti_type]
        for i, feat_row in enumerate(features):
            text = texts[i % len(texts)]
            record = {f"Q{j+1}": feat_row[j] for j in range(15)}
            record["mbti_type"] = mbti_type
            record["text_sample"] = text
            records.append(record)
    df = pd.DataFrame(records).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


# ── 2. SKILLS ────────────────────────────────────────────────────────────────
SKILLS = {
    "Technical": [
        "Python", "Java", "SQL", "Machine Learning", "Data Analysis",
        "Cloud Computing", "Networking", "Cybersecurity", "Web Development", "Statistics"
    ],
    "Interpersonal": [
        "Communication", "Leadership", "Teamwork", "Negotiation", "Mentoring",
        "Conflict Resolution", "Public Speaking", "Customer Service", "Empathy", "Collaboration"
    ],
    "Creative": [
        "Design Thinking", "Content Creation", "Storytelling", "Visual Design", "Innovation",
        "Problem Solving", "Creativity", "Writing", "Research", "Prototyping"
    ],
    "Analytical": [
        "Critical Thinking", "Data Interpretation", "Financial Analysis", "Risk Assessment",
        "Project Planning", "Decision Making", "Systems Thinking", "Research Analysis",
        "Forecasting", "Process Optimization"
    ]
}

ALL_SKILLS = [s for skills in SKILLS.values() for s in skills]


# ── 3. CAREERS ───────────────────────────────────────────────────────────────
CAREERS = [
    # Technology
    {"name": "Software Engineer",       "domain": "Technology",    "mbti_match": ["INTJ","INTP","ENTP","ISTP","ENTJ"],  "required_skills": ["Python","Java","Problem Solving","Critical Thinking","Teamwork"],         "soft_skills": ["Communication","Teamwork"]},
    {"name": "Data Scientist",          "domain": "Technology",    "mbti_match": ["INTP","INTJ","ENTP","ISTJ","INFJ"],  "required_skills": ["Python","Machine Learning","Statistics","Data Analysis","Research"],       "soft_skills": ["Critical Thinking","Communication"]},
    {"name": "Cybersecurity Analyst",   "domain": "Technology",    "mbti_match": ["ISTJ","INTJ","ISTP","INTP","ESTJ"],  "required_skills": ["Cybersecurity","Networking","Risk Assessment","Critical Thinking","Python"],"soft_skills": ["Attention to Detail","Problem Solving"]},
    {"name": "DevOps Engineer",         "domain": "Technology",    "mbti_match": ["ISTP","INTP","ENTP","INTJ","ESTP"],  "required_skills": ["Cloud Computing","Python","Networking","Systems Thinking","Collaboration"], "soft_skills": ["Adaptability","Communication"]},
    {"name": "AI Researcher",           "domain": "Technology",    "mbti_match": ["INTP","INTJ","INFJ","ENTP","INFP"],  "required_skills": ["Machine Learning","Python","Research","Statistics","Innovation"],           "soft_skills": ["Creativity","Critical Thinking"]},
    {"name": "UX Designer",             "domain": "Technology",    "mbti_match": ["INFP","ENFP","INFJ","ISFP","ENFJ"],  "required_skills": ["Design Thinking","Prototyping","Visual Design","Research","Empathy"],       "soft_skills": ["Creativity","Communication"]},
    {"name": "Product Manager",         "domain": "Technology",    "mbti_match": ["ENTJ","ENFJ","ENTP","ESTJ","INTJ"],  "required_skills": ["Project Planning","Leadership","Decision Making","Communication","Teamwork"],"soft_skills": ["Negotiation","Strategic Thinking"]},
    {"name": "Network Engineer",        "domain": "Technology",    "mbti_match": ["ISTJ","ISTP","ESTJ","INTJ","INTP"],  "required_skills": ["Networking","Cybersecurity","Systems Thinking","SQL","Cloud Computing"],    "soft_skills": ["Problem Solving","Detail Orientation"]},
    {"name": "Cloud Architect",         "domain": "Technology",    "mbti_match": ["INTJ","ENTJ","INTP","ISTJ","ENTP"],  "required_skills": ["Cloud Computing","Systems Thinking","Python","Networking","Decision Making"],"soft_skills": ["Leadership","Strategic Thinking"]},
    {"name": "Database Administrator",  "domain": "Technology",    "mbti_match": ["ISTJ","INTJ","ISTP","ESTJ","INTP"],  "required_skills": ["SQL","Data Analysis","Systems Thinking","Python","Process Optimization"],   "soft_skills": ["Precision","Reliability"]},

    # Business
    {"name": "Financial Analyst",       "domain": "Business",      "mbti_match": ["INTJ","ISTJ","ENTJ","ESTJ","INTP"],  "required_skills": ["Financial Analysis","Data Interpretation","Statistics","Excel","Forecasting"],    "soft_skills": ["Attention to Detail","Communication"]},
    {"name": "Marketing Manager",       "domain": "Business",      "mbti_match": ["ENFP","ENFJ","ENTP","ESFJ","ENTJ"],  "required_skills": ["Content Creation","Communication","Creativity","Research","Storytelling"],         "soft_skills": ["Leadership","Persuasion"]},
    {"name": "HR Manager",              "domain": "Business",      "mbti_match": ["ENFJ","ESFJ","INFJ","ISFJ","ENFP"],  "required_skills": ["Communication","Empathy","Conflict Resolution","Leadership","Mentoring"],           "soft_skills": ["Empathy","Organisation"]},
    {"name": "Business Consultant",     "domain": "Business",      "mbti_match": ["ENTJ","ENTP","INTJ","ESTJ","INTP"],  "required_skills": ["Critical Thinking","Communication","Research Analysis","Decision Making","Negotiation"],"soft_skills": ["Adaptability","Leadership"]},
    {"name": "Project Manager",         "domain": "Business",      "mbti_match": ["ESTJ","ENTJ","ISTJ","ENFJ","INTJ"],  "required_skills": ["Project Planning","Leadership","Communication","Risk Assessment","Teamwork"],       "soft_skills": ["Organisation","Decision Making"]},
    {"name": "Operations Manager",      "domain": "Business",      "mbti_match": ["ESTJ","ISTJ","ENTJ","ESTP","INTJ"],  "required_skills": ["Process Optimization","Leadership","Decision Making","Systems Thinking","Teamwork"],"soft_skills": ["Efficiency","Problem Solving"]},
    {"name": "Sales Manager",           "domain": "Business",      "mbti_match": ["ESTP","ENTJ","ENFJ","ESFJ","ENTP"],  "required_skills": ["Negotiation","Communication","Leadership","Customer Service","Persuasion"],        "soft_skills": ["Confidence","Resilience"]},
    {"name": "Supply Chain Analyst",    "domain": "Business",      "mbti_match": ["ISTJ","ESTJ","INTJ","INTP","ENTJ"],  "required_skills": ["Data Analysis","Forecasting","Process Optimization","Systems Thinking","SQL"],     "soft_skills": ["Analytical Thinking","Coordination"]},
    {"name": "Accountant",              "domain": "Business",      "mbti_match": ["ISTJ","ISFJ","ESTJ","INTJ","INTP"],  "required_skills": ["Financial Analysis","Data Interpretation","Statistics","Process Optimization","SQL"],"soft_skills": ["Precision","Reliability"]},
    {"name": "Entrepreneur",            "domain": "Business",      "mbti_match": ["ENTP","ENTJ","ENFP","ESTP","INTJ"],  "required_skills": ["Innovation","Decision Making","Leadership","Risk Assessment","Communication"],      "soft_skills": ["Resilience","Creativity"]},

    # Creative Arts
    {"name": "Graphic Designer",        "domain": "Creative Arts", "mbti_match": ["ISFP","INFP","ENFP","ISTP","ESTP"],  "required_skills": ["Visual Design","Creativity","Design Thinking","Prototyping","Innovation"],         "soft_skills": ["Attention to Detail","Communication"]},
    {"name": "Content Writer",          "domain": "Creative Arts", "mbti_match": ["INFP","INFJ","ENFP","INTP","ISFP"],  "required_skills": ["Writing","Research","Storytelling","Content Creation","Communication"],            "soft_skills": ["Creativity","Discipline"]},
    {"name": "Film Director",           "domain": "Creative Arts", "mbti_match": ["ENFJ","ENTJ","INFJ","ENFP","ENTP"],  "required_skills": ["Storytelling","Leadership","Visual Design","Creativity","Communication"],           "soft_skills": ["Vision","Collaboration"]},
    {"name": "Animator",                "domain": "Creative Arts", "mbti_match": ["ISFP","INFP","INTP","ISTP","INFJ"],  "required_skills": ["Visual Design","Creativity","Prototyping","Innovation","Design Thinking"],         "soft_skills": ["Patience","Attention to Detail"]},
    {"name": "Musician",                "domain": "Creative Arts", "mbti_match": ["ISFP","INFP","ENFP","ESFP","ENFJ"],  "required_skills": ["Creativity","Storytelling","Innovation","Communication","Collaboration"],           "soft_skills": ["Discipline","Expression"]},
    {"name": "Game Designer",           "domain": "Creative Arts", "mbti_match": ["INTP","ENTP","INFP","ISTP","INTJ"],  "required_skills": ["Design Thinking","Creativity","Python","Innovation","Prototyping"],                "soft_skills": ["Problem Solving","Teamwork"]},
    {"name": "Photographer",            "domain": "Creative Arts", "mbti_match": ["ISFP","INFP","ISTP","ESFP","ENFP"],  "required_skills": ["Visual Design","Creativity","Innovation","Storytelling","Research"],               "soft_skills": ["Observation","Patience"]},
    {"name": "Interior Designer",       "domain": "Creative Arts", "mbti_match": ["ISFJ","INFJ","ESFJ","ISFP","INFP"],  "required_skills": ["Visual Design","Design Thinking","Creativity","Research","Prototyping"],           "soft_skills": ["Empathy","Communication"]},
    {"name": "Illustrator",             "domain": "Creative Arts", "mbti_match": ["INFP","ISFP","INTP","INFJ","ISTP"],  "required_skills": ["Visual Design","Creativity","Storytelling","Innovation","Design Thinking"],        "soft_skills": ["Creativity","Attention to Detail"]},
    {"name": "Copywriter",              "domain": "Creative Arts", "mbti_match": ["ENTP","ENFP","INFP","INTP","ENTJ"],  "required_skills": ["Writing","Storytelling","Content Creation","Research","Communication"],            "soft_skills": ["Persuasion","Creativity"]},

    # Healthcare
    {"name": "Doctor",                  "domain": "Healthcare",    "mbti_match": ["INTJ","ISTJ","INFJ","ENTJ","ISFJ"],  "required_skills": ["Critical Thinking","Research","Decision Making","Communication","Empathy"],        "soft_skills": ["Empathy","Precision"]},
    {"name": "Nurse",                   "domain": "Healthcare",    "mbti_match": ["ISFJ","ESFJ","INFJ","ENFJ","ISFP"],  "required_skills": ["Empathy","Communication","Customer Service","Teamwork","Collaboration"],           "soft_skills": ["Compassion","Resilience"]},
    {"name": "Pharmacist",              "domain": "Healthcare",    "mbti_match": ["ISTJ","INTJ","ISFJ","INTP","ESTJ"],  "required_skills": ["Data Analysis","Research","Critical Thinking","Statistics","Process Optimization"],"soft_skills": ["Precision","Communication"]},
    {"name": "Medical Researcher",      "domain": "Healthcare",    "mbti_match": ["INTJ","INTP","INFJ","ISTJ","ENTP"],  "required_skills": ["Research","Statistics","Data Analysis","Machine Learning","Critical Thinking"],    "soft_skills": ["Curiosity","Precision"]},
    {"name": "Physical Therapist",      "domain": "Healthcare",    "mbti_match": ["ISFJ","ESFJ","ENFJ","INFJ","ISFP"],  "required_skills": ["Empathy","Communication","Mentoring","Collaboration","Customer Service"],         "soft_skills": ["Patience","Encouragement"]},
    {"name": "Psychologist",            "domain": "Healthcare",    "mbti_match": ["INFJ","INFP","ENFJ","ISFJ","INTJ"],  "required_skills": ["Empathy","Communication","Research","Critical Thinking","Mentoring"],             "soft_skills": ["Active Listening","Empathy"]},
    {"name": "Nutritionist",            "domain": "Healthcare",    "mbti_match": ["ISFJ","ESFJ","INFJ","ENFP","ISFP"],  "required_skills": ["Research","Communication","Empathy","Data Analysis","Customer Service"],          "soft_skills": ["Motivation","Education"]},
    {"name": "Radiologist",             "domain": "Healthcare",    "mbti_match": ["INTJ","ISTJ","INTP","ISTP","INFJ"],  "required_skills": ["Data Interpretation","Critical Thinking","Research","Statistics","Systems Thinking"],"soft_skills": ["Precision","Attention to Detail"]},
    {"name": "Surgeon",                 "domain": "Healthcare",    "mbti_match": ["INTJ","ENTJ","ISTJ","ISTP","ESTJ"],  "required_skills": ["Decision Making","Critical Thinking","Risk Assessment","Research","Teamwork"],    "soft_skills": ["Precision","Resilience"]},
    {"name": "Biomedical Engineer",     "domain": "Healthcare",    "mbti_match": ["INTJ","INTP","ISTJ","ENTP","ENTJ"],  "required_skills": ["Machine Learning","Python","Research","Statistics","Innovation"],                "soft_skills": ["Problem Solving","Precision"]},

    # Education
    {"name": "Teacher",                 "domain": "Education",     "mbti_match": ["ENFJ","ESFJ","INFJ","ISFJ","ENFP"],  "required_skills": ["Communication","Mentoring","Public Speaking","Creativity","Empathy"],             "soft_skills": ["Patience","Dedication"]},
    {"name": "Professor",               "domain": "Education",     "mbti_match": ["INTJ","INFJ","ENTP","INTP","ENTJ"],  "required_skills": ["Research","Communication","Public Speaking","Critical Thinking","Writing"],       "soft_skills": ["Expertise","Mentoring"]},
    {"name": "School Counselor",        "domain": "Education",     "mbti_match": ["INFJ","ENFJ","ISFJ","INFP","ESFJ"],  "required_skills": ["Empathy","Communication","Mentoring","Conflict Resolution","Research"],          "soft_skills": ["Active Listening","Compassion"]},
    {"name": "Curriculum Designer",     "domain": "Education",     "mbti_match": ["INTJ","INFJ","ISTJ","ENFJ","INFP"],  "required_skills": ["Research","Writing","Design Thinking","Critical Thinking","Project Planning"],   "soft_skills": ["Creativity","Organisation"]},
    {"name": "Education Researcher",    "domain": "Education",     "mbti_match": ["INTJ","INTP","INFJ","INFP","ISTJ"],  "required_skills": ["Research","Data Analysis","Writing","Statistics","Critical Thinking"],           "soft_skills": ["Curiosity","Precision"]},
    {"name": "Corporate Trainer",       "domain": "Education",     "mbti_match": ["ENFJ","ENTJ","ESFJ","ESTJ","ENFP"],  "required_skills": ["Public Speaking","Communication","Mentoring","Leadership","Storytelling"],       "soft_skills": ["Energy","Adaptability"]},
    {"name": "Librarian",               "domain": "Education",     "mbti_match": ["ISFJ","INFJ","ISTJ","INFP","ISFP"],  "required_skills": ["Research","Communication","Data Analysis","Writing","Customer Service"],         "soft_skills": ["Organisation","Patience"]},
    {"name": "Tutor",                   "domain": "Education",     "mbti_match": ["INFP","INFJ","ENFJ","ISFJ","INTP"],  "required_skills": ["Communication","Empathy","Mentoring","Critical Thinking","Research"],            "soft_skills": ["Patience","Encouragement"]},
    {"name": "Academic Advisor",        "domain": "Education",     "mbti_match": ["ENFJ","INFJ","ESFJ","ISFJ","ENFP"],  "required_skills": ["Communication","Empathy","Mentoring","Decision Making","Research Analysis"],     "soft_skills": ["Active Listening","Guidance"]},
    {"name": "Special Education Teacher","domain":"Education",     "mbti_match": ["ISFJ","ENFJ","INFJ","ESFJ","ISFP"],  "required_skills": ["Empathy","Communication","Patience","Mentoring","Collaboration"],                "soft_skills": ["Compassion","Creativity"]},
]


def main():
    out = "data/raw"
    os.makedirs(out, exist_ok=True)

    # --- Dataset ---
    print("Generating MBTI dataset...")
    df = build_dataset(1000)
    df.to_csv(f"{out}/mbti_dataset.csv", index=False)
    print(f"  Saved {len(df)} rows → {out}/mbti_dataset.csv")
    print(f"  Class distribution:\n{df['mbti_type'].value_counts().to_string()}\n")

    # --- Skills ---
    with open(f"{out}/skills.json", "w") as f:
        json.dump(SKILLS, f, indent=2)
    print(f"Saved {len(ALL_SKILLS)} skills → {out}/skills.json")

    # --- Careers ---
    with open(f"{out}/careers.json", "w") as f:
        json.dump(CAREERS, f, indent=2)
    print(f"Saved {len(CAREERS)} careers → {out}/careers.json")
    print("\nAll data generated successfully!")


if __name__ == "__main__":
    main()