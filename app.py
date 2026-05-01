"""Streamlit interface for the Personality-Based Career Recommendation System."""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from modules import (
    find_recommendation_path,
    optimize_recommendations_hill_climbing,
    predict_mbti_personality,
    solve_career_constraints,
)


st.set_page_config(page_title="CareerBloom", layout="wide")

PAGE_LABELS = [
    "Start",
    "About You",
    "How You Think",
    "What Fits",
    "Results",
]

PROFILE_PROMPT = (
    "Tell me what you're like on a normal day. Do you like working alone, or bouncing ideas off people? "
    "What kinds of problems pull you in? When you make decisions, what do you trust most?"
)

PROFILE_PROMPT_LINES = [
    "What do you naturally enjoy doing when nobody is assigning you work?",
    "Do you like working alone, or bouncing ideas off people?",
    "What kinds of problems or topics keep your attention the longest?",
    "When you have to decide, do you lean on logic, gut feel, structure, or something else?",
]

PAGE_PROGRESS_NOTES = [
    "A quick five-step read on how you work best.",
    "Nice, this helps a lot already.",
    "Trust your first instinct here.",
    "Almost there. This last step sharpens the match.",
    "Here is the shape that came through.",
]

QUESTION_SECTIONS: list[tuple[str, list[tuple[str, str]]]] = [
    (
        "Social Energy",
        [
            ("Q1", "I enjoy being the center of attention in social situations"),
            ("Q2", "I feel energized after spending time with a large group of people"),
            ("Q3", "I prefer working in teams over working alone"),
            ("Q4", "I find it easy to start conversations with strangers"),
        ],
    ),
    (
        "Ideas and Attention",
        [
            ("Q5", "I prefer focusing on facts and concrete details over abstract ideas"),
            ("Q6", "I enjoy thinking about theories and possibilities"),
            ("Q7", "I tend to think about the big picture rather than specific details"),
            ("Q8", "I am drawn to creative and imaginative thinking"),
        ],
    ),
    (
        "How You Decide",
        [
            ("Q9", "I make decisions based on logic rather than feelings"),
            ("Q10", "I prioritize fairness and harmony over cold hard facts"),
            ("Q11", "I find it easy to stay objective when making tough decisions"),
        ],
    ),
    (
        "Planning Rhythm",
        [
            ("Q12", "I like to have a clear plan before starting a task"),
            ("Q13", "I prefer structure and routine over spontaneity"),
            ("Q14", "I like to finish one task completely before starting another"),
            ("Q15", "I prefer making decisions early rather than keeping options open"),
        ],
    ),
]

QUESTION_SECTION_DESCRIPTIONS = {
    "Social Energy": "Think about an ordinary week, not your best social day.",
    "Ideas and Attention": "Go with the kind of thinking that feels most natural, not most impressive.",
    "How You Decide": "This is about your default instinct under pressure.",
    "Planning Rhythm": "Pick the pace and structure that usually feels relieving, not restrictive.",
}

QUESTION_ANCHORS = {
    "Q1": ("Prefer the background", "Enjoy the spotlight"),
    "Q2": ("Crowds drain me", "Crowds energize me"),
    "Q3": ("Prefer working alone", "Prefer teams"),
    "Q4": ("Slow to warm up", "Easy with strangers"),
    "Q5": ("Concrete details", "Abstract ideas"),
    "Q6": ("Practical and literal", "Theories and possibilities"),
    "Q7": ("Specific details", "Big picture"),
    "Q8": ("Grounded and literal", "Creative and imaginative"),
    "Q9": ("Lead with feelings", "Lead with logic"),
    "Q10": ("Cold hard facts", "Fairness and harmony"),
    "Q11": ("Personal and emotional", "Objective and steady"),
    "Q12": ("Prefer to improvise", "Like a clear plan"),
    "Q13": ("Spontaneous flow", "Structure and routine"),
    "Q14": ("Juggle a few things", "Finish one thing first"),
    "Q15": ("Keep options open", "Decide early"),
}

DOMAIN_OPTIONS = ["Technology", "Healthcare", "Business", "Finance", "Education", "Creative"]
DOMAIN_ALIASES = {
    "Technology": "Technology",
    "Healthcare": "Healthcare",
    "Business": "Business",
    "Finance": "Business",
    "Education": "Education",
    "Creative": "Creative Arts",
}
REVERSE_SCORED_QUESTIONS = {"Q5", "Q10"}
ROOT_DIR = Path(__file__).resolve().parent
MODEL_ARTIFACT_PATHS = [
    ROOT_DIR / "models" / "text_model.pkl",
    ROOT_DIR / "models" / "questionnaire_model.pkl",
    ROOT_DIR / "models" / "questionnaire_dimension_model.pkl",
    ROOT_DIR / "models" / "module1_mnb.pkl",
]

MBTI_DESCRIPTIONS = {
    "INTJ": "Strategic, independent, and future-focused. INTJs usually enjoy solving complex problems with long-range plans.",
    "INTP": "Curious, analytical, and idea-driven. INTPs often enjoy systems, theories, and open-ended problem solving.",
    "ENTJ": "Decisive, ambitious, and organized. ENTJs tend to lead with structure and big-picture execution.",
    "ENTP": "Inventive, energetic, and flexible. ENTPs often thrive on novelty, debate, and creative problem solving.",
    "INFJ": "Insightful, thoughtful, and purpose-driven. INFJs often mix empathy with long-term vision.",
    "INFP": "Reflective, imaginative, and values-led. INFPs are often drawn to meaningful, creative work.",
    "ENFJ": "Supportive, expressive, and motivating. ENFJs often enjoy guiding people and building strong teams.",
    "ENFP": "Curious, enthusiastic, and people-oriented. ENFPs often shine in creative, exploratory environments.",
    "ISTJ": "Reliable, practical, and detail-aware. ISTJs often prefer clarity, consistency, and follow-through.",
    "ISFJ": "Steady, caring, and dependable. ISFJs often bring patience, structure, and strong support to others.",
    "ESTJ": "Direct, structured, and results-focused. ESTJs often enjoy responsibility, planning, and execution.",
    "ESFJ": "Warm, organized, and community-minded. ESFJs often prioritize teamwork, harmony, and dependability.",
    "ISTP": "Hands-on, logical, and adaptable. ISTPs often like practical challenges and figuring things out in real time.",
    "ISFP": "Creative, calm, and observant. ISFPs often prefer meaningful work with room for personal expression.",
    "ESTP": "Action-oriented, bold, and practical. ESTPs often enjoy fast feedback, variety, and real-world decisions.",
    "ESFP": "Outgoing, spontaneous, and energetic. ESFPs often thrive in lively settings with people and creativity.",
}


def inject_styles() -> None:
    """Apply a warmer, less templated visual system."""
    st.markdown(
        """
        <style>
            :root {
                --page-top: #f4ede4;
                --page-bottom: #ede2d4;
                --paper: rgba(255, 250, 244, 0.92);
                --panel: rgba(247, 240, 231, 0.88);
                --ink: #2d241f;
                --muted: #6f6258;
                --line: rgba(117, 95, 77, 0.16);
                --line-strong: rgba(95, 74, 59, 0.28);
                --accent: #6f8573;
                --accent-strong: #506556;
                --accent-soft: #dce7dd;
                --coral: #d38e74;
                --cocoa: #6a4a39;
                --butter: #efdca8;
                --shadow: rgba(58, 41, 29, 0.10);
            }

            .stApp {
                background:
                    radial-gradient(circle at top right, rgba(111, 133, 115, 0.12), transparent 26%),
                    radial-gradient(circle at top left, rgba(211, 142, 116, 0.12), transparent 20%),
                    linear-gradient(180deg, var(--page-top) 0%, #f6f0e8 40%, var(--page-bottom) 100%);
                color: var(--ink);
            }

            .block-container {
                max-width: 1220px;
                padding-top: 1.4rem;
                padding-bottom: 4rem;
                padding-left: 1.25rem;
                padding-right: 1.25rem;
            }

            html, body, [class*="css"] {
                font-family: "Aptos", "Segoe UI", "Helvetica Neue", sans-serif;
                font-size: 16px;
            }

            h1, h2, h3 {
                font-family: Georgia, "Times New Roman", serif;
                letter-spacing: -0.02em;
                color: var(--ink);
                line-height: 1.04;
            }

            h1 {
                font-size: clamp(2.4rem, 5.2vw, 4.2rem);
            }

            h2 {
                font-size: clamp(1.7rem, 3.6vw, 2.7rem);
            }

            h3 {
                font-size: clamp(1.22rem, 2.4vw, 1.75rem);
            }

            p, label, li, span, div[data-testid="stMarkdownContainer"] p {
                font-size: 1.03rem;
                line-height: 1.68;
            }

            .hero-card h1, .result-card h3, .site-header h3 {
                margin-top: 0;
            }

            .site-header {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 1.3rem;
                margin-bottom: 0.9rem;
                padding-bottom: 1rem;
                border-bottom: 1px solid var(--line);
            }

            .brand-wrap {
                display: flex;
                align-items: flex-start;
                gap: 1rem;
                min-width: 0;
            }

            .brand-mark {
                width: 58px;
                height: 58px;
                border-radius: 20px;
                background: linear-gradient(145deg, rgba(111, 133, 115, 0.22), rgba(211, 142, 116, 0.26));
                border: 1px solid rgba(106, 74, 57, 0.18);
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 0.98rem;
                font-weight: 700;
                letter-spacing: 0.08em;
                position: relative;
                flex-shrink: 0;
                color: var(--cocoa);
            }

            .brand-mark::before {
                content: "";
                width: 11px;
                height: 11px;
                border-radius: 999px;
                background: var(--accent);
                border: 1px solid rgba(80, 101, 86, 0.28);
                position: absolute;
                top: -3px;
                right: -3px;
            }

            .brand-name {
                font-family: Georgia, "Times New Roman", serif;
                font-size: 1.52rem;
                color: var(--ink);
                line-height: 1.1;
            }

            .brand-tag {
                color: var(--muted);
                font-size: 0.98rem;
                margin-top: 0.22rem;
                max-width: 42rem;
            }

            .header-nav {
                display: flex;
                align-items: center;
                justify-content: flex-end;
                gap: 0.55rem;
                flex-wrap: wrap;
            }

            .nav-chip {
                border: 1px solid var(--line);
                border-radius: 999px;
                padding: 0.48rem 0.86rem;
                background: rgba(255, 250, 244, 0.7);
                color: var(--muted);
                font-size: 0.92rem;
                white-space: nowrap;
            }

            .nav-chip.active {
                background: var(--accent-soft);
                border-color: rgba(80, 101, 86, 0.26);
                color: var(--ink);
                font-weight: 600;
            }

            .eyebrow {
                text-transform: uppercase;
                font-size: 0.76rem;
                letter-spacing: 0.15em;
                color: #7e6c60;
                font-weight: 700;
                margin-bottom: 0.48rem;
            }

            .lead {
                color: var(--muted);
                line-height: 1.72;
                font-size: 1.08rem;
            }

            .progress-wrap {
                margin: 0.2rem 0 1.7rem 0;
            }

            .progress-meta {
                display: flex;
                align-items: flex-end;
                justify-content: space-between;
                gap: 1rem;
                margin-bottom: 0.85rem;
            }

            .progress-title {
                font-family: Georgia, "Times New Roman", serif;
                font-size: 1.28rem;
                color: var(--ink);
                line-height: 1.15;
            }

            .progress-note {
                color: var(--muted);
                font-size: 0.96rem;
                max-width: 28rem;
                text-align: right;
            }

            .step-strip {
                display: flex;
                gap: 0.5rem;
                flex-wrap: wrap;
                margin: 0.7rem 0 0 0;
            }

            .step-chip {
                border: 1px solid var(--line);
                border-radius: 999px;
                padding: 0.38rem 0.78rem;
                background: rgba(255, 250, 244, 0.56);
                color: var(--muted);
                font-size: 0.9rem;
            }

            .step-chip.active {
                background: rgba(111, 133, 115, 0.16);
                border-color: rgba(80, 101, 86, 0.22);
                color: var(--ink);
                font-weight: 600;
            }

            .small-muted {
                color: var(--muted);
                font-size: 0.97rem;
                line-height: 1.62;
            }

            .chain {
                font-size: 1.08rem;
                font-weight: 700;
                color: var(--ink);
                line-height: 1.8;
            }

            .progress-shell {
                height: 12px;
                border-radius: 999px;
                border: 1px solid rgba(95, 74, 59, 0.12);
                background: rgba(255, 250, 244, 0.72);
                overflow: hidden;
                box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.7);
            }

            .progress-fill {
                height: 100%;
                background: linear-gradient(90deg, var(--coral), var(--accent));
                border-radius: 999px;
                transition: width 0.2s ease;
            }

            div[data-testid="stMetric"] {
                background: rgba(255, 250, 244, 0.78);
                border: 1px solid var(--line);
                border-radius: 20px;
                padding: 0.95rem 1rem;
                box-shadow: 0 12px 22px var(--shadow);
            }

            .stButton > button {
                border-radius: 16px !important;
                border: 1px solid var(--line-strong) !important;
                background: rgba(255, 250, 244, 0.72) !important;
                color: var(--ink) !important;
                font-weight: 600 !important;
                font-size: 1rem !important;
                padding: 0.82rem 1.24rem !important;
                min-height: 50px !important;
                box-shadow: 0 8px 18px rgba(58, 41, 29, 0.06) !important;
                transition: transform 0.18s ease, box-shadow 0.18s ease, background 0.18s ease !important;
            }

            .stButton > button:hover {
                transform: translateY(-2px);
                box-shadow: 0 14px 22px rgba(58, 41, 29, 0.10) !important;
            }

            .stButton > button:active {
                transform: translateY(1px);
            }

            .stButton > button[kind="primary"] {
                background: linear-gradient(180deg, #5e7563, #4d6353) !important;
                border-color: #455948 !important;
                color: #fff9f2 !important;
                box-shadow: 0 14px 28px rgba(80, 101, 86, 0.24) !important;
            }

            .stButton > button[kind="primary"]:hover {
                background: linear-gradient(180deg, #556b5a, #465846) !important;
                box-shadow: 0 18px 32px rgba(80, 101, 86, 0.28) !important;
            }

            .stButton > button[kind="secondary"] {
                background: rgba(255, 250, 244, 0.68) !important;
                color: var(--ink) !important;
            }

            .career-title {
                font-family: Georgia, "Times New Roman", serif;
                font-size: 1.28rem;
                margin-bottom: 0.4rem;
                color: var(--ink);
            }

            .hero-split {
                display: grid;
                grid-template-columns: minmax(0, 1.55fr) minmax(270px, 0.92fr);
                gap: 1.4rem;
                align-items: start;
            }

            .hero-card {
                position: relative;
                overflow: hidden;
                border: 1px solid rgba(95, 74, 59, 0.12);
                border-radius: 34px;
                padding: 2rem 2rem 1.85rem 2rem;
                background: linear-gradient(135deg, rgba(255, 250, 244, 0.92), rgba(247, 240, 231, 0.86));
                box-shadow: 0 18px 38px var(--shadow);
            }

            .hero-card::before {
                content: "";
                position: absolute;
                right: -60px;
                top: -80px;
                width: 220px;
                height: 220px;
                border-radius: 999px;
                background: radial-gradient(circle, rgba(111, 133, 115, 0.22), transparent 70%);
            }

            .hero-card::after {
                content: "";
                position: absolute;
                left: -30px;
                bottom: -90px;
                width: 180px;
                height: 180px;
                border-radius: 999px;
                background: radial-gradient(circle, rgba(211, 142, 116, 0.18), transparent 68%);
            }

            .hero-aside {
                background: rgba(248, 241, 233, 0.82);
                border: 1px solid rgba(95, 74, 59, 0.10);
                border-radius: 22px;
                padding: 1.15rem 1.1rem;
                box-shadow: 0 10px 24px rgba(58, 41, 29, 0.05);
            }

            .pill-row {
                display: flex;
                gap: 0.6rem;
                flex-wrap: wrap;
                margin-top: 1.15rem;
            }

            .soft-pill {
                border-radius: 999px;
                padding: 0.5rem 0.84rem;
                border: 1px solid var(--line);
                font-size: 0.92rem;
                color: var(--ink);
                background: rgba(255, 250, 244, 0.82);
            }

            .soft-pill.rose { background: rgba(211, 142, 116, 0.15); }
            .soft-pill.sage { background: rgba(111, 133, 115, 0.16); }
            .soft-pill.sky { background: rgba(224, 214, 198, 0.48); }
            .soft-pill.butter { background: rgba(239, 220, 168, 0.30); }

            .mini-list {
                margin: 0.8rem 0 0 0;
                padding-left: 1rem;
                color: var(--muted);
            }

            .mini-list li {
                margin-bottom: 0.48rem;
            }

            .mini-stat-grid {
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 0.7rem;
                margin-top: 1rem;
            }

            .mini-stat {
                background: rgba(255, 250, 244, 0.7);
                border-radius: 18px;
                padding: 0.9rem;
                border: 1px solid rgba(95, 74, 59, 0.08);
            }

            .mini-stat strong {
                display: block;
                font-size: 0.86rem;
                text-transform: uppercase;
                letter-spacing: 0.12em;
                color: #7a6b61;
                margin-bottom: 0.3rem;
            }

            .mini-stat span {
                color: var(--ink);
                font-size: 1rem;
                line-height: 1.45;
            }

            .section-intro {
                display: grid;
                grid-template-columns: minmax(0, 1.5fr) minmax(240px, 0.8fr);
                gap: 1.5rem;
                align-items: start;
                margin: 0.25rem 0 1.35rem 0;
            }

            .section-copy {
                max-width: 48rem;
            }

            .section-copy h2 {
                margin: 0 0 0.55rem 0;
            }

            .section-aside {
                padding: 1rem 1rem 0.95rem 1rem;
                background: rgba(255, 250, 244, 0.58);
                border-left: 3px solid rgba(111, 133, 115, 0.45);
                border-radius: 0 18px 18px 0;
            }

            .section-marker {
                margin: 1.8rem 0 0.9rem 0;
                padding-top: 1.2rem;
                border-top: 1px solid var(--line);
            }

            [data-testid="stTextInputRoot"] input,
            [data-testid="stTextArea"] textarea,
            [data-baseweb="select"] > div,
            [data-baseweb="tag"] {
                background: rgba(255, 250, 244, 0.88) !important;
                border-color: rgba(95, 74, 59, 0.14) !important;
                border-radius: 18px !important;
                color: var(--ink) !important;
                font-size: 1rem !important;
                box-shadow: 0 4px 14px rgba(58, 41, 29, 0.03);
            }

            [data-testid="stTextArea"] textarea {
                min-height: 18rem;
                line-height: 1.7;
            }

            [data-testid="stTextInputRoot"] label,
            [data-testid="stTextArea"] label,
            .stMultiSelect label {
                font-size: 0.92rem !important;
                color: var(--muted) !important;
                letter-spacing: 0.02em;
            }

            .prompt-card,
            .reaction-card,
            .note-card,
            .question-shell,
            .result-card {
                border-radius: 22px;
            }

            .prompt-card,
            .reaction-card,
            .note-card {
                background: rgba(255, 250, 244, 0.72);
                border: 1px solid rgba(95, 74, 59, 0.10);
                padding: 1rem 1.05rem;
            }

            .reaction-card {
                background: rgba(220, 231, 221, 0.44);
                border-color: rgba(80, 101, 86, 0.18);
            }

            .question-shell {
                margin: 0.75rem 0 0.3rem 0;
                padding: 0.95rem 1rem 0.85rem 1rem;
                background: rgba(255, 250, 244, 0.58);
                border: 1px solid rgba(95, 74, 59, 0.08);
            }

            .question-text {
                font-size: 1.02rem;
                color: var(--ink);
                line-height: 1.55;
                margin-bottom: 0.68rem;
            }

            .slider-anchor-row,
            .slider-response {
                display: flex;
                justify-content: space-between;
                gap: 1rem;
                align-items: center;
            }

            .slider-anchor-row span {
                color: var(--muted);
                font-size: 0.9rem;
            }

            .slider-response {
                margin: 0.3rem 0 1.05rem 0;
            }

            .slider-response span {
                color: var(--muted);
                font-size: 0.84rem;
            }

            .slider-response strong {
                color: var(--cocoa);
                font-size: 0.92rem;
                font-weight: 600;
                text-align: center;
                flex: 1;
            }

            [data-testid="stSlider"] {
                margin-top: -0.05rem;
            }

            [data-testid="stSlider"] [data-baseweb="slider"] {
                padding-top: 0.15rem;
                padding-bottom: 0.25rem;
            }

            [data-testid="stSlider"] [role="slider"] {
                width: 20px !important;
                height: 20px !important;
                background: var(--accent-strong) !important;
                border: 3px solid #fffaf4 !important;
                box-shadow: 0 6px 12px rgba(80, 101, 86, 0.24) !important;
            }

            [data-testid="stMarkdownContainer"] table,
            .stTable table {
                border-collapse: separate;
                border-spacing: 0;
                width: 100%;
                background: rgba(255, 250, 244, 0.8);
                border: 1px solid rgba(95, 74, 59, 0.10);
                border-radius: 20px;
                overflow: hidden;
            }

            .stTable th,
            .stTable td {
                border-bottom: 1px solid rgba(95, 74, 59, 0.08);
                padding: 0.85rem 0.95rem;
                text-align: left;
            }

            .stAlert {
                border-radius: 18px;
                border: 1px solid rgba(95, 74, 59, 0.10);
                background: rgba(255, 250, 244, 0.76);
            }

            .result-card {
                background: rgba(255, 250, 244, 0.84);
                border: 1px solid rgba(95, 74, 59, 0.10);
                padding: 1.05rem 1.1rem;
                box-shadow: 0 12px 24px rgba(58, 41, 29, 0.05);
            }

            .result-card.accent {
                background: linear-gradient(180deg, rgba(220, 231, 221, 0.56), rgba(255, 250, 244, 0.86));
                border-color: rgba(80, 101, 86, 0.18);
            }

            .result-card.plain {
                background: rgba(248, 241, 233, 0.62);
                box-shadow: none;
            }

            .result-card.warm {
                background: linear-gradient(180deg, rgba(211, 142, 116, 0.12), rgba(255, 250, 244, 0.86));
            }

            .result-hero {
                margin-top: 0.7rem;
                margin-bottom: 0.6rem;
                padding: 1.3rem 1.3rem 1.15rem 1.3rem;
            }

            .result-hero .type-lockup {
                display: flex;
                align-items: baseline;
                gap: 0.9rem;
                flex-wrap: wrap;
                margin-bottom: 0.55rem;
            }

            .type-badge {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                min-width: 5rem;
                padding: 0.3rem 0.8rem;
                border-radius: 999px;
                background: rgba(80, 101, 86, 0.12);
                border: 1px solid rgba(80, 101, 86, 0.16);
                color: var(--cocoa);
                font-weight: 700;
                letter-spacing: 0.08em;
            }

            .subsection-heading {
                margin: 2rem 0 0.9rem 0;
            }

            .subsection-heading h3 {
                margin-bottom: 0.15rem;
            }

            @media (max-width: 900px) {
                .site-header,
                .hero-split,
                .section-intro {
                    display: grid;
                    grid-template-columns: 1fr;
                }

                .header-nav {
                    justify-content: flex-start;
                }

                .progress-meta {
                    display: grid;
                    gap: 0.45rem;
                }

                .progress-note {
                    text-align: left;
                }

                .brand-name {
                    font-size: 1.28rem;
                }

                .block-container {
                    padding-left: 0.95rem;
                    padding-right: 0.95rem;
                }

                .hero-card {
                    padding: 1.5rem 1.25rem;
                }

                .mini-stat-grid {
                    grid-template-columns: 1fr;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_state() -> None:
    """Seed session state for multi-step navigation."""
    defaults: dict[str, Any] = {
        "page": 0,
        "profile_text": "",
        "skills_text": "",
        "preferred_domains": [],
        "blocked_roles_text": "",
        "current_role": "",
        "analysis_result": None,
        "analysis_signature": "",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

    for index in range(1, 16):
        st.session_state.setdefault(f"Q{index}", 3)


def reset_form() -> None:
    """Clear all saved answers and return the app to its initial state."""
    st.session_state.page = 0
    st.session_state.profile_text = ""
    st.session_state.skills_text = ""
    st.session_state.preferred_domains = []
    st.session_state.blocked_roles_text = ""
    st.session_state.current_role = ""
    st.session_state.analysis_result = None
    st.session_state.analysis_signature = ""
    for index in range(1, 16):
        st.session_state[f"Q{index}"] = 3


def invalidate_analysis_cache() -> None:
    """Force results recomputation after any user input change."""
    st.session_state.analysis_result = None
    st.session_state.analysis_signature = ""


def set_page(page_number: int) -> None:
    """Move to a specific page and rerun the app."""
    st.session_state.page = page_number
    st.rerun()


def word_count(text: str) -> int:
    """Count words in a free-text response."""
    return len(re.findall(r"\b[\w'-]+\b", text))


def join_with_and(items: list[str]) -> str:
    """Join short phrases into a natural-language list."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def profile_reaction(text: str) -> tuple[str, str]:
    """Generate a small human reaction to the user's self-description."""
    total_words = word_count(text)
    lowered = text.lower()

    if total_words == 0:
        return "", ""
    if total_words < 12:
        return (
            "A couple more honest lines will make this feel more like you.",
            "You do not need a polished bio here. Just describe how your mind tends to work.",
        )

    signals: list[str] = []
    if any(token in lowered for token in ("alone", "independent", "quiet", "one on one", "solitude")):
        signals.append("independent")
    if any(token in lowered for token in ("abstract", "theory", "theories", "ideas", "big picture", "possibilities")):
        signals.append("idea-driven")
    if any(token in lowered for token in ("logic", "data", "analysis", "analytical", "critical thinking")):
        signals.append("analytical")
    if any(token in lowered for token in ("plan", "structure", "routine", "organized", "clear plan")):
        signals.append("structured")

    if signals:
        return (
            "This already tells me a lot about you.",
            f"You come across as {join_with_and(signals)}.",
        )

    if total_words >= 45:
        return (
            "This already gives the model a real signal.",
            "There is enough detail here to read your work style as a whole instead of guessing from fragments.",
        )

    return (
        "This is a good start.",
        "A little more detail about how you think or decide would make the read sharper.",
    )


def slider_feedback(value: int, left_anchor: str, right_anchor: str) -> str:
    """Return conversational feedback for a slider position."""
    left = left_anchor.lower()
    right = right_anchor.lower()

    if value == 1:
        return f"Strong pull toward {left}."
    if value == 2:
        return f"You lean a bit toward {left}."
    if value == 3:
        return "You sit somewhere in the middle on this one."
    if value == 4:
        return f"You lean a bit toward {right}."
    return f"Strong pull toward {right}."


def render_slider_question(question_key: str, question_text: str) -> None:
    """Render one questionnaire prompt as a 5-option radio row."""
    left_anchor, right_anchor = QUESTION_ANCHORS.get(question_key, ("Not me at all", "Very me"))

    options = ["1", "2", "3", "4", "5"]
    option_labels = {
        "1": f"1 — {left_anchor}",
        "2": "2",
        "3": "3 — Neutral",
        "4": "4",
        "5": f"5 — {right_anchor}",
    }

    current = st.session_state.get(question_key, 3)

    st.markdown(
        f'<div class="question-shell"><div class="question-text">{html.escape(question_text)}</div></div>',
        unsafe_allow_html=True,
    )

    selected = st.radio(
        question_text,
        options=options,
        format_func=lambda x: option_labels[x],
        index=options.index(str(current)),
        key=f"radio_{question_key}",
        horizontal=True,
        label_visibility="collapsed",
    )

    new_val = int(selected)
    if st.session_state.get(question_key) != new_val:
        st.session_state[question_key] = new_val
        invalidate_analysis_cache()


def humanize_mbti_summary(mbti_type: str) -> str:
    """Turn the stock MBTI description into a more personal read."""
    base = MBTI_DESCRIPTIONS.get(mbti_type, "").strip()
    if not base:
        return "You seem to have a mix of traits that does not map cleanly to one summary yet."

    first_char = base[0].lower()
    return f"You seem like someone who is {first_char}{base[1:]}"


def parse_comma_separated(text: str) -> list[str]:
    """Split list-style user input into cleaned strings."""
    values = re.split(r"[\n,;|]+", text or "")
    return [value.strip() for value in values if value.strip()]


def raw_question_payload() -> dict[str, int]:
    """Collect the raw questionnaire values from session state."""
    return {f"Q{index}": int(st.session_state.get(f"Q{index}", 3)) for index in range(1, 16)}


def scored_question_payload() -> dict[str, int]:
    """Apply reverse scoring to questionnaire items that point opposite to the model scale."""
    payload = raw_question_payload()
    for question_key in REVERSE_SCORED_QUESTIONS:
        payload[question_key] = 6 - payload[question_key]
    return payload


def normalize_domains(selected_domains: list[str]) -> set[str]:
    """Map UI domain labels to the dataset's domain labels."""
    normalized = {DOMAIN_ALIASES[domain] for domain in selected_domains if domain in DOMAIN_ALIASES}
    return normalized


def question_payload() -> dict[str, int]:
    """Return model-ready questionnaire values."""
    return scored_question_payload()


def model_artifact_signature() -> dict[str, str]:
    """Return a lightweight fingerprint for the currently loaded model files."""
    signature: dict[str, str] = {}
    for path in MODEL_ARTIFACT_PATHS:
        key = path.name
        if not path.exists():
            signature[key] = "missing"
            continue

        stat = path.stat()
        signature[key] = f"{stat.st_size}:{stat.st_mtime_ns}"
    return signature


def build_signature() -> str:
    """Create a stable signature for the current inputs and active model artifacts."""
    payload = {
        "text": st.session_state.profile_text,
        "questionnaire": raw_question_payload(),
        "scored_questionnaire": scored_question_payload(),
        "reverse_scored_questions": sorted(REVERSE_SCORED_QUESTIONS),
        "skills": st.session_state.skills_text,
        "preferred_domains": st.session_state.preferred_domains,
        "blocked_roles": st.session_state.blocked_roles_text,
        "current_role": st.session_state.current_role,
        "models": model_artifact_signature(),
    }
    return json.dumps(payload, sort_keys=True)


def mbti_dimension_difference_count(first: str | None, second: str | None) -> int | None:
    """Count how many MBTI letters differ between two 4-letter labels."""
    if not first or not second:
        return None

    left = str(first).upper().strip()
    right = str(second).upper().strip()
    if len(left) != 4 or len(right) != 4:
        return None

    return sum(1 for left_letter, right_letter in zip(left, right, strict=True) if left_letter != right_letter)


def build_prediction_consistency_note(
    prediction: dict[str, Any],
    text_prediction: dict[str, Any] | None,
    questionnaire_prediction: dict[str, Any] | None,
) -> str:
    """Explain when the supporting models disagree enough to reduce certainty."""
    final_type = prediction.get("mbti_type")
    text_type = text_prediction.get("mbti_type") if text_prediction else None
    questionnaire_type = questionnaire_prediction.get("mbti_type") if questionnaire_prediction else None

    text_vs_questionnaire = mbti_dimension_difference_count(text_type, questionnaire_type)
    final_vs_text = mbti_dimension_difference_count(final_type, text_type)
    final_vs_questionnaire = mbti_dimension_difference_count(final_type, questionnaire_type)
    final_confidence = float(prediction.get("confidence", 0.0))

    if text_vs_questionnaire is not None and text_vs_questionnaire >= 2:
        return (
            f"The text and questionnaire models disagreed on {text_vs_questionnaire} of 4 MBTI dimensions, "
            "so treat the final type as directional rather than definitive."
        )

    if (
        final_confidence < 0.7
        and (final_vs_text is not None and final_vs_text >= 2 or final_vs_questionnaire is not None and final_vs_questionnaire >= 2)
    ):
        return (
            "The combined prediction had relatively low confidence and diverged from one of the supporting models, "
            "so it is best read as a rough blend instead of a precise label."
        )

    return ""


def safe_predict(
    input_data: str | dict[str, int],
    *,
    use_hybrid: bool,
    use_dimension_voting: bool,
) -> dict[str, Any] | None:
    """Run a prediction and return None when the selected model path is unavailable."""
    try:
        return predict_mbti_personality(
            input_data,
            use_hybrid=use_hybrid,
            use_dimension_voting=use_dimension_voting,
        )
    except Exception:
        return None


def compute_results() -> dict[str, Any]:
    """Run the full prediction and recommendation pipeline for the current form state."""
    profile_text = st.session_state.profile_text.strip()
    current_role = st.session_state.current_role.strip()
    user_skills = set(parse_comma_separated(st.session_state.skills_text))
    blocked_roles = set(parse_comma_separated(st.session_state.blocked_roles_text))
    preferred_domains = normalize_domains(st.session_state.preferred_domains)
    scored_questionnaire = question_payload()

    prediction_input = {"text": profile_text, **scored_questionnaire}
    text_prediction = safe_predict(
        profile_text,
        use_hybrid=False,
        use_dimension_voting=False,
    )
    questionnaire_prediction = safe_predict(
        scored_questionnaire,
        use_hybrid=False,
        use_dimension_voting=False,
    )

    prediction_note = ""
    prediction = safe_predict(
        prediction_input,
        use_hybrid=True,
        use_dimension_voting=True,
    )
    if prediction is None:
        prediction = safe_predict(
            prediction_input,
            use_hybrid=True,
            use_dimension_voting=False,
        )
        if prediction is not None:
            prediction_note = (
                "Dimension-level voting was unavailable, so the app used the next best combined prediction mode."
            )

    if prediction is None and questionnaire_prediction is not None:
        prediction = questionnaire_prediction
        prediction_note = (
            "Text analysis was unavailable, so the result was generated from the questionnaire model only."
        )
    elif prediction is None and text_prediction is not None:
        prediction = text_prediction
        prediction_note = (
            "Questionnaire analysis was unavailable, so the result was generated from the text model only."
        )

    if prediction is None:
        raise RuntimeError("No prediction model is currently available.")

    constrained = solve_career_constraints(
        mbti_type=prediction["mbti_type"],
        user_skills=user_skills,
        blocked_roles=blocked_roles,
        preferred_domains=preferred_domains or None,
        min_skill_overlap=0,
        top_k=12,
    )

    fallback_note = ""
    if not constrained and preferred_domains:
        constrained = solve_career_constraints(
            mbti_type=prediction["mbti_type"],
            user_skills=user_skills,
            blocked_roles=blocked_roles,
            preferred_domains=None,
            min_skill_overlap=0,
            top_k=12,
        )
        if constrained:
            fallback_note = (
                "No roles matched your selected domain filter exactly, so the app is showing the best overall MBTI-fit roles instead."
            )

    ranked = optimize_recommendations_hill_climbing(
        constrained,
        preferred_domains=preferred_domains,
    )
    top_roles = ranked[:5]

    transition_path: list[str] = []
    if current_role and top_roles:
        transition_path = find_recommendation_path(current_role, top_roles[0]["role"])

    return {
        "prediction": prediction,
        "text_prediction": text_prediction,
        "questionnaire_prediction": questionnaire_prediction,
        "top_roles": top_roles,
        "transition_path": transition_path,
        "fallback_note": fallback_note,
        "prediction_note": prediction_note,
        "consistency_note": build_prediction_consistency_note(
            prediction,
            text_prediction,
            questionnaire_prediction,
        ),
        "questionnaire_scoring_note": (
            "Q5 and Q10 are reverse-scored automatically before prediction so their wording matches the model's training scale."
        ),
        "scored_questionnaire": scored_questionnaire,
        "skills": sorted(user_skills),
        "current_role": current_role,
        "preferred_domains": st.session_state.preferred_domains,
    }


def render_progress() -> None:
    """Render the step progress display."""
    current_page = int(st.session_state.page)
    progress = (current_page + 1) / len(PAGE_LABELS)
    progress_label = PAGE_PROGRESS_NOTES[current_page]
    st.markdown(
        f"""
        <div class="progress-wrap">
            <div class="progress-meta">
                <div>
                    <div class="eyebrow">Step {current_page + 1} of {len(PAGE_LABELS)}</div>
                    <div class="progress-title">{PAGE_LABELS[current_page]}</div>
                </div>
                <div class="progress-note">{progress_label}</div>
            </div>
            <div class="progress-shell">
                <div class="progress-fill" style="width: {progress * 100:.0f}%;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    chips = []
    for index, label in enumerate(PAGE_LABELS):
        css_class = "step-chip active" if index == current_page else "step-chip"
        chips.append(f"<span class='{css_class}'>{index + 1}. {label}</span>")

    st.markdown(f"<div class='step-strip'>{''.join(chips)}</div>", unsafe_allow_html=True)


def render_site_header() -> None:
    """Render the global site header."""
    current_page = PAGE_LABELS[int(st.session_state.page)]
    st.markdown(
        f"""
        <div class="site-header">
            <div class="brand-wrap">
                <div class="brand-mark">CB</div>
                <div>
                    <div class="brand-name">CareerBloom</div>
                    <div class="brand-tag">A more human way to read how you work, think, and where you might fit best.</div>
                </div>
            </div>
            <div class="header-nav">
                <span class="nav-chip active">{current_page}</span>
                <span class="nav-chip">guided reflection</span>
                <span class="nav-chip">career signals</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_intro(title: str, body: str, *, side_label: str, side_text: str) -> None:
    """Render a reusable section header block."""
    st.markdown(
        f"""
        <div class="section-intro">
            <div class="section-copy">
                <div class="eyebrow">{PAGE_LABELS[st.session_state.page]}</div>
                <h2>{title}</h2>
                <p class="lead">{body}</p>
            </div>
            <div class="section-aside">
                <div class="eyebrow">{side_label}</div>
                <p class="small-muted">{side_text}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_welcome_page() -> None:
    """Render the landing screen."""
    st.markdown(
        """
        <div class="hero-card hero-split">
            <div>
                <div class="eyebrow">Personality + Career Fit</div>
                <h1>Find work that fits how your mind actually moves.</h1>
                <p class="lead">
                    This is not a stiff test. You will write a little, move through a few guided prompts,
                    and get a read on the kinds of roles that may feel natural for you.
                </p>
                <div class="pill-row">
                    <span class="soft-pill rose">write a little</span>
                    <span class="soft-pill sage">answer 15 prompts</span>
                    <span class="soft-pill sky">blend the signals</span>
                    <span class="soft-pill butter">see role matches</span>
                </div>
            </div>
            <div class="hero-aside">
                <div class="eyebrow">The Journey</div>
                <p class="small-muted">
                    A five-step flow that starts with you and ends with roles worth a closer look.
                </p>
                <div class="mini-stat-grid">
                    <div class="mini-stat">
                        <strong>About you</strong>
                        <span>Write the honest version, not the polished one.</span>
                    </div>
                    <div class="mini-stat">
                        <strong>How you think</strong>
                        <span>Use the sliders to show your natural leaning.</span>
                    </div>
                    <div class="mini-stat">
                        <strong>What fits</strong>
                        <span>Add your skills, filters, and current role.</span>
                    </div>
                    <div class="mini-stat">
                        <strong>Results</strong>
                        <span>See a blended read and a few grounded career paths.</span>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    left, right = st.columns([1.15, 1.85])
    with left:
        if st.button("Start your reflection", type="primary", use_container_width=True):
            set_page(1)
    with right:
        st.caption("A few honest minutes is enough. You do not need to over-explain anything.")


def render_profile_page() -> None:
    """Render the free-text profile page."""
    render_section_intro(
        "Tell me what you're like on a normal day.",
        "A few honest lines are enough. Do not overthink it.",
        side_label="Why this matters",
        side_text="The text model works best when it can hear your natural voice instead of a carefully edited answer.",
    )

    prompt_col, support_col = st.columns([1.45, 0.9], gap="large")
    with prompt_col:
        st.markdown(
            """
            <div class="prompt-card">
                <div class="eyebrow">A Good Prompt</div>
                <p class="small-muted">Use any of these to get started:</p>
                <ul class="mini-list">
                    <li>What do you naturally enjoy doing when nobody is assigning work?</li>
                    <li>Do you like working alone, or bouncing ideas off people?</li>
                    <li>What kinds of problems, systems, or topics keep your attention?</li>
                    <li>When you decide, what do you trust most?</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.text_area(
            "Your self-description",
            key="profile_text",
            on_change=invalidate_analysis_cache,
            height=260,
            placeholder="I usually do my best work when... The kinds of problems I enjoy are... I tend to decide by...",
            label_visibility="collapsed",
        )

    with support_col:
        st.markdown(
            f"""
            <div class="note-card">
                <div class="eyebrow">Try touching on</div>
                <p class="small-muted">{PROFILE_PROMPT}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        reaction_title, reaction_body = profile_reaction(st.session_state.profile_text)
        if reaction_title:
            st.markdown(
                f"""
                <div class="reaction-card">
                    <div class="eyebrow">Live Read</div>
                    <p><strong>{reaction_title}</strong></p>
                    <p class="small-muted">{reaction_body}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    total_words = word_count(st.session_state.profile_text)
    if total_words >= 50:
        st.success(f"{total_words} words. This is plenty of signal.")
    elif total_words > 0:
        st.info(f"{total_words} words so far. A little more detail will sharpen the read.")
    else:
        st.caption("You only need a few honest lines to get going.")

    back_col, spacer_col, next_col = st.columns([1, 3, 1])
    with back_col:
        if st.button("Back", key="profile_back", type="secondary", use_container_width=True):
            set_page(0)
    with next_col:
        if st.button("Keep going", key="profile_next", type="primary", use_container_width=True):
            if not st.session_state.profile_text.strip():
                st.error("Write a few lines first so the app has something real to work with.")
            else:
                set_page(2)


def render_questionnaire_page() -> None:
    """Render the 15-question slider page."""
    render_section_intro(
        "Show me how you naturally lean.",
        "Use the full scale when it helps, but trust your first instinct more than perfect consistency.",
        side_label="Quick note",
        side_text="One slider will not define you. The pattern across all of them matters more.",
    )

    st.markdown(
        """
        <div class="note-card">
            <div class="eyebrow">Scale</div>
            <p class="small-muted">1 means not me at all. 5 means very me. Q5 and Q10 are reverse-scored behind the scenes so the model stays consistent.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    for section_title, questions in QUESTION_SECTIONS:
        st.markdown(
            f"""
            <div class="note-card">
                <div class="eyebrow">{section_title}</div>
                <p class="small-muted">{QUESTION_SECTION_DESCRIPTIONS.get(section_title, "Go with what feels most natural.")}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for question_key, question_text in questions:
            render_slider_question(question_key, question_text)
        st.write("")

    back_col, spacer_col, next_col = st.columns([1, 3, 1])
    with back_col:
        if st.button("Back", key="questionnaire_back", type="secondary", use_container_width=True):
            set_page(1)
    with next_col:
        if st.button("On to the fit", key="questionnaire_next", type="primary", use_container_width=True):
            set_page(3)


def render_preferences_page() -> None:
    """Render the skills, preferences, and current-role page."""
    render_section_intro(
        "What should the results pay attention to?",
        "This step narrows the suggestions so they feel more like your world and less like generic matches.",
        side_label="Use this for",
        side_text="Skills strengthen the match. Domain and blocked roles stop the results from wandering too far from what you actually want.",
    )

    left_col, right_col = st.columns([1.25, 0.95], gap="large")
    with left_col:
        st.text_input(
            "Skills to bring with you",
            key="skills_text",
            on_change=invalidate_analysis_cache,
            placeholder="e.g. Python, data analysis, research, critical thinking",
        )
        st.text_input(
            "Roles to avoid",
            key="blocked_roles_text",
            on_change=invalidate_analysis_cache,
            placeholder="e.g. Sales Manager, HR Manager",
        )
        st.markdown(
            """
            <div class="note-card">
                <div class="eyebrow">Tiny reminder</div>
                <p class="small-muted">Write the skills you already trust, not the ones you only want to have someday.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right_col:
        st.multiselect(
            "Domains that feel interesting",
            options=DOMAIN_OPTIONS,
            key="preferred_domains",
            on_change=invalidate_analysis_cache,
            placeholder="Pick one or more domains",
        )
        st.caption("Finance maps to Business in the current dataset, and Creative maps to Creative Arts.")
        st.text_input(
            "Current role, if you want a transition path",
            key="current_role",
            on_change=invalidate_analysis_cache,
            placeholder="e.g. Software Engineer, Teacher, Accountant",
        )

    back_col, spacer_col, next_col = st.columns([1, 3, 1])
    with back_col:
        if st.button("Back", key="preferences_back", type="secondary", use_container_width=True):
            set_page(2)
    with next_col:
        if st.button("See what fits", key="preferences_next", type="primary", use_container_width=True):
            set_page(4)


def render_recommendation_cards(top_roles: list[dict[str, Any]]) -> None:
    """Render the top career recommendations."""
    if not top_roles:
        st.warning("No career recommendations were found with the current filters. Try broadening your domains or blocked roles.")
        return

    widths = [1.18, 1.0, 0.92]
    columns = st.columns(widths[: len(top_roles)]) if len(top_roles) == 3 else st.columns(len(top_roles))
    for index, (column, item) in enumerate(zip(columns, top_roles, strict=False), start=1):
        overlap = item.get("skill_overlap") or []
        overlap_text = ", ".join(overlap) if overlap else "No direct overlap yet"
        card_class = "result-card accent" if index == 1 else "result-card warm" if index == 2 else "result-card plain"
        with column:
            st.markdown(
                f"""
                <div class="{card_class}">
                    <div class="eyebrow">Top Match #{index}</div>
                    <div class="career-title">{item['role']}</div>
                    <p class="small-muted"><strong>Domain:</strong> {item['domain']}</p>
                    <p class="small-muted"><strong>Score:</strong> {item['score'] * 100:.1f}%</p>
                    <p class="small-muted"><strong>Skill overlap:</strong> {overlap_text}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_prediction_comparison(result: dict[str, Any]) -> None:
    """Render side-by-side prediction cards for questionnaire and combined outputs."""
    cards: list[tuple[str, dict[str, Any] | None]] = [
        ("From your answers", result.get("questionnaire_prediction")),
        ("Blended view", result.get("prediction")),
    ]
    if result.get("text_prediction") is not None:
        cards.insert(1, ("From your writing", result.get("text_prediction")))

    widths = [0.92, 1.06, 1.12]
    columns = st.columns(widths[: len(cards)]) if len(cards) == 3 else st.columns(len(cards))
    for index, (column, (label, prediction)) in enumerate(zip(columns, cards, strict=False), start=1):
        with column:
            if prediction is None:
                st.warning(f"{label} prediction is unavailable.")
                continue

            card_class = "result-card plain" if index == 1 else "result-card warm" if index == 2 else "result-card accent"
            st.markdown(
                f"""
                <div class="{card_class}">
                    <div class="eyebrow">{label}</div>
                    <h3>{prediction.get('mbti_type', 'Unknown')}</h3>
                    <p class="small-muted"><strong>Confidence:</strong> {float(prediction.get('confidence', 0.0)) * 100:.1f}%</p>
                    <p class="small-muted"><strong>Model:</strong> {prediction.get('model_used', 'N/A')}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_transition_path(result: dict[str, Any]) -> None:
    """Render the optional A* transition path."""
    current_role = result["current_role"]
    transition_path = result["transition_path"]
    top_roles = result["top_roles"]

    if not current_role:
        st.info("Enter your current role to see a transition path.")
        return

    if not top_roles:
        st.warning("A transition path could not be generated because no target role was available.")
        return

    if not transition_path:
        st.warning("A transition path could not be generated.")
        return

    target_role = top_roles[0]["role"]
    st.markdown(
        f"""
        <div class="result-card plain">
            <div class="eyebrow">Transition Path</div>
            <h3>A path toward {target_role}</h3>
            <p class="chain">{' &rarr; '.join(transition_path)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if len(transition_path) == 1:
        st.success("Your current role already matches the target role.")
    elif len(transition_path) == 2:
        st.success("This looks like a direct transition in the current role graph.")
    else:
        st.caption("The intermediate role can act as a stepping stone before the target role.")


def render_voting_breakdown(prediction: dict[str, Any]) -> None:
    """Render per-dimension voting details when combined input was used."""
    voting_info = prediction.get("voting_info")
    dimension_percentages = prediction.get("dimension_percentages", {})

    if not voting_info:
        if dimension_percentages:
            fallback_rows = [
                {"Dimension": pair, "Final Split": percentage}
                for pair, percentage in dimension_percentages.items()
            ]
            st.table(pd.DataFrame(fallback_rows))
        else:
            st.info("Per-dimension voting details are not available for this prediction mode.")
        return

    dimension_labels = {
        "EI": "Extroversion vs Introversion",
        "NS": "Intuition vs Sensing",
        "TF": "Thinking vs Feeling",
        "JP": "Judging vs Perceiving",
    }

    rows = []
    for pair_key, label in dimension_labels.items():
        vote = voting_info[pair_key]
        rows.append(
            {
                "Dimension": label,
                "Final Split": dimension_percentages.get(pair_key, "N/A"),
                "Text Vote": vote.get("text_winner", "N/A"),
                "Questionnaire Vote": vote.get("quest_winner", "N/A"),
                "Module 1 Vote": vote.get("m1_winner", "N/A"),
                "Final Vote": vote.get("fused_winner", "N/A"),
            }
        )

    st.table(pd.DataFrame(rows))


def render_results_page() -> None:
    """Render the results page and compute the pipeline output if needed."""
    render_section_intro(
        "Here is the shape that came through.",
        "Read this as a starting point, not a verdict. The goal is to give you something useful to react to.",
        side_label="Best way to read it",
        side_text="Notice which parts feel instantly true, which feel a little off, and which career directions spark curiosity.",
    )

    signature = build_signature()
    if st.session_state.analysis_signature != signature:
        try:
            with st.spinner("Analyzing your profile and scoring career matches..."):
                st.session_state.analysis_result = compute_results()
                st.session_state.analysis_signature = signature
        except Exception as exc:
            st.error(f"The app could not generate results right now: {exc}")
            st.stop()

    result = st.session_state.analysis_result or {}
    prediction = result.get("prediction", {})
    mbti_type = prediction.get("mbti_type", "Unknown")
    confidence = float(prediction.get("confidence", 0.0))

    if not result.get("skills"):
        st.warning("No skills were detected from the skills field for this run.")

    left, center, right = st.columns(3)
    with left:
        st.metric("Best-fit type", mbti_type)
    with center:
        st.metric("Confidence", f"{confidence * 100:.1f}%")
    with right:
        st.metric("Roles shown", len(result.get("top_roles", [])))

    st.markdown(
        f"""
        <div class="result-card accent result-hero">
            <div class="eyebrow">Blended Read</div>
            <div class="type-lockup">
                <h3>{mbti_type}</h3>
                <span class="type-badge">{confidence * 100:.0f}% sure</span>
            </div>
            <p class="lead">{humanize_mbti_summary(mbti_type)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if result.get("fallback_note"):
        st.info(result["fallback_note"])
    if result.get("prediction_note"):
        st.info(result["prediction_note"])
    if result.get("consistency_note"):
        st.warning(result["consistency_note"])
    if result.get("questionnaire_scoring_note"):
        st.info(result["questionnaire_scoring_note"])

    st.markdown(
        """
        <div class="subsection-heading">
            <div class="eyebrow">Model Read</div>
            <h3>How the different signals saw you</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_prediction_comparison(result)

    st.write("")
    st.markdown(
        """
        <div class="subsection-heading">
            <div class="eyebrow">Career Match</div>
            <h3>Roles worth a closer look</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_recommendation_cards(result.get("top_roles", []))

    st.write("")
    st.markdown(
        """
        <div class="subsection-heading">
            <div class="eyebrow">Dimension Read</div>
            <h3>Where the signals agreed</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_voting_breakdown(prediction)

    st.write("")
    st.markdown(
        """
        <div class="subsection-heading">
            <div class="eyebrow">Next Move</div>
            <h3>A path from where you are</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_transition_path(result)

    with st.expander("Debug: Inputs used in this run"):
        st.write("Parsed skills:", result.get("skills", []))
        st.write("Scored questionnaire:", result.get("scored_questionnaire", {}))
        st.write("Input signature:", st.session_state.analysis_signature)

    back_col, spacer_col, restart_col = st.columns([1, 3, 1])
    with back_col:
        if st.button("Back", key="results_back", type="secondary", use_container_width=True):
            set_page(3)
    with restart_col:
        if st.button("Start over", key="results_restart", type="secondary", use_container_width=True):
            reset_form()
            st.rerun()


def main() -> None:
    """Run the Streamlit application."""
    inject_styles()
    initialize_state()

    render_site_header()
    render_progress()

    current_page = int(st.session_state.page)
    if current_page == 0:
        render_welcome_page()
    elif current_page == 1:
        render_profile_page()
    elif current_page == 2:
        render_questionnaire_page()
    elif current_page == 3:
        render_preferences_page()
    else:
        render_results_page()


if __name__ == "__main__":
    main()
