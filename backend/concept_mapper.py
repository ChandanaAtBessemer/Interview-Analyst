# concept_mapper.py

import os
import json
import hashlib
import numpy as np
from typing import List, Tuple
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
# Load API key
client = OpenAI()

# Base concepts to match against

BASE_CONCEPTS = {
    "Supply Chain Issues": "Delayed shipments, long lead times, and backorders",
    "Manual Processes": "We manually update spreadsheets and track orders",
    "Automation/API": "We use APIs or software to automate data entry",
    "Transparency": "Everyone should see accurate real-time data",
    "Customer Impact": "Our customer was angry because we missed the delivery",
    "Vendor Relations": "We called the vendor to ask about a delay",
    "Technical Debt": "We still rely on outdated systems that slow us down",
    "Change Resistance": "People don’t want to use new software or tools",
    "Data Inaccuracy": "The numbers in the portal don't match the EDI feed",
    "Operational Efficiency": "We want to save time and reduce errors",
    # ADD THESE NEW ONES:
    "Project Management": "Managing multiple projects, tracking orders, and coordinating deliveries",
    "Workload Management": "Handling 30-40 projects, managing multiple bids and quotes",
    "System Integration": "Using Eclipse, Square D, ERP systems for project tracking",
    "Lead Time Management": "12 week lead times, 50 week delays, delivery scheduling",
    "Customer Communication": "Sending ship schedules, providing updates to contractors",
    "Equipment Specification": "Power distribution equipment, panels, switchgear configuration",
    "Field Operations": "RTI programs, field installation, temporary solutions",
    "Cost Management": "Change orders, value engineering, budget considerations",

    # Caregiving / mobility
    "Caregiver Burden": "Exhausted, juggling chores, barely hanging on, need sleep, doing best I can",
    "Resilience & Coping": "Read a book, watch a movie, finding small ways to recharge",
    "Advocacy": "Becoming her advocate, doing what's best for her, speaking up for needs",
    "ALS Progression & Needs": "Core strength gone, losing arm ability, wheelchair reliance, final diagnosis ALS",
    "Medical Logistics": "Trips to med center, therapy, primary care, frequent appointments",
    "Accessible Vehicle Choice": "Toyota Sienna, side ramp vs rear, kneel/tilt feature, ramp lights",
    "Conversion Quality & Reliability": "BraunAbility reputation, dependable conversion, heavy power chair fit",
    "Dealer & Service Experience": "United Access support, quick fixes, follow-up surveys, paid Uber rides",
    "Home Modifications": "Stairlift install, bathroom adaptations, hospital bed, Hoyer lift",
    "Equipment Issues": "Stairlift problems, leaking A/C drain in van, service not responsive",
    "Bathing & Transfers": " tilt-back needs, safe transfers without holding her up",
    "Securement & Safety": "Locking bar Q'Straint lock-in, parking clearance for side ramp",
    "Driving Risk & Damage": "Fender bender could bend floor, ramp may stop working",
    "Financing Approach": "Refinanced house, paid cash, trade-in RAV4, budget realities",
    "Insurance & Warranty": "Allstate coverage limits, extended warranties on van & conversion, shopping other insurers",
    "Program Eligibility": "Medicaid over-income by $50, Medicare DME coverage limits, foundations help",
    "Word-of-Mouth & Research": "Mechanic and industry friends, online searches, brand reviews",
    "Training & Onboarding Gap": "Zero orientation, learned by doing, wish guidance existed",
    "Accessible Travel Constraints": "Cruise rooms not accessible, year-in-advance booking, limited mobility rooms",
    "Daily Outings & Community Access": "Costco curb unload, using flashers, practical workarounds",
    "Brand Trust & Care": "BraunAbility is caring, checks in with surveys, best quality, dependable support",
    "Independence & Quality of Life": "Van is a godsend, provides safety and independence, able to go places again",

    "challenges": "problems, difficulties, obstacles, issues, struggles",
    "solutions": "fixes, improvements, recommendations, suggestions, ways to solve",
    "experiences": "what happened, stories, situations, events",
    "opinions": "thoughts, feelings, preferences, beliefs, views", 
    "processes": "how things work, procedures, workflows, methods",
    "relationships": "interactions between people, teams, systems"
}

EMBEDDING_CACHE = {}

def get_cache_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def get_embedding(text: str) -> List[float]:
    key = get_cache_key(text)
    if key in EMBEDDING_CACHE:
        return EMBEDDING_CACHE[key]

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    embedding = response.data[0].embedding
    EMBEDDING_CACHE[key] = embedding

    

    return embedding

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    a = np.array(vec1)
    b = np.array(vec2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def map_to_concept(phrase: str) -> Tuple[str, float]:
    phrase_emb = get_embedding(phrase)
    best_concept = None
    best_score = 0.0

    for concept, desc in BASE_CONCEPTS.items():
        concept_emb = get_embedding(desc)
        score = cosine_similarity(phrase_emb, concept_emb)
        print(f" - {concept}: {score:.2f}")
        if score > best_score:
            best_concept = concept
            best_score = score
   

    return best_concept, best_score  # Always return the best, no threshold check


# Test
if __name__ == "__main__":
    test_phrases = [
        "delayed order shipping",
        "manual tracking of inventory",
        "bad data coming from vendor",
        "staff don’t want to adopt AI",
        "projects stalled due to slow updates",
        "we use APIs to pull data",
    ]
    for phrase in test_phrases:
        concept, score = map_to_concept(phrase)
        print(f"{phrase} → {concept} (score: {score:.2f})")
    