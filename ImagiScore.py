import subprocess
import sys
import os
import random
import streamlit as st
import spacy
import textstat
from nltk.corpus import wordnet as wn
from textblob import TextBlob
import time
import pyttsx3
import numpy as np
from langdetect import detect
from collections import Counter

# Load the spaCy model
nlp_en = spacy.load('en_core_web_sm')

vocabulary_words = [
    "sector", "available", "financial", "process", "individual", "specific", "principle", "estimate", "variables", "method", "data", "research", "contract", "environment", "export", "source", "assessment", "policy", "identified", "create", "derived", "factors", "procedure", "definition", "assume", "theory", "benefit", "evidence", "established", "authority", "major", "issues", "labour", "occur", "economic", "involved", "percent", "interpretation", "consistent", "income", "structure", "legal", "concept", "formula", "section", "required", "constitutional", "analysis", "distribution", "function", "area", "approach", "role", "legislation", "indicate", "response", "period", "context", "significant", "similar", "community", "resident", "range", "construction", "strategies", "elements", "previous", "conclusion", "security", "aspects", "acquisition", "features", "text", "commission", "regulations", "computer", "items", "consumer", "achieve", "final", "positive", "evaluation", "assistance", "normal", "relevant", "distinction", "region", "traditional", "impact", "consequences", "chapter", "equation", "appropriate", "resources", "participation", "survey", "potential", "cultural", "transfer", "select", "credit", "affect", "categories", "perceived", "sought", "focus", "purchase", "injury", "site", "journal", "primary", "complex", "institute", "investment", "administration", "maintenance", "design", "obtained", "restricted", "conduct", "comments", "convention", "published", "framework", "implies", "negative", "dominant", "illustrated", "outcomes", "constant", "shift", "deduction", "ensure", "specified", "justification", "funds", "reliance", "physical", "partnership", "location", "link", "coordination", "alternative", "initial", "validity", "task", "techniques", "excluded", "consent", "proportion", "demonstrate", "reaction", "criteria", "minorities", "technology", "philosophy", "removed", "sex", "compensation", "sequence", "corresponding", "maximum", "circumstances", "instance", "considerable", "sufficient", "corporate", "interaction", "contribution", "immigration", "component", "constraints", "technical", "emphasis", "scheme", "layer", "volume", "document", "registered", "core", "overall", "emerged", "regime", "implementation", "project", "hence", "occupational", "internal", "goals", "retained", "sum", "integration", "mechanism", "parallel", "imposed", "despite", "job", "parameters", "approximate", "label", "concentration", "principal", "series", "predicted", "summary", "attitudes", "undertaken", "cycle", "communication", "ethnic", "hypothesis", "professional", "status", "conference", "attributed", "annual", "obvious", "error", "implications", "apparent", "commitment", "subsequent", "debate", "dimensions", "promote", "statistics", "option", "domestic", "output", "access", "code", "investigation", "phase", "prior", "granted", "stress", "civil", "contrast", "resolution", "adequate", "alter", "stability", "energy", "aware", "licence", "enforcement", "draft", "styles", "precise", "medical", "pursue", "symbolic", "marginal", "capacity", "generation", "exposure", "decline", "academic", "modified", "external", "psychology", "fundamental", "adjustment", "ratio", "whereas", "enable", "version", "perspective", "contact", "network", "facilitate", "welfare", "transition", "amendment", "logic", "rejected", "expansion", "clause", "prime", "target", "objective", "sustainable", "equivalent", "liberal", "notion", "substitution", "generated", "trend", "revenue", "compounds", "evolution", "conflict", "image", "discretion", "entities", "orientation", "consultation", "mental", "monitoring", "challenge", "intelligence", "transformation", "presumption", "acknowledged", "utility", "furthermore", "accurate", "diversity", "attached", "recovery", "assigned", "tapes", "motivation", "bond", "edition", "nevertheless", "transport", "cited", "fees", "scope", "enhanced", "incorporated", "instructions", "subsidiary", "input", "abstract", "ministry", "capable", "expert", "preceding", "display", "incentive", "inhibition", "trace", "ignored", "incidence", "estate", "cooperative", "revealed", "index", "lecture", "discrimination", "overseas", "explicit", "aggregate", "gender", "underlying", "brief", "domain", "rational", "minimum", "interval", "neutral", "migration", "flexibility", "federal", "author", "initiatives", "allocation", "exceed", "intervention", "confirmed", "definite", "classical", "chemical", "voluntary", "release", "visible", "finite", "publication", "channel", "file", "thesis", "equipment", "disposal", "solely", "deny", "identical", "submitted", "grade", "phenomenon", "paradigm", "ultimately", "extract", "survive", "converted", "transmission", "global", "inferred", "guarantee", "advocate", "dynamic", "simulation", "topic", "insert", "reverse", "decades", "comprise", "hierarchical", "unique", "comprehensive", "couple", "mode", "differentiation", "eliminate", "priority", "empirical", "ideology", "somewhat", "aid", "foundation", "adults", "adaptation", "quotation", "contrary", "media", "successive", "innovation", "prohibited", "isolated", "highlighted", "eventually", "inspection", "termination", "displacement", "arbitrary", "reinforced", "denote", "offset", "exploitation", "detected", "abandon", "random", "revision", "virtually", "uniform", "predominantly", "thereby", "implicit", "tension", "ambiguous", "vehicle", "clarity", "conformity", "contemporary", "automatically", "accumulation", "appendix", "widespread", "infrastructure", "deviation", "fluctuations", "restore", "guidelines", "commodity", "minimises", "practitioners", "radical", "plus", "visual", "chart", "appreciation", "prospect", "dramatic", "contradiction", "currency", "inevitably", "complement", "accompany", "paragraph", "induced", "schedule", "intensity", "crucial", "via", "exhibit", "bias", "manipulation", "theme", "nuclear", "bulk", "behalf", "unified", "commenced", "erosion", "anticipated", "minimal", "ceases", "vision", "mutual", "norms", "intermediate", "manual", "supplementary", "incompatible", "concurrent", "ethical", "preliminary", "integral", "conversely", "relaxed", "confined", "accommodation", "temporary", "distorted", "passive", "subordinate", "analogous", "military", "scenario", "revolution", "diminished", "coherence", "suspended", "mature", "assurance", "rigid", "controversy", "sphere", "mediation", "format", "trigger", "qualitative", "portion", "medium", "coincide", "violation", "device", "insights", "refine", "devoted", "team", "overlap", "attained", "restraints", "inherent", "route", "protocol", "founded", "duration", "whereby", "inclination", "encountered", "convinced", "assembly", "albeit", "enormous", "reluctant", "posed", "persistent", "undergo", "notwithstanding", "straightforward", "panel", "odd", "intrinsic", "compiled", "adjacent", "integrity", "forthcoming", "conceived", "ongoing", "so-called", "likewise", "nonetheless", "levy", "invoked", "colleagues", "depression", "collapse"]

def run_streamlit():
    """Run the Streamlit app using subprocess."""
    if "STREAMLIT" not in os.environ:
        os.environ["STREAMLIT"] = "1"
        try:
            subprocess.run([sys.executable, "-m", "streamlit", "run", __file__], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running Streamlit: {e}")
            sys.exit(1)
    else:
        main_app()


# Function to get definitions from WordNet
def get_definitions(word):
    synsets = wn.synsets(word, lang='eng')
    if not synsets:
        return ["No definition found."]
    definitions = [syn.definition() for syn in synsets]
    return definitions


# Function to generate a vocabulary quiz
def generate_quiz(vocab_list):
    quiz = []
    for _ in range(len(vocab_list)):
        word = random.choice(vocab_list)
        definitions = get_definitions(word)
        if definitions[0] == "No definition found.":
            continue
        correct_definition = definitions[0]
        options = [correct_definition]
        while len(options) < 4:
            random_word = random.choice(vocab_list)
            random_definitions = get_definitions(random_word)
            if random_definitions[0] != "No definition found." and random_definitions[0] not in options:
                options.append(random_definitions[0])
        random.shuffle(options)
        quiz.append({
            'word': word,
            'options': options,
            'answer': correct_definition
        })
    return quiz


# Function to create flashcards
def create_flashcards(vocab_list):
    flashcards = []
    for _ in range(len(vocab_list)):
        word = random.choice(vocab_list)
        definitions = get_definitions(word)
        if definitions[0] != "No definition found.":
            flashcards.append({'word': word, 'definition': definitions[0]})
    return flashcards


# Function to pronounce the word using pyttsx3
def pronounce_word(word):
    tts_engine = pyttsx3.init()
    tts_engine.say(word)
    tts_engine.runAndWait()


# List of daily-life problems
problems = ["Technology has rapidly integrated into all aspects of our lives. Discuss how technology has changed human relationships. Are these changes positive or negative? Why?",
"Does today's education system adequately prepare students for the future? What changes could be made to improve the current education system? Provide suggestions based on your own experiences.",
"How socially responsible should young people be? What steps can young people take to address global issues like climate change and inequality?",
"Social media plays a huge role in the lives of young people. How does social media influence the process of identity formation? Discuss its positive and negative impacts.",
"In a rapidly advancing technological world, which careers do you think will be in demand in the future? How do you evaluate your own career plans in this context?",
"What does leadership mean to you? What are the most important qualities of a leader? Share a leadership experience you've had.",
"How should art and creativity be incorporated into daily life? Why is art education important? Share your personal experiences or observations on this topic.",
"How do you define success? Does your personal definition of success align with society's general understanding of it? Provide examples from your own life or from events you've witnessed.",
"What role do universal values play in the modern world? Do you think societies should pay more attention to ethical principles and universal values? Why?",
"Think of a time when you faced a challenge. How did you overcome it? Discuss the importance of perseverance and determination in personal growth.",
"Is it ethical to edit human embryos to prevent genetic diseases? What are the potential risks and consequences of gene editing for future generations?",
"As surveillance technology improves, should governments and corporations have access to personal data for security purposes? Where should the line be drawn between privacy and safety?",
"Should we intervene in the Earth’s climate through geoengineering to prevent climate disasters, even if the long-term effects are unknown? What are the ethical concerns of taking such drastic action?",
"Is it ethical to develop autonomous drones or robots that can kill without human intervention? Should countries ban the use of autonomous weapons?",
"If technology allows humans to enhance their physical or cognitive abilities, should everyone have access to it? How do we prevent inequality between augmented and non-augmented humans?",
"As humans begin to explore colonizing Mars or other celestial bodies, do we have the right to alter or exploit these environments? Should space exploration focus on Earth's problems instead?",
"Should advancements in biotechnology be used to develop weapons based on genetic information? How can the misuse of such technology be prevented?",
"Should wealthy countries be required to take in refugees displaced by climate change? What are the ethical responsibilities of nations that contribute more to climate change?",
"If technology allows for the transfer of human consciousness into a digital format, should we pursue digital immortality? What does it mean to be human in a digital existence?",
"Should we create synthetic organisms to solve environmental or health problems? How do we ensure that these organisms do not pose a threat to natural ecosystems?",
"As robots and AI systems become more human-like, should they be granted certain rights or protections? How do we balance the rights of humans and machines?",
"Is it ethical for companies to collect and analyze personal health data through wearable devices for profit? Should individuals be compensated for their data?",
"If cloning technology advances, should humans be allowed to clone themselves? What ethical concerns arise around identity and individuality in clones?",
"Should we rely on lab-grown meat to solve global food security problems, even if it may disrupt traditional farming industries? What are the moral implications for animal rights and the environment?",
"Should nanotechnology be used to extend human life or enhance intelligence, even if it is only accessible to the wealthy? How do we ensure fair access to life-extending technologies?",
"If time travel were possible, should humans be allowed to alter past events? How might changing history impact future generations, and is it ethical to do so?",
"As research into brain-computer interfaces advances, should we use technology to control or influence human thoughts and behavior? What are the ethical limits of such interventions?",
"Should international law prioritize the protection of oceans over economic activities like deep-sea mining or fishing? How do we balance environmental preservation with economic needs?",
"If science offers the ability to halt aging and extend life indefinitely, should we pursue it? What would be the societal and ethical consequences of living forever?"]

# Function to calculate Jaccard similarity
# Jaccard similarity calculation
def jaccard_similarity(str1, str2):
    a = set(str1.split())
    b = set(str2.split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


# Coherence evaluation using readability
def evaluate_coherence(prompt):
    # Flesch Reading Ease score (0 to 100, higher is easier to read)
    reading_ease = textstat.flesch_reading_ease(prompt)

    # Grade level readability score
    grade_level = textstat.flesch_kincaid_grade(prompt)

    # You can combine these into a score or set thresholds for acceptable coherence
    return reading_ease, grade_level


# Keyword relevance evaluation
def evaluate_keywords(prompt, expected_keywords):
    # Tokenize the prompt
    prompt_words = set(prompt.lower().split())

    # Check the overlap with expected keywords
    common_keywords = prompt_words.intersection(expected_keywords)

    # Return a score based on how many expected keywords are found
    return len(common_keywords) / len(expected_keywords)


# Final prompt evaluation function
def evaluate_prompt(prompt, previous_prompts, problem):
    score = 0

    # 1. Language Detection
    try:
        detected_lang = detect(prompt)
    except:
        detected_lang = 'unknown'

    # 2. Length and Complexity (Language-independent)
    word_count = len(prompt.split())
    score += min(word_count, 750)  # Add points up to a maximum of 750 words.

    # 3. Penalizing Repetitions
    word_freq = Counter(prompt.split())
    repetition_penalty = sum([count - 1 for count in word_freq.values() if count > 1])
    score -= repetition_penalty

    # 4. Originality: Calculate Jaccard Similarity with Previous Prompts
    if previous_prompts:
        similarities = [jaccard_similarity(prompt, prev_prompt) for prev_prompt in previous_prompts]
        originality_score = 1 - np.mean(similarities)
        score += originality_score * 50  # Assign weight to originality
    else:
        score += 50  # First prompt is considered original by default

    # 5. Relevance and Coherence only if detected language is English
    if detected_lang == 'en':
        # Relevance (problem-related keywords)
        problem_keywords = set(problem.lower().split())
        relevance_score = evaluate_keywords(prompt, problem_keywords)
        score += relevance_score * 50  # Weight for relevance score

        # Coherence (Readability scores)
        reading_ease, grade_level = evaluate_coherence(prompt)
        # Penalize very low readability (hard to read) or high grade level (too complex)
        if reading_ease < 30 or grade_level > 12:
            score -= 10  # Deduct points for difficult or overly complex text

        # Sentiment Analysis
        blob = TextBlob(prompt)
        sentiment_score = blob.sentiment.polarity + 1  # Sentiment polarity, normalized (0 to 2)
        score += sentiment_score * 25  # Weight for sentiment analysis

    return score, detected_lang


# Main Streamlit App
def main_app():
    st.title("ImagiScore")

    # Initialize session state
    if 'quiz_index' not in st.session_state:
        st.session_state.quiz_index = 0
        st.session_state.quiz_score = 0
        st.session_state.quiz = generate_quiz(vocabulary_words)

    if 'flashcard_index' not in st.session_state:
        st.session_state.flashcard_index = 0
        st.session_state.show_definition = False

    if 'prompts' not in st.session_state:
        st.session_state.prompts = []
        st.session_state.prompt_scores = []
        st.session_state.current_problem_index = random.randint(0, len(problems) - 1)

    # Vocabulary Quiz Section
    st.header("Vocabulary Quiz")
    quiz = st.session_state.quiz
    index = st.session_state.quiz_index
    score = st.session_state.quiz_score

    if index < len(quiz):
        q = quiz[index]
        st.subheader(f"Word: {q['word']}")
        answer = st.radio("Choose the correct definition:", q['options'], key=f"quiz_{index}")
        if st.button("Submit Answer", key=f"submit_quiz_{index}"):
            if answer == q['answer']:
                st.success("Correct!")
                score += 1
                time.sleep(1.5)  # Delay for success message
            else:
                st.error(f"Wrong! The correct answer is: {q['answer']}")
                time.sleep(3)  # Delay for error message
            st.session_state.quiz_index += 1
            st.session_state.quiz_score = score
            st.experimental_rerun()
    else:
        st.write(f"Quiz finished! Your score: {score}/{len(quiz)}")
        if st.button("Restart Quiz"):
            st.session_state.quiz_index = 0
            st.session_state.quiz_score = 0
            st.session_state.quiz = generate_quiz(vocabulary_words)
            st.experimental_rerun()


    # Flashcards Section
    st.header("Interactive Flashcards")

    # Check if flashcards are already created, if not create them and store in session state
    if 'flashcards' not in st.session_state:
        st.session_state.flashcards = create_flashcards(vocabulary_words)

    flashcards = st.session_state.flashcards
    flashcard_index = st.session_state.flashcard_index
    show_definition = st.session_state.show_definition
    flashcard = flashcards[flashcard_index]

    st.subheader(f"Word: {flashcard['word']}")
    if st.button("Pronounce Word"):
        pronounce_word(flashcard['word'])

    if st.button("Show Definition"):
        st.session_state.show_definition = True
        st.experimental_rerun()

    if show_definition:
        st.write(f"Definition: {flashcard['definition']}")

    if st.button("Next Flashcard"):
        st.session_state.flashcard_index = (flashcard_index + 1) % len(flashcards)
        st.session_state.show_definition = False
        st.experimental_rerun()

    # Creativity Section
    st.header("Creative Writing Practice")
    current_problem = problems[st.session_state.current_problem_index]
    st.write(f"Problem: {current_problem}")

    new_prompt = st.text_area("Now solve that problem:")
    if st.button("Submit"):
        if new_prompt:
            score = evaluate_prompt(new_prompt, [p['prompt'] for p in st.session_state.prompts], current_problem)
            st.session_state.prompts.append({
                'prompt': new_prompt,
                'score': score
            })
            st.success("Submitted successfully!")
    # Navigation Buttons
    if st.button("Next Problem"):
                st.session_state.current_problem_index = random.randint(0, len(problems) - 1)
                st.experimental_rerun()

    # Display and Evaluate Prompts
    st.header("Submitted Ones")
    for idx, prompt_data in enumerate(st.session_state.prompts):
        st.write(f"Prompt {idx + 1}: {prompt_data['prompt']}")
        st.write(f"Creativity Score: {prompt_data['score']}")

if __name__ == "__main__":
    if "STREAMLIT" in os.environ:
        main_app()
    else:
        run_streamlit()
