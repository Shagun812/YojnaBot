#  YojnaBot — Multilingual Government Scheme Discovery Agent

> Built for **HackIndia AI Agents Hackathon 2026** | Adaptive Data Track

YojnaBot is an AI agent that helps rural Indian citizens discover government schemes they qualify for — in their own language (Hindi, Tamil, Telugu). Powered by **Adaptive Data** by Adaption for a continuously improving, multilingual dataset.

---

##  Problem Statement

Millions of eligible Indians miss out on government schemes due to language barriers and lack of awareness. YojnaBot solves this by:
- Conversationally collecting a user profile in their language
- Matching them to relevant schemes from a database of 1000+
- Generating a personalised action plan with documents, steps, and apply links
- Continuously improving through user corrections fed back into Adaptive Data

---

##  Architecture

```
User (Hindi/Tamil/Telugu)
        ↓
Profile Extraction Agent (Groq/Llama-3.3-70b, multi-turn)
        ↓
Structured UserProfile JSON
        ↓
Scheme Filter (keyword + scoring over cleaned_schemes.jsonl)
        ↓
Eligibility + Action Plan Agent (Gemini/Groq, single call)
        ↓
Scheme Cards in user's language
        ↓
User Correction → Adaptive Data re-adaptation → Dataset improves
```

---

##  Adaptive Data Integration

This project uses **[Adaptive Data](https://adaptionlabs.ai/adaptive-data)** as the dataset backbone:

| Stage | What YojnaBot does |
|---|---|
| **Ingest** | Uploads `stripped_schemes.jsonl` (38 schemes, multilingual) to Adaption |
| **Adapt** | Runs adaptation job to produce model-ready instruction-response pairs |
| **Evaluate** | Gets quality score (85%) shown live in Dataset Health sidebar |
| **Export** | Downloads adapted output as knowledge base |
| **Feedback Loop** | User corrections → re-uploaded to Adaption → re-adapted → score updates |

**Dataset:** https://huggingface.co/datasets/Aarav-Jain62677/yojnabot-schemes

---

##  Languages Supported

- हिंदी (Hindi)
- தமிழ் (Tamil)  
- తెలుగు (Telugu)

---

##  Setup & Run

```bash
git clone https://github.com/Shagun812/YojnaBot
cd YojnaBot

python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Create .env file
echo "GROQ_API_KEY=your_key" > .env
echo "ADAPTION_API_KEY=your_key" >> .env

# Initialize Adaptive Data pipeline
python adaptive_data.py

# Run the app
streamlit run app.py
```

---

##  Project Structure

```
YojnaBot/
├── app.py                          # Main Streamlit app
├── adaptive_data.py                # Adaption SDK integration
├── agent/
│   ├── profile_extractor.py        # Profile extraction agent
│   ├── scheme_filter.py            # Scheme matching logic
│   └── eligibility_agent.py        # Eligibility + action plan agent
├── data/
│   ├── cleaned_schemes.jsonl       # 1000+ government schemes
│   ├── adapted_eligibility_final.jsonl  # Multilingual adapted dataset
│   ├── stripped_schemes.jsonl      # Adaption-ready format
│   └── adaption_state.json         # Dataset ID + eval score
└── requirements.txt
```

---

##  Tech Stack

| Component | Tool |
|---|---|
| Frontend | Streamlit |
| Agent LLM | Groq (Llama-3.3-70b) |
| Dataset Platform | Adaptive Data (Adaption) |
| Languages | Python |
| Dataset Storage | HuggingFace Datasets |

---

##  Key Features

-  Conversational profile extraction (multi-turn, decides what to ask next)
-  Eligibility reasoning — LLM evaluates "does this user actually qualify" 
-  Multilingual output — full responses in Hindi/Tamil/Telugu
-  Live Dataset Health score from Adaption's evaluation
-  User correction → live re-adaptation loop
-  Download scheme results as text file
-  1000+ real government schemes database
