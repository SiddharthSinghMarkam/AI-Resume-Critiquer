"""
Resume Analyzer Pro — Enhanced Edition
Fully Offline | NLP-Powered | No AI | No API | No Internet Required
Features: ATS Score, Spelling, Grammar, Tense Check, Action Verbs,
          Impact Metrics, Bullet Quality, Sections, Skills,
          Contact Validator, Industry Detector, Word Frequency,
          Career Timeline, Personalized Action Plan, Readability
"""

import streamlit as st
import pdfplumber
import re
import io
from collections import Counter
import logging

logging.getLogger("nltk").setLevel(logging.ERROR)

# ══════════════════════════════════════════════════════════════════
#  NLTK BOOTSTRAP
# ══════════════════════════════════════════════════════════════════
import nltk

def _bootstrap_nltk():
    for pkg, path in [
        ("punkt",                      "tokenizers/punkt"),
        ("punkt_tab",                  "tokenizers/punkt_tab"),
        ("stopwords",                  "corpora/stopwords"),
        ("averaged_perceptron_tagger", "taggers/averaged_perceptron_tagger"),
    ]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)

_bootstrap_nltk()

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords as _sw
from spellchecker import SpellChecker
import textstat

# ══════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════
STOP_WORDS = set(_sw.words("english"))

ACTION_VERBS = {
    "🏆 Leadership":    ["led","managed","directed","supervised","oversaw",
                         "coordinated","guided","spearheaded","established",
                         "founded","launched","initiated","pioneered",
                         "orchestrated","championed","delegated","mentored",
                         "empowered","mobilized","aligned"],
    "📈 Achievement":   ["achieved","accomplished","delivered","exceeded",
                         "improved","increased","reduced","saved","generated",
                         "boosted","maximized","optimized","streamlined",
                         "transformed","surpassed","doubled","tripled",
                         "accelerated","expanded","scaled"],
    "💻 Technical":     ["developed","designed","built","created","implemented",
                         "engineered","programmed","automated","integrated",
                         "deployed","configured","architected","coded",
                         "debugged","tested","migrated","refactored",
                         "containerized","secured","optimized"],
    "🔍 Analysis":      ["analyzed","evaluated","assessed","researched",
                         "identified","diagnosed","investigated","measured",
                         "monitored","forecasted","modeled","reviewed",
                         "audited","examined","quantified","benchmarked",
                         "interpreted","validated","synthesized","mapped"],
    "🤝 Communication": ["presented","communicated","collaborated","negotiated",
                         "consulted","advised","mentored","trained",
                         "facilitated","persuaded","drafted","authored",
                         "published","reported","documented","briefed",
                         "liaised","promoted","advocated","illustrated"],
    "📋 Project Mgmt":  ["planned","executed","organized","scheduled",
                         "prioritized","delegated","budgeted","allocated",
                         "defined","scoped","completed","finalized",
                         "tracked","estimated","coordinated","launched",
                         "oversaw","controlled","delivered","managed"],
}

ALL_VERBS_FLAT = {v for cat in ACTION_VERBS.values() for v in cat}

RESUME_SECTIONS = {
    "📧 Contact Information":   ["email","phone","mobile","linkedin","github","address","contact"],
    "📝 Summary / Objective":   ["summary","objective","profile","about me","overview","introduction","professional summary"],
    "🎓 Education":             ["education","degree","university","college","school","gpa","graduation","academic"],
    "💼 Work Experience":       ["experience","work experience","employment","career","professional experience","internship","job history"],
    "⚙️ Skills":                ["skills","technologies","tools","competencies","expertise","tech stack","technical skills"],
    "🚀 Projects":              ["projects","portfolio","case studies","personal projects","academic projects"],
    "📜 Certifications":        ["certification","certificate","courses","training","credential","license"],
    "🏅 Awards & Achievements": ["award","achievement","honor","recognition","accomplishment","prize","scholarship"],
}

TECH_SKILLS = [
    "python","java","javascript","typescript","c++","c#","ruby","go","rust","swift",
    "kotlin","scala","matlab","php","perl","bash","shell","html","css","react","angular",
    "vue","node.js","express","django","flask","fastapi","spring","laravel","asp.net",
    "next.js","svelte","machine learning","deep learning","nlp","computer vision",
    "tensorflow","pytorch","keras","scikit-learn","pandas","numpy","matplotlib","seaborn",
    "tableau","power bi","data analysis","data science","statistics","jupyter","hadoop",
    "spark","sql","mysql","postgresql","mongodb","redis","sqlite","oracle","elasticsearch",
    "dynamodb","firebase","cassandra","aws","azure","google cloud","gcp","docker",
    "kubernetes","jenkins","git","github","gitlab","ci/cd","terraform","ansible","linux",
    "unix","agile","scrum","rest api","graphql","microservices","cybersecurity","blockchain",
    "android","ios","figma","adobe xd","photoshop","illustrator","sketch","excel",
    "powerpoint","jira","confluence","slack",
]

SOFT_SKILLS = [
    "communication","leadership","teamwork","problem solving","critical thinking",
    "time management","adaptability","creativity","collaboration","detail oriented",
    "self motivated","organized","multitasking","interpersonal","presentation skills",
    "negotiation","decision making","strategic thinking","project management",
    "conflict resolution","emotional intelligence","analytical","innovative",
    "proactive","flexible",
]

WEAK_PHRASES = [
    "responsible for","worked on","helped with","assisted in","participated in",
    "was involved in","duties included","was in charge of","tried to",
    "attempted to","was responsible","helped to","worked with",
]

BUZZWORDS = [
    "team player","hard worker","go-getter","results-driven","fast learner",
    "self-starter","dynamic professional","proven track record",
    "think outside the box","synergy","leverage","game changer",
    "guru","ninja","rockstar","wizard","passionate individual",
]

INDUSTRY_PROFILES = {
    "💻 Software Engineering": {
        "signals":        ["software","developer","engineer","backend","frontend","fullstack","api","devops"],
        "must_skills":    ["git","agile","sql","rest api","docker"],
        "power_words":    ["scalable","robust","optimized","architected","microservices","modular"],
        "extra_keywords": ["system design","code review","unit testing","performance tuning","ci/cd","clean code","design patterns","security"],
    },
    "📊 Data Science / ML": {
        "signals":        ["data scientist","machine learning","deep learning","data analyst","ai engineer","statistics"],
        "must_skills":    ["python","sql","pandas","numpy","machine learning"],
        "power_words":    ["predicted","modeled","clustered","classified","accuracy","f1 score","auc-roc"],
        "extra_keywords": ["feature engineering","model deployment","a/b testing","data pipeline","etl","visualization","mlflow","hyperparameter tuning"],
    },
    "🎨 UI/UX Design": {
        "signals":        ["ui designer","ux designer","product designer","interaction design","figma","wireframe"],
        "must_skills":    ["figma","user research","prototyping","wireframing"],
        "power_words":    ["designed","prototyped","researched","usability","wireframed","iterated"],
        "extra_keywords": ["user research","a/b testing","design system","accessibility","responsive design","information architecture"],
    },
    "📈 Marketing / Digital": {
        "signals":        ["marketing","seo","social media","content marketing","digital marketing","brand","growth"],
        "must_skills":    ["seo","analytics","content strategy","google analytics"],
        "power_words":    ["grew","increased","engaged","converted","targeted","acquired","retained"],
        "extra_keywords": ["conversion rate","roi","kpi","funnel","lead generation","ab testing","email marketing","ppc","sem"],
    },
    "💰 Finance / Accounting": {
        "signals":        ["finance","accounting","banking","investment","audit","financial analyst","cpa","budget"],
        "must_skills":    ["excel","financial modeling","accounting","sql"],
        "power_words":    ["reconciled","forecasted","audited","analyzed","managed budget","reported"],
        "extra_keywords": ["financial modeling","risk management","compliance","reconciliation","variance analysis","p&l","forecasting"],
    },
    "⚙️ Operations / Management": {
        "signals":        ["operations","manager","director","strategy","process improvement","supply chain","logistics"],
        "must_skills":    ["project management","strategic planning","excel"],
        "power_words":    ["streamlined","reduced costs","scaled","led team","optimized","improved efficiency"],
        "extra_keywords": ["kpi","process improvement","cross-functional","stakeholder management","lean","six sigma","change management"],
    },
}

CONTACT_PATTERNS = {
    "📧 Email":     re.compile(r'[\w.+\-]+@[\w\-]+\.[\w.]+'),
    "📱 Phone":     re.compile(r'(?<!\d)(?:\+?91[\s\-]?)?[6-9]\d{9}(?!\d)|(?<!\d)\+?[1-9]\d{1,3}[\s.\-]?\(?\d{2,4}\)?[\s.\-]?\d{3,4}[\s.\-]?\d{3,4}(?!\d)'),
    "🔗 LinkedIn":  re.compile(r'linkedin\.com/in/[\w\-]+', re.IGNORECASE),
    "🐙 GitHub":    re.compile(r'github\.com/[\w\-]+', re.IGNORECASE),
    "🌐 Portfolio": re.compile(r'https?://(?!(?:www\.)?linkedin|(?:www\.)?github)[\w.\-]+\.(?:com|io|dev|me|net|in)(?:/[\w\-/]*)?', re.IGNORECASE),
}

IMPACT_RE = re.compile(
    r'(\d+\s*%'
    r'|\$\s*\d[\d,.]*'
    r'|₹\s*\d[\d,.]*'
    r'|\d+\s*(?:x|X)\b'
    r'|\b\d+\s*(?:k|m|b)\b'
    r'|\b\d+\s*(?:million|billion|thousand|lakh|crore)\b'
    r'|\btop\s*\d+\b'
    r'|\b\d+\s*(?:hours?|days?|weeks?|months?|years?)\b'
    r'|\b(?:first|second|third|#\d+)\b'
    r'|\b\d{3,}\b)',
    re.IGNORECASE
)

DATE_RE = re.compile(
    r'(?:'
    r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|'
    r'Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'
    r'[\s,\-]+\d{4}'
    r'|\d{1,2}[/\-]\d{4}'
    r'|\d{4}\s*[–\-]\s*(?:\d{4}|present|current|now|ongoing)'
    r'|(?:present|current|now|ongoing)'
    r')',
    re.IGNORECASE
)

PRESENT_VERBS_RE = re.compile(
    r'\b(develop|create|build|manage|lead|design|implement|achieve|'
    r'improve|increase|deliver|analyze|coordinate|launch|execute|plan|'
    r'reduce|generate|optimize|automate|integrate|deploy|train|mentor|'
    r'present|collaborate|research|identify|complete|maintain|support|'
    r'provide|handle|write|test|review|track|monitor|work|use)\b',
    re.IGNORECASE
)

# ══════════════════════════════════════════════════════════════════
#  CORE ANALYSIS FUNCTIONS
# ══════════════════════════════════════════════════════════════════

def extract_text_from_pdf(file_bytes: bytes) -> str:
    text = ""
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                pg = page.extract_text()
                if pg:
                    text += pg + "\n"
    except Exception as e:
        st.error(f"PDF read error: {e}")
    return text.strip()


def calculate_ats(resume: str, jd: str):
    if not jd.strip():
        return None, [], [], [], []
    def kw_set(t):
        return {w for w in word_tokenize(t.lower())
                if w.isalpha() and len(w) > 2 and w not in STOP_WORDS}
    rk, jk  = kw_set(resume), kw_set(jd)
    matched = sorted(rk & jk)
    missing = sorted(jk - rk)
    score   = round(len(matched) / max(len(jk), 1) * 100, 1)
    rl, jl  = resume.lower(), jd.lower()
    t_match = [s for s in TECH_SKILLS if s in rl and s in jl]
    t_miss  = [s for s in TECH_SKILLS if s in jl and s not in rl]
    return min(score, 100.0), matched[:50], missing[:30], t_match, t_miss


def check_spelling(text: str):
    spell    = SpellChecker()
    clean    = re.sub(r"[^a-zA-Z\s]", " ", text)
    cands    = [w.lower() for w in clean.split()
                if w.isalpha() and len(w) > 2
                and not w.isupper() and w[0].islower()]
    results  = []
    for word in sorted(spell.unknown(cands))[:30]:
        sugg = sorted(spell.candidates(word) or [])[:4]
        results.append({"word": word, "suggestions": sugg})
    return results


def grammar_check(text: str):
    issues  = []
    passive = re.compile(r"\b(was|were|is|are|been|being)\s+\w+ed\b", re.IGNORECASE)
    filler  = re.compile(
        r"\b(very|really|basically|honestly|quite|just|so|simply|kind of|sort of)\b",
        re.IGNORECASE
    )
    for sent in sent_tokenize(text):
        s  = sent.strip()
        wc = len([w for w in word_tokenize(s) if w.isalpha()])
        if not s:
            continue
        if wc > 40:
            issues.append(("🔴 Long Sentence",  s[:130], "Break into ≤25-word sentences."))
        if 0 < wc < 4:
            issues.append(("🟡 Too Short",       s,       "Expand with quantifiable impact."))
        if passive.search(s):
            issues.append(("🟡 Passive Voice",   s[:130], "Rewrite with an active action verb."))
        if re.match(r"^I\b", s):
            issues.append(("🟡 Starts with 'I'", s[:100], "Omit 'I'. Start with an action verb."))
        m = filler.search(s)
        if m:
            issues.append(("🟢 Filler Word", s[:100], f'Remove "{m.group()}" — weakens impact.'))
    return issues[:25]


def check_tense(text: str):
    issues = []
    for sent in sent_tokenize(text):
        s = sent.strip()
        m = PRESENT_VERBS_RE.search(s)
        if m and len(s.split()) > 4:
            issues.append({
                "sentence": s[:120],
                "verb":     m.group(),
                "fix":      f'Change "{m.group()}" → past tense (e.g. "{m.group().rstrip("e")}ed").'
            })
    return issues[:20]


def action_verb_analysis(text: str):
    tl     = text.lower()
    tokens = set(word_tokenize(tl))
    found  = {cat: [v for v in verbs if v in tokens]
              for cat, verbs in ACTION_VERBS.items()}
    found  = {k: v for k, v in found.items() if v}
    weak   = [p for p in WEAK_PHRASES if p in tl]
    buzz   = [b for b in BUZZWORDS   if b in tl]
    sugg   = {cat: [v for v in verbs if v not in (found.get(cat) or [])][:5]
              for cat, verbs in ACTION_VERBS.items()}
    return found, weak, buzz, sugg


def section_check(text: str):
    tl      = text.lower()
    present = [s for s, kws in RESUME_SECTIONS.items() if any(k in tl for k in kws)]
    missing = [s for s in RESUME_SECTIONS if s not in present]
    return present, missing


def extract_skills(text: str):
    tl = text.lower()
    return ([s for s in TECH_SKILLS if s in tl],
            [s for s in SOFT_SKILLS if s in tl])


def get_readability(text: str):
    try:
        wc = textstat.lexicon_count(text)
        sc = textstat.sentence_count(text)
        return {
            "Word Count":           wc,
            "Sentence Count":       sc,
            "Avg Words/Sentence":   round(wc / max(sc, 1), 1),
            "Flesch Reading Ease":  round(textstat.flesch_reading_ease(text), 1),
            "Flesch-Kincaid Grade": round(textstat.flesch_kincaid_grade(text), 1),
        }
    except Exception:
        return {}


def check_impact(text: str):
    bullets = [l.strip() for l in text.splitlines()
               if l.strip() and len(l.split()) >= 5]
    quantified, needs_work = [], []
    for b in bullets:
        (quantified if IMPACT_RE.search(b) else needs_work).append(b)
    pct = round(len(quantified) / max(len(bullets), 1) * 100, 1)
    return pct, quantified[:20], needs_work[:20]


def validate_contact(text: str):
    found, missing = {}, []
    for label, pat in CONTACT_PATTERNS.items():
        m = pat.search(text)
        if m:
            found[label] = m.group()
        else:
            missing.append(label)
    return found, missing


def analyze_bullets(text: str):
    results = []
    for line in text.splitlines():
        line = line.strip()
        if len(line.split()) < 5 or len(line) < 20:
            continue
        words   = word_tokenize(line.lower())
        first_w = words[0] if words else ""
        wc      = len([w for w in words if w.isalpha()])
        has_verb   = first_w in ALL_VERBS_FLAT
        has_impact = bool(IMPACT_RE.search(line))
        good_len   = 8 <= wc <= 30
        score      = sum([has_verb, has_impact, good_len])
        results.append({
            "text": line[:130], "has_verb": has_verb,
            "has_impact": has_impact, "good_len": good_len, "score": score,
        })
    return results[:40]


def detect_industry(text: str):
    tl = text.lower()
    scores = {ind: sum(1 for s in data["signals"] if s in tl)
              for ind, data in INDUSTRY_PROFILES.items()}
    if not any(scores.values()):
        return None, {}
    top = max(scores, key=scores.get)
    return top, INDUSTRY_PROFILES[top]


def word_frequency(text: str):
    tokens  = word_tokenize(text.lower())
    cleaned = [t for t in tokens
               if t.isalpha() and len(t) > 3 and t not in STOP_WORDS]
    freq    = Counter(cleaned)
    overused = {w: c for w, c in freq.items() if c >= 4}
    return freq.most_common(20), overused


def extract_timeline(text: str):
    dates = DATE_RE.findall(text)
    years = sorted(set(int(y) for y in re.findall(r'\b(20\d{2}|19\d{2})\b', text)))
    gaps  = []
    for i in range(1, len(years)):
        if years[i] - years[i-1] > 2:
            gaps.append(f"{years[i-1]} → {years[i]}  ({years[i]-years[i-1]} year gap)")
    return dates[:30], years, gaps


def generate_action_plan(spell, gram, verbs, secs_m, impact_pct,
                          contact_missing, bullet_res, ats_val):
    tips = []
    if contact_missing:
        tips.append(("🔴 Critical", f"Add missing contact info: {', '.join(contact_missing)}"))
    if secs_m:
        tips.append(("🔴 Critical", f"Add missing resume sections: {', '.join(secs_m[:3])}"))
    if ats_val is not None and ats_val < 60:
        tips.append(("🔴 Critical", f"ATS score is {ats_val}% — tailor your resume to the job description."))
    if spell:
        tips.append(("🟠 High", f"Fix {len(spell)} spelling error(s): {', '.join(i['word'] for i in spell[:5])}"))
    if impact_pct < 40:
        tips.append(("🟠 High", f"Only {impact_pct}% bullets are quantified. Add numbers, %, $ amounts."))
    weak_bullets = [b for b in bullet_res if b["score"] == 0]
    if weak_bullets:
        tips.append(("🟠 High", f"{len(weak_bullets)} bullet(s) score 0/3 — add action verbs and metrics."))
    no_verb = [b for b in bullet_res if not b["has_verb"]]
    if no_verb:
        tips.append(("🟡 Medium", f"{len(no_verb)} bullet(s) don't start with an action verb."))
    passive_count = sum(1 for t, _, _ in gram if "Passive" in t)
    if passive_count:
        tips.append(("🟡 Medium", f"Fix {passive_count} passive-voice sentence(s)."))
    long_count = sum(1 for t, _, _ in gram if "Long" in t)
    if long_count:
        tips.append(("🟡 Medium", f"{long_count} sentence(s) too long — split them up."))
    if sum(len(v) for v in verbs.values()) < 5:
        tips.append(("🟡 Medium", "Too few action verbs. Start every bullet with a power verb."))
    if not tips:
        tips.append(("🟢 Great", "Your resume looks solid! Polish remaining sections for a perfect score."))
    return tips


def compute_overall_score(secs_f, spell, gram, verbs, rd, impact_pct, contact_found):
    bd = {}
    bd["Sections"]     = (round(len(secs_f) / max(len(RESUME_SECTIONS), 1) * 15), 15)
    bd["Spelling"]     = (max(0, 15 - len(spell) * 2), 15)
    bd["Grammar"]      = (max(0, 10 - len(gram)), 10)
    vt = sum(len(v) for v in verbs.values())
    bd["Action Verbs"] = (min(15, vt * 2), 15)
    fk = rd.get("Flesch-Kincaid Grade", 15) if rd else 15
    bd["Readability"]  = (10 if 8 <= fk <= 12 else (7 if 5 <= fk <= 16 else 4), 10)
    bd["Impact"]       = (round(impact_pct / 100 * 20), 20)
    bd["Contact"]      = (round(len(contact_found) / max(len(CONTACT_PATTERNS), 1) * 15), 15)
    total = sum(v[0] for v in bd.values())
    return min(total, 100), bd


# ══════════════════════════════════════════════════════════════════
#  UI HELPERS
# ══════════════════════════════════════════════════════════════════

def badge(score: int):
    if score >= 80: return "🟢 Excellent"
    if score >= 65: return "🟡 Good"
    if score >= 50: return "🟠 Fair"
    return "🔴 Needs Work"

def chip(text, color="blue"):
    palette = {
        "blue":  ("#EEF2FF", "#3730a3"),
        "green": ("#D1FAE5", "#065f46"),
        "red":   ("#FEE2E2", "#991b1b"),
        "gray":  ("#F3F4F6", "#374151"),
    }
    bg, fg = palette.get(color, palette["blue"])
    return (f'<span style="background:{bg};color:{fg};border-radius:20px;'
            f'padding:2px 10px;margin:2px;font-size:0.80rem;'
            f'font-weight:500;display:inline-block;">{text}</span>')

def chips(items, color="blue"):
    return " ".join(chip(i, color) for i in items)


# ══════════════════════════════════════════════════════════════════
#  STREAMLIT  APP
# ══════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Resume Analyzer Pro", page_icon="📄", layout="wide")

st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size:1.5rem; font-weight:700; }
.tip-critical { background:#DC2626; border-left:4px solid #991b1b;
                padding:8px 12px; border-radius:4px; margin:4px 0;
                color:#ffffff !important; }
.tip-high     { background:#D97706; border-left:4px solid #92400e;
                padding:8px 12px; border-radius:4px; margin:4px 0;
                color:#ffffff !important; }
.tip-medium   { background:#2563EB; border-left:4px solid #1e3a8a;
                padding:8px 12px; border-radius:4px; margin:4px 0;
                color:#ffffff !important; }
.tip-great    { background:#059669; border-left:4px solid #064e3b;
                padding:8px 12px; border-radius:4px; margin:4px 0;
                color:#ffffff !important; }
</style>
""", unsafe_allow_html=True)

st.title(" 📄 Resume Analyzer Pro — Enhanced Edition")
st.caption("Fully Offline · NLP-Powered · No AI · No API · No Internet Required")

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("📁 Upload Resume")
    uploaded = st.file_uploader("PDF only", type=["pdf"])
    st.header("📌 Job Description (Optional)")
    jd = st.text_area("Paste JD for ATS Score", height=180,
                      placeholder="Paste the full job description here…")
    run_btn = st.button("🔍 Analyze Resume", use_container_width=True, type="primary")
    st.divider()
    st.markdown("""
**Tech Stack (100% offline)**
- `pdfplumber` — PDF text extraction
- `NLTK` — Tokenization & NLP
- `pyspellchecker` — Spell check
- `textstat` — Readability metrics
""")

# ── Landing Page ──────────────────────────────────────────────────
if not uploaded:
    c1, c2, c3, c4 = st.columns(4)
    c1.info("**🎯 ATS Scoring**\nKeyword match % against any job description")
    c2.info("**📊 Impact Check**\nQuantified bullet scoring (%, $, numbers)")
    c3.info("**🔵 Bullet Quality**\nPer-bullet verb + metric + length audit")
    c4.info("**💡 Action Plan**\nPrioritized fix list with exact recommendations")
    st.markdown("""
---
### ✅ All 15 Features
| # | Feature | What it checks |
|---|---|---|
| 1 | 🎯 ATS Score | Keyword % match with job description |
| 2 | 🔤 Spell Check | 170K-word dictionary, skips tech terms |
| 3 | 📐 Grammar | Passive voice, filler words, long sentences |
| 4 | 🔄 Tense Check | Flags present-tense verbs in experience |
| 5 | 💪 Action Verbs | 120 power verbs across 6 categories |
| 6 | 📊 Impact Metrics | % bullets with numbers/$/% quantification |
| 7 | 🔵 Bullet Quality | Per-bullet: verb ✓ metric ✓ length ✓ |
| 8 | 📋 Sections | 8 standard resume sections audit |
| 9 | ⚙️ Skills | 80+ tech + 25 soft skills detection |
| 10 | 📞 Contact Info | Email, phone, LinkedIn, GitHub validator |
| 11 | 🏭 Industry | Auto-detect industry + tailored keywords |
| 12 | 🔁 Word Frequency | Top 20 words + overused word detection |
| 13 | 📅 Timeline | Dates, years, employment gap detection |
| 14 | 💡 Action Plan | Priority-ranked specific improvements |
| 15 | 📖 Readability | Flesch-Kincaid grade + reading ease |

> 👈 Upload your resume PDF in the sidebar and click **Analyze Resume**.
""")
    st.stop()

# ── Run Analysis ──────────────────────────────────────────────────
if run_btn:
    with st.spinner("Extracting text from PDF…"):
        raw_bytes = uploaded.read()
        text = extract_text_from_pdf(raw_bytes)

    if not text.strip():
        st.error("❌ Could not extract text. The PDF may be image-based or password-protected.")
        st.stop()

    with st.spinner("Running full NLP analysis…"):
        ats_res                        = calculate_ats(text, jd)
        spell_issues                   = check_spelling(text)
        gram_issues                    = grammar_check(text)
        tense_issues                   = check_tense(text)
        verbs, weak_p, buzz_w, sugg    = action_verb_analysis(text)
        secs_f, secs_m                 = section_check(text)
        tech_sk, soft_sk               = extract_skills(text)
        rd                             = get_readability(text)
        impact_pct, q_bul, nq_bul      = check_impact(text)
        contact_f, contact_m           = validate_contact(text)
        bullet_res                     = analyze_bullets(text)
        top_industry, ind_data         = detect_industry(text)
        top_words, overused_w          = word_frequency(text)
        dates, years, gaps             = extract_timeline(text)
        ats_val                        = ats_res[0]
        action_plan                    = generate_action_plan(
                                             spell_issues, gram_issues, verbs,
                                             secs_m, impact_pct, contact_m,
                                             bullet_res, ats_val)
        o_score, breakdown             = compute_overall_score(
                                             secs_f, spell_issues, gram_issues,
                                             verbs, rd, impact_pct, contact_f)

    st.session_state["result"] = dict(
        text=text, ats=ats_res, spell=spell_issues, gram=gram_issues,
        tense=tense_issues, verbs=verbs, weak=weak_p, buzz=buzz_w, sugg=sugg,
        secs_f=secs_f, secs_m=secs_m, tech=tech_sk, soft=soft_sk, rd=rd,
        impact_pct=impact_pct, q_bul=q_bul, nq_bul=nq_bul,
        contact_f=contact_f, contact_m=contact_m, bullets=bullet_res,
        industry=top_industry, ind_data=ind_data, words=top_words,
        overused=overused_w, dates=dates, years=years, gaps=gaps,
        plan=action_plan, score=o_score, breakdown=breakdown,
    )
    st.success("✅ Analysis complete! Explore results in the tabs below.")

if "result" not in st.session_state:
    st.info("Upload a PDF and click **Analyze Resume** to start.")
    st.stop()

# ══════════════════════════════════════════════════════════════════
#  RESULTS DISPLAY
# ══════════════════════════════════════════════════════════════════
d = st.session_state["result"]

# ── Top Score Dashboard ───────────────────────────────────────────
st.subheader("📊 Resume Score Dashboard")
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("🏆 Overall",      f"{d['score']}/100", badge(d["score"]))
ats_v = d["ats"][0]
m2.metric("🎯 ATS Match",    f"{ats_v}%" if ats_v else "—",
          "paste JD" if not ats_v else ("✅" if ats_v >= 70 else "⚠️ low"))
m3.metric("📊 Impact Score", f"{d['impact_pct']}%",
          "✅ good" if d["impact_pct"] >= 60 else "⚠️ add metrics")
m4.metric("🔤 Spelling",     len(d["spell"]),
          "✅ clean" if not d["spell"] else "⚠️ errors found")
m5.metric("📋 Sections",     f"{len(d['secs_f'])}/{len(RESUME_SECTIONS)}",
          "✅ complete" if not d["secs_m"] else f"{len(d['secs_m'])} missing")
m6.metric("📞 Contact",      f"{len(d['contact_f'])}/{len(CONTACT_PATTERNS)}",
          "✅ complete" if len(d["contact_f"]) >= 3 else "⚠️ incomplete")

st.progress(d["score"] / 100)

with st.expander("📉 Score Breakdown by Category"):
    cols = st.columns(len(d["breakdown"]))
    for i, (name, (got, mx)) in enumerate(d["breakdown"].items()):
        with cols[i]:
            st.metric(name, f"{got}/{mx}")
            st.progress(got / mx)

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────
TABS = st.tabs([
    "💡 Action Plan",
    "🎯 ATS Score",
    "🔤 Spelling",
    "📐 Grammar & Tense",
    "💪 Action Verbs",
    "📊 Impact & Bullets",
    "📋 Sections",
    "⚙️ Skills",
    "📞 Contact Info",
    "🏭 Industry",
    "🔁 Word Analysis",
    "📅 Timeline",
    "📖 Readability",
    "📄 Raw Text",
])

# ─── 0 · Action Plan ─────────────────────────────────────────────
with TABS[0]:
    st.subheader("💡 Your Personalized Improvement Plan")
    st.caption("Issues ranked by priority — fix 🔴 Critical items first.")
    css_map = {"🔴 Critical": "tip-critical", "🟠 High": "tip-high",
               "🟡 Medium": "tip-medium", "🟢 Great": "tip-great"}
    for priority, tip in d["plan"]:
        st.markdown(
            f'<div class="{css_map.get(priority,"tip-medium")}">'
            f'<strong>{priority}</strong> — {tip}</div>',
            unsafe_allow_html=True
        )
    st.divider()
    st.markdown("**📌 Universal Resume Formula (for every bullet point):**")
    st.info(
        "**[Action Verb]** + **[What you did]** + **[Measurable result]**\n\n"
        "*Example: \"Reduced API latency by **42%** by rewriting database queries with indexed joins.\"*"
    )
    st.markdown("""
**Quick wins:**
- Tailor keywords for **every** application — copy exact phrases from the JD.
- Quantify **at least 50%** of bullet points (%, $, time saved, team size, etc.).
- Use **past tense** for all previous roles; present tense only for current job.
- Keep to **1–2 pages** with consistent formatting and readable fonts (≥10pt).
- Put your name + contact on **every page** header.
""")

# ─── 1 · ATS Score ───────────────────────────────────────────────
with TABS[1]:
    if not ats_v:
        st.info("📌 Paste a **Job Description** in the sidebar to calculate your ATS score.")
    else:
        ats_score, matched, missing, t_match, t_miss = d["ats"]
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("ATS Match Score", f"{ats_score}%")
            st.progress(ats_score / 100)
            if ats_score >= 80:    st.success("🟢 Excellent match!")
            elif ats_score >= 60:  st.warning("🟡 Good — add missing keywords")
            else:                  st.error("🔴 Low — heavily tailor your resume!")
            st.metric("Keywords Matched", len(matched))
            st.metric("Keywords Missing", len(missing))
        with c2:
            if matched:
                st.markdown("**✅ Matched Keywords**")
                st.markdown(chips(matched, "green"), unsafe_allow_html=True)
            if missing:
                st.markdown("**❌ Missing Keywords (add naturally into bullets)**")
                st.markdown(chips(missing, "red"), unsafe_allow_html=True)
            if t_match:
                st.markdown("**✅ Matched Tech Skills**")
                st.markdown(chips(t_match, "green"), unsafe_allow_html=True)
            if t_miss:
                st.markdown("**❌ Missing Tech Skills**")
                st.markdown(chips(t_miss, "red"), unsafe_allow_html=True)
        st.info("💡 Aim for >75% ATS match. Never keyword-stuff — work them into genuine bullet points.")

# ─── 2 · Spelling ────────────────────────────────────────────────
with TABS[2]:
    if not d["spell"]:
        st.success("✅ No spelling errors detected — great!")
    else:
        st.error(f"**{len(d['spell'])}** potential spelling issue(s) found.")
        for item in d["spell"]:
            with st.expander(f"❌  {item['word']}"):
                st.write("**Suggestions:** " + (", ".join(item["suggestions"]) or "No suggestions (may be a proper noun or tech term)."))
        st.caption("⚠️ Tech terms, abbreviations, and proper nouns may be incorrectly flagged — review manually.")

# ─── 3 · Grammar & Tense ─────────────────────────────────────────
with TABS[3]:
    g_tab, t_tab = st.tabs(["📐 Grammar & Style", "🔄 Tense Consistency"])

    with g_tab:
        issues = d["gram"]
        if not issues:
            st.success("✅ No grammar issues detected!")
        else:
            cnt = Counter(i[0] for i in issues)
            for t, c in sorted(cnt.items()):
                st.write(f"**{t}** — {c} instance(s)")
            st.divider()
            for typ, sent, tip in issues:
                with st.expander(f"{typ}: *{sent[:70]}…*"):
                    st.write(f"**Found:** {sent}")
                    st.info(f"💡 **Fix:** {tip}")

    with t_tab:
        tense = d["tense"]
        if not tense:
            st.success("✅ Tense looks consistent!")
        else:
            st.warning(f"**{len(tense)}** sentence(s) may use present tense where past tense is expected.")
            for t in tense:
                with st.expander(f"🔄 `{t['verb']}` — *{t['sentence'][:60]}…*"):
                    st.write(f"**Sentence:** {t['sentence']}")
                    st.info(f"💡 {t['fix']}")
            st.caption("Past tense is standard for previous roles. Present tense is only OK for your current/ongoing job.")

# ─── 4 · Action Verbs ────────────────────────────────────────────
with TABS[4]:
    found_v = d["verbs"]
    total_v = sum(len(v) for v in found_v.values())
    if found_v:
        st.success(f"✅ {total_v} strong action verbs found across {len(found_v)} categories.")
        for cat, verbs in found_v.items():
            st.markdown(f"**{cat}**: " + chips(verbs, "green"), unsafe_allow_html=True)
    else:
        st.warning("No strong action verbs detected. Start every bullet point with a power verb!")

    st.divider()
    st.subheader("💡 Suggested Verbs to Add")
    for cat, verbs in d["sugg"].items():
        if verbs:
            st.markdown(f"**{cat}**: " + chips(verbs, "blue"), unsafe_allow_html=True)

    if d["weak"]:
        st.divider()
        st.subheader("⚠️ Weak Phrases — Replace These")
        for p in d["weak"]:
            st.error(f'❌  "{p}"  →  Replace with a strong action verb')

    if d["buzz"]:
        st.divider()
        st.subheader("🚫 Overused Clichés")
        for b in d["buzz"]:
            st.warning(f'⚠️  "{b}"  →  Be specific. Give a real example instead.')

# ─── 5 · Impact & Bullets ────────────────────────────────────────
with TABS[5]:
    imp_tab, bul_tab = st.tabs(["📊 Impact / Quantification", "🔵 Bullet Quality"])

    with imp_tab:
        st.metric("Quantification Score", f"{d['impact_pct']}%",
                  "✅ great" if d["impact_pct"] >= 60 else "⚠️ needs improvement")
        st.progress(d["impact_pct"] / 100)
        if d["impact_pct"] >= 60:
            st.success("✅ More than half your bullets have quantifiable impact.")
        elif d["impact_pct"] >= 40:
            st.warning("🟡 Some bullets have metrics — aim for 60%+.")
        else:
            st.error("🔴 Most bullets lack metrics. Add numbers, %, or $ amounts.")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**✅ Bullets WITH Metrics**")
            for b in d["q_bul"][:10]:
                st.success(b[:110])
        with c2:
            st.markdown("**❌ Bullets NEEDING Metrics**")
            for b in d["nq_bul"][:10]:
                st.error(b[:110])

        st.info(
            "**How to add metrics:**\n"
            "- *\"Improved performance\"* → *\"Improved performance by **35%**\"*\n"
            "- *\"Handled support\"* → *\"Resolved **50+ tickets/day** with **98% CSAT**\"*\n"
            "- *\"Led team\"* → *\"Led **8-person** team, shipped **2 weeks** early\"*"
        )

    with bul_tab:
        bullets = d["bullets"]
        if not bullets:
            st.info("No bullet points detected.")
        else:
            perfect = [b for b in bullets if b["score"] == 3]
            good    = [b for b in bullets if b["score"] == 2]
            needs   = [b for b in bullets if b["score"] <= 1]

            bc1, bc2, bc3 = st.columns(3)
            bc1.metric("🌟 Perfect (3/3)",   len(perfect))
            bc2.metric("✅ Good (2/3)",       len(good))
            bc3.metric("⚠️ Needs Work (≤1)", len(needs))
            st.caption("Criteria: ✅ Starts with action verb  ✅ Contains metric  ✅ Good length (8–30 words)")
            st.divider()

            def bullet_row(b):
                return (f"{'✅' if b['has_verb'] else '❌'} Verb  "
                        f"{'✅' if b['has_impact'] else '❌'} Metric  "
                        f"{'✅' if b['good_len'] else '❌'} Length  —  {b['text']}")

            if needs:
                st.markdown("**⚠️ Needs Improvement**")
                for b in needs[:12]:
                    st.error(bullet_row(b))
            if good:
                with st.expander(f"✅ Good Bullets ({len(good)})"):
                    for b in good[:12]:
                        st.warning(bullet_row(b))
            if perfect:
                with st.expander(f"🌟 Perfect Bullets ({len(perfect)})"):
                    for b in perfect[:12]:
                        st.success(bullet_row(b))

# ─── 6 · Sections ────────────────────────────────────────────────
with TABS[6]:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader(f"✅ Present ({len(d['secs_f'])})")
        for s in d["secs_f"]:
            st.success(s)
    with c2:
        st.subheader(f"❌ Missing ({len(d['secs_m'])})")
        if d["secs_m"]:
            for s in d["secs_m"]:
                st.error(s)
            st.caption("Missing sections lower your ATS score and recruiter impression.")
        else:
            st.success("🎉 All sections present!")
    st.divider()
    st.info("**Recommended order:** Contact → Summary → Experience → Education → Skills → Projects → Certifications → Awards")

# ─── 7 · Skills ──────────────────────────────────────────────────
with TABS[7]:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader(f"💻 Technical Skills ({len(d['tech'])})")
        if d["tech"]:
            for sk in d["tech"]: st.write(f"✅  {sk}")
        else:
            st.warning("No tech skills detected. Add a dedicated Skills section.")
    with c2:
        st.subheader(f"🤝 Soft Skills ({len(d['soft'])})")
        if d["soft"]:
            for sk in d["soft"]: st.write(f"✅  {sk}")
        else:
            st.warning("No soft skills detected from our database.")

    if d["industry"] and d["ind_data"]:
        st.divider()
        st.subheader(f"💡 Skills Recommended for {d['industry']}")
        tl   = d["text"].lower()
        ms   = d["ind_data"].get("must_skills", [])
        miss = [s for s in ms if s not in tl]
        if miss:
            st.markdown("**❌ Must-Have Skills Missing:**")
            st.markdown(chips(miss, "red"), unsafe_allow_html=True)
        else:
            st.success("✅ All must-have skills for your industry are present!")

# ─── 8 · Contact Info ────────────────────────────────────────────
with TABS[8]:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader(f"✅ Found ({len(d['contact_f'])})")
        for label, value in d["contact_f"].items():
            st.success(f"**{label}**: `{value}`")
    with c2:
        st.subheader(f"❌ Missing ({len(d['contact_m'])})")
        if d["contact_m"]:
            for label in d["contact_m"]:
                st.error(f"{label} not detected")
        else:
            st.success("🎉 All contact fields found!")
    st.divider()
    st.info("""
**Best Practices:**
- All contact info belongs in the **resume header** (very top).
- Customize your LinkedIn URL: `linkedin.com/in/yourname`
- GitHub is essential for **tech roles**.
- Use a professional email (avoid nicknames/numbers).
- Add country code for phone if applying internationally.
""")

# ─── 9 · Industry ────────────────────────────────────────────────
with TABS[9]:
    if not d["industry"]:
        st.warning("Could not confidently detect your industry. Ensure your resume has clear role titles and relevant keywords.")
    else:
        st.subheader(f"🏭 Detected Industry: **{d['industry']}**")
        ind = d["ind_data"]
        tl  = d["text"].lower()
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**💪 Power Words for Your Industry**")
            st.markdown(chips(ind.get("power_words", []), "green"), unsafe_allow_html=True)
        with c2:
            extra = [k for k in ind.get("extra_keywords", []) if k not in tl]
            st.markdown("**🔑 Suggested Keywords to Add**")
            st.markdown(chips(extra, "red") if extra else "✅ All extra keywords present!", unsafe_allow_html=True)
        st.divider()
        ms      = ind.get("must_skills", [])
        have    = [s for s in ms if s in tl]
        havenot = [s for s in ms if s not in tl]
        st.markdown("**Must-Have Skills Present:** " + (chips(have, "green") if have else "*None*"), unsafe_allow_html=True)
        st.markdown("**Must-Have Skills Missing:** " + (chips(havenot, "red") if havenot else "✅ All present!"), unsafe_allow_html=True)
        st.info("Tip: Explicitly mention your target role title in your summary/objective section.")

# ─── 10 · Word Analysis ──────────────────────────────────────────
with TABS[10]:
    st.subheader("🔤 Top 20 Most-Used Words")
    top_w = d["words"]
    if top_w:
        max_f = top_w[0][1]
        for word, freq in top_w:
            c1, c2, c3 = st.columns([2, 5, 1])
            c1.write(f"**{word}**")
            c2.progress(freq / max_f)
            c3.write(str(freq))

    if d["overused"]:
        st.divider()
        st.subheader("⚠️ Overused Words (4+ times)")
        st.caption("Repetition makes your resume monotonous. Replace some instances with synonyms.")
        for word, count in sorted(d["overused"].items(), key=lambda x: -x[1]):
            st.warning(f'**`{word}`** appears **{count} times** — consider replacing some occurrences.')
    else:
        st.success("✅ No significantly overused words detected.")

# ─── 11 · Timeline ───────────────────────────────────────────────
with TABS[11]:
    st.subheader("📅 Career Timeline")
    if d["years"]:
        st.markdown("**📆 Years Detected**")
        st.markdown(chips([str(y) for y in d["years"]], "blue"), unsafe_allow_html=True)
    if d["dates"]:
        st.markdown("**🗓 Date Mentions**")
        unique_dates = list(dict.fromkeys(d["dates"]))
        st.markdown(chips(unique_dates[:20], "gray"), unsafe_allow_html=True)
    if d["gaps"]:
        st.divider()
        st.subheader("⚠️ Potential Employment Gaps")
        for g in d["gaps"]:
            st.error(f"🕳  {g}")
        st.caption("Prepare to address gaps. Consider adding freelance work, courses, or projects to fill them.")
    else:
        st.success("✅ No major employment gaps detected (>2 years).")
    st.divider()
    st.info("""
**Date Formatting Tips:**
- Use: `Jan 2022 – Present` or `January 2022 – March 2024`
- Write **"Present"** for your current role, not today's date.
- Include both month AND year for recent positions.
""")

# ─── 12 · Readability ────────────────────────────────────────────
with TABS[12]:
    rd = d["rd"]
    if not rd:
        st.warning("Could not compute readability metrics.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.metric("📝 Word Count",        rd["Word Count"])
            st.metric("📄 Sentence Count",     rd["Sentence Count"])
            st.metric("📏 Avg Words/Sentence", rd["Avg Words/Sentence"])
        with c2:
            fk = rd["Flesch-Kincaid Grade"]
            st.metric("🎓 Flesch-Kincaid Grade", fk)
            st.metric("📖 Flesch Reading Ease",  rd["Flesch Reading Ease"])
            if 8 <= fk <= 12:    st.success("✅ Ideal reading level for resumes (Grade 8–12).")
            elif fk > 12:        st.warning("⚠️ Text is too complex — simplify your language.")
            else:                st.warning("⚠️ Text may be too simple — use more professional vocabulary.")
        st.divider()
        wc = rd["Word Count"]
        if wc < 300:
            st.error(f"⚠️ Only {wc} words — too short. Aim for 450–700 words (1-page resume).")
        elif wc > 900:
            st.warning(f"⚠️ {wc} words — possibly too long. Trim to 2 pages max.")
        else:
            st.success(f"✅ {wc} words — good length.")
        st.info("""
**Guide:**
- **Flesch Reading Ease**: 60–70 ideal (100 = easiest).
- **F-K Grade**: 8–12 perfect. Grade 15+ is too complex.
- **Word Count**: 400–700 for 1 page; up to 900 for 2 pages.
- **Avg Words/Sentence**: Keep under 20 for clarity.
""")

# ─── 13 · Raw Text ───────────────────────────────────────────────
with TABS[13]:
    st.subheader("📄 Extracted Resume Text")
    st.caption("This is exactly what NLP tools and ATS systems see from your PDF.")
    st.text_area("", d["text"], height=500, label_visibility="collapsed")
