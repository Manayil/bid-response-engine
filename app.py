import streamlit as st
import pandas as pd
import pdfplumber
import docx
from groq import Groq
import io
import os
import ast
from dotenv import load_dotenv
from vector_engine import search_library

# 1. INITIALIZATION AND CONFIGURATION
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)

st.set_page_config(page_title="Bid and Proposal Response Engine", layout="wide")

# Workspace session state management (PRD 6.7)
if 'workspaces' not in st.session_state:
    st.session_state['workspaces'] = {}
if 'active_rfp' not in st.session_state:
    st.session_state['active_rfp'] = None

# 2. DATA LOADING
@st.cache_data
def load_datasets():
    try:
        history = pd.read_csv('data/bid_history.csv')
        library = pd.read_csv('data/capability_library.csv')
        return history, library
    except Exception as e:
        st.error(f"Initialization error: Data files not found in /data directory. {e}")
        return None, None

history_df, library_df = load_datasets()

# 3. CORE LOGIC FUNCTIONS
def read_file_content(uploaded_file):
    """Handles multi-format ingestion for PDF and DOCX (PRD 6.1)."""
    if uploaded_file.name.endswith('.pdf'):
        with pdfplumber.open(uploaded_file) as pdf:
            return "".join([page.extract_text() or "" for page in pdf.pages])
    elif uploaded_file.name.endswith('.docx'):
        doc = docx.Document(uploaded_file)
        return "\n".join([para.text for para in doc.paragraphs])
    return None

def call_llm(prompt):
    """Standardized LLM interface."""
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Service Error: {e}"

def generate_word_document(text, rfp_name):
    """Exports structured proposal as a DOCX file (PRD 6.7)."""
    doc = docx.Document()
    doc.add_heading(f"Proposal Response: {rfp_name}", 0)
    doc.add_paragraph(text)
    buffer = io.BytesIO()
    doc.save(buffer)
    return buffer.getvalue()

# 4. USER INTERFACE - SIDEBAR
with st.sidebar:
    st.header("Workspace Controls")
    file_upload = st.file_uploader("Upload RFP (PDF or DOCX)", type=["pdf", "docx"])
    
    if file_upload:
        name = file_upload.name
        if name not in st.session_state['workspaces']:
            with st.status(f"Processing {name}...") as status:
                # Document Parsing
                content = read_file_content(file_upload)
                
                # Metadata & Sector Extraction (PRD 6.2)
                available_domains = library_df['Domain'].unique().tolist()
                meta_prompt = f"""
                Analyze the following text and provide a JSON response:
                1. 'domain': Choose one from {available_domains}
                2. 'deadline': Extraction of submission date
                3. 'budget': Extraction of estimated project value (PKR)
                4. 'requirements': List of 5 mandatory technical clauses
                
                Text: {content[:3000]}
                """
                meta_raw = call_llm(meta_prompt)
                try:
                    # Parse structured JSON from AI
                    start = meta_raw.find("{")
                    end = meta_raw.rfind("}") + 1
                    extracted = ast.literal_eval(meta_raw[start:end])
                except:
                    extracted = {"domain": "Unspecified", "deadline": "TBD", "budget": "TBD", "requirements": []}

                # Compliance RAG Mapping (PRD 6.3 & 6.5)
                compliance_map = []
                pass_total = 0
                for req in extracted.get('requirements', []):
                    search_res = search_library(req)
                    is_match = search_res['distances'][0][0] < 1.1
                    if is_match: pass_total += 1
                    compliance_map.append({
                        "Requirement": req,
                        "Status": "PASS" if is_match else "FAIL",
                        "Evidence": search_res['documents'][0][0] if is_match else "No internal evidence found"
                    })

                # Weighted Heuristic Scoring (PRD 6.6)
                hist_subset = history_df[history_df['Sector'] == extracted['domain']]
                hist_rate = (len(hist_subset[hist_subset['Outcome'] == 'Win']) / len(hist_subset)) * 100 if not hist_subset.empty else 50.0
                comp_rate = (pass_total / len(extracted['requirements'])) * 100 if extracted['requirements'] else 0
                aggregate_score = (hist_rate * 0.4) + (comp_rate * 0.6)

                # Store workspace data
                st.session_state['workspaces'][name] = {
                    "text": content,
                    "domain": extracted['domain'],
                    "deadline": extracted['deadline'],
                    "budget": extracted['budget'],
                    "compliance": compliance_map,
                    "score": aggregate_score,
                    "history": hist_rate
                }
                status.update(label="Workspace Analysis Ready", state="complete")
        
        st.session_state['active_rfp'] = name

    if st.session_state['workspaces']:
        st.divider()
        st.subheader("Active Projects")
        for ws_key in st.session_state['workspaces'].keys():
            if st.button(ws_key, use_container_width=True):
                st.session_state['active_rfp'] = ws_key

# 5. USER INTERFACE - MAIN CONTENT
st.title("Bid and Proposal Response Engine")

if st.session_state['active_rfp']:
    active = st.session_state['workspaces'][st.session_state['active_rfp']]
    
    # Header Statistics
    st.subheader(f"Analysis: {st.session_state['active_rfp']}")
    metric_1, metric_2, metric_3 = st.columns(3)
    metric_1.metric("Win Probability", f"{active['score']:.1f}%")
    metric_2.metric("Submission Deadline", active['deadline'])
    metric_3.metric("Project Budget", active['budget'])

    # Section Tabs
    tab_analytics, tab_compliance, tab_draft = st.tabs(["Strategic Analytics", "Compliance Matrix", "Proposal Draft"])

    with tab_analytics:
        st.write(f"The calculated win probability of {active['score']:.1f}% is based on a {active['history']:.1f}% historical success rate in the {active['domain']} sector.")
        
        # Decision rationale (PRD 6.6)
        if active['score'] >= 75:
            st.success("Decision: GO. System identifies strong capability alignment and historical performance.")
        elif active['score'] >= 50:
            st.warning("Decision: REVIEW REQUIRED. Potential compliance gaps identified in the library search.")
        else:
            st.error("Decision: NO-GO. High risk due to insufficient evidence and poor historical win rates.")

    with tab_compliance:
        st.subheader("Requirement Pass/Fail Mapping")
        st.table(active['compliance'])

    with tab_draft:
        st.subheader("Automated Response Narrative")
        # Narrative generation grounded in RAG evidence (PRD 6.4)
        evidence_snippet = active['compliance'][0]['Evidence'] if active['compliance'] else "general capabilities"
        draft_prompt = f"Draft a professional technical proposal section for {active['domain']} sector. Context: {active['text'][:1500]}. Ground the response in our evidence: {evidence_snippet}."
        
        proposal_text = call_llm(draft_prompt)
        st.write(proposal_text)
        
        # Word Export Action
        word_data = generate_word_document(proposal_text, st.session_state['active_rfp'])
        st.download_button(
            label="Export Structured Proposal (.docx)",
            data=word_data,
            file_name=f"Proposal_{active['domain']}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

else:
    st.info("System awaiting document ingestion. Upload an RFP via the sidebar to initiate analysis.")

st.divider()
st.caption("Bid and Proposal Response Engine | Documentation Version 1.0 | PRD Compliant")