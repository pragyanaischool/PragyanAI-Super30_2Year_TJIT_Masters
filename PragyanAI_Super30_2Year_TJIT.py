import streamlit as st
import pandas as pd
import requests
import io
import logging
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

# --- Page Configuration ---
st.set_page_config(
    page_title="PragyanAI - 2-Year Master's Certificate Program in AI & GenAI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- SESSION STATE INITIALIZATION ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if 'user_name' not in st.session_state:
    st.session_state.user_name = ""
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'chat_dialog_open' not in st.session_state:
    st.session_state.chat_dialog_open = False

# --- Custom CSS for Enhanced Styling ---
st.markdown("""
<style>
    /* General Styles */
    .stApp {
        background-color: #0c111e;
        color: #e2e8f0;
    }
    h1, h2, h3, h4 {
        color: #ffffff;
        font-weight: 700;
    }
    p, .stMarkdown, li {
        color: #e2e8f0;
        font-size: 1.1rem;
    }
    .stButton>button {
        background: linear-gradient(90deg, #f97316, #ea580c);
        color: white;
        border-radius: 0.5rem;
        padding: 0.85rem 1.6rem;
        font-weight: 700;
        border: none;
        transition: all 0.3s ease-in-out;
        box-shadow: 0 4px 15px 0 rgba(249, 115, 22, 0.4);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px 0 rgba(249, 115, 22, 0.5);
    }
    .stLinkButton>a {
        background-color: transparent;
        color: #f97316 !important;
        border: 2px solid #f97316;
        border-radius: 0.5rem;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s;
        text-decoration: none;
    }
    .stLinkButton>a:hover {
        background-color: rgba(249, 115, 22, 0.1);
        border-color: #ea580c;
    }
    
    /* --- Custom UI Component Styles --- */
    .section-header { text-align: center; margin-bottom: 3rem; }
    .gradient-text {
        background: linear-gradient(to right, #fb923c, #f97316);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .highlight-card {
        background: #1e293b; padding: 2rem; border-radius: 1rem;
        border: 1px solid #334155; text-align: center; height: 100%;
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
    }
    .benefit-card {
        background: rgba(30, 41, 59, 0.5); padding: 1.5rem; border-radius: 1rem;
        border-left: 4px solid #f97316; margin-bottom: 1rem;
    }
    .comparison-card {
        background: #1e293b; padding: 1.5rem; border-radius: 1rem; height: 100%;
    }
    .instructor-card {
        background: #1e293b; padding: 2rem; border-radius: 1rem;
        text-align: center;
    }
    .footer {
        border-top: 1px solid #334155; padding: 2rem 0; margin-top: 4rem; text-align: center; color: #94a3b8;
    }
    
    /* Tab Styles */
    .stTabs [data-baseweb="tab-list"] {
		gap: 24px;
        justify-content: center;
	}
	.stTabs [data-baseweb="tab"] {
		height: 50px;
        background-color: transparent;
		padding: 0 20px;
        font-weight: 600;
        font-size: 1.1rem;
        color: #94a3b8;
	}
	.stTabs [aria-selected="true"] {
  		background-color: #1e293b;
        color: #ffffff;
        border-bottom: 3px solid #f97316;
	}
</style>
""", unsafe_allow_html=True)


# --- Data Loading and Caching Functions (Keep as is) ---
@st.cache_data(ttl=3600)
def load_faq_data(sheet_url):
    """Loads Q&A data from a public Google Sheet."""
    try:
        csv_url = sheet_url.replace("/edit?usp=sharing", "/export?format=csv")
        df = pd.read_csv(csv_url)
        if df.shape[1] < 2: return None
        df = df.iloc[:, [0, 1]]; df.columns = ['FAQs', 'Answere']; df.dropna(inplace=True)
        return df
    except Exception as e:
        st.error(f"Failed to load FAQ data: {e}", icon="📄")
        return None

@st.cache_data(ttl=3600)
def load_pdf_from_gdrive(drive_url):
    """Downloads and extracts text from a PDF in Google Drive."""
    try:
        file_id = drive_url.split('/d/')[1].split('/')[0]
        download_url = f'https://drive.google.com/uc?export=download&id={file_id}'
        response = requests.get(download_url)
        response.raise_for_status()
        pdf_file = io.BytesIO(response.content)
        reader = PdfReader(pdf_file)
        text = "".join(page.extract_text() for page in reader.pages)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_text(text)
        return [Document(page_content=split, metadata={"source": "program_brochure"}) for split in splits]
    except Exception as e:
        st.error(f"Failed to load PDF brochure: {e}", icon="📄")
        return None

@st.cache_resource
def get_embeddings_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource
def create_vector_store(_faq_df, _pdf_docs, _embeddings):
    documents = []
    if _faq_df is not None:
        documents.extend([Document(page_content=f"Question: {row['FAQs']}\nAnswer: {row['Answere']}", metadata={"source": "faq_sheet"}) for _, row in _faq_df.iterrows()])
    if _pdf_docs is not None:
        documents.extend(_pdf_docs)
    return FAISS.from_documents(documents, _embeddings) if documents else None

# --- RAG Chain & Pre-loading ---
def get_history(_): return "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.get('messages', [])])
def get_name(_): return st.session_state.get('user_name', "")

def initialize_knowledge_base():
    if st.session_state.vector_store is None:
        with st.spinner("Initializing AI Advisor..."):
            faq_sheet_url = "https://docs.google.com/spreadsheets/d/14NTraereEwWwLyhycjCP0TKJ2-a6eY38xjy5EbAN-jM/edit?usp=sharing"
            # UPDATED BROCHURE LINK
            brochure_drive_url = "https://drive.google.com/file/d/177VRFLhiyCC_xJ8idVBOhCPYcM6sz6un/view?usp=sharing"
            faq_data = load_faq_data(faq_sheet_url)
            pdf_docs = load_pdf_from_gdrive(brochure_drive_url)
            if faq_data is not None or pdf_docs is not None:
                embeddings = get_embeddings_model()
                st.session_state.vector_store = create_vector_store(faq_data, pdf_docs, embeddings)

initialize_knowledge_base()

# --- CHAT DIALOG FUNCTION (Keep as is) ---
@st.dialog("AI Marketing Advisor")
def run_chat_dialog():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]): st.markdown(message["content"])
    if prompt := st.chat_input("Ask about curriculum, placements, fees..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            try:
                retriever = st.session_state.vector_store.as_retriever(search_kwargs={'k': 5})
                groq_api_key = st.secrets["GROQ_API_KEY"]
                llm = ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=groq_api_key)
                system_prompt_template = """You are an expert academic advisor for the Pragyan AI Master's Certificate Program for college students. Your name is PragyanAI Bot. Your goal is to advise a student named {user_name} on why this 2-year program is a critical step for their future career.
                Follow these rules strictly:
                1. Persona: Be persuasive, encouraging, and an expert on AI career paths for graduates. Frame every answer to highlight the *value and benefits* for a student.
                2. Acknowledge and Address: Always address the user by their name, {user_name}.
                3. Data Source: Base your answers ONLY on the retrieved context from the program brochure and FAQ sheet.
                4. Value Proposition: Don't just state facts. Explain the 'so what'. For example, if asked about the 100+ projects, explain that this provides a massive, undeniable portfolio that makes them stand out against peers with only 1-2 academic projects.
                5. Context is Key: Use the conversation history to maintain a natural, flowing dialogue.
                6. Handle Unknowns: If the context doesn't have the answer, you MUST say: "That's an excellent and very specific question, {user_name}. While I don't have the details on that, it's something our program director can certainly clarify. I highly recommend reaching out to them at pragyan.ai.school@gmail.com for a direct answer." Do not invent information.
                <CONVERSATION_HISTORY>{chat_history}</CONVERSATION_HISTORY>
                <RETRIEVED_CONTEXT>{context}</RETRIEVED_CONTEXT>
                Given the context and conversation history, answer the user's question: {user_question}"""
                prompt_template = ChatPromptTemplate.from_template(system_prompt_template)
                rag_chain = (RunnableParallel({"context": retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs)), "user_question": RunnablePassthrough(), "chat_history": RunnableLambda(get_history), "user_name": RunnableLambda(get_name)}) | prompt_template | llm | StrOutputParser())
                response = rag_chain.invoke(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()
            except Exception as e:
                logging.error(f"AI Assistant Error: {e}")
                st.error("I'm sorry, I encountered a connection issue. Please check your API key or try again later.")

# --- UI LAYOUT ---
# Header
col1, col2, col3 = st.columns([2, 5, 2])
with col2:
    st.image("PragyanAI_Transperent_github.png", use_container_width=True)

# Hero Section
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<h1 style="text-align: center;">Build Your Future with the <span class="gradient-text">AI Leaders Program</span></h1>', unsafe_allow_html=True)
st.markdown('<h3 style="text-align: center; color: #94a3b8;">The 2-Year Master\'s Certificate Program in Data Science, AI, and GenAI</h3>', unsafe_allow_html=True)
st.markdown("""<p style="text-align: center; max-width: 800px; margin: auto; padding-top: 1rem;">
    In an industry where thousands of engineers graduate every year, a basic degree is not enough. This program is your unfair advantage—a deep, hands-on journey to gain the skills that will make you a top-tier candidate for the best AI roles, right out of college.
</p>""", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# --- TABBED INTERFACE ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["**Program Overview**", "**Curriculum & Roadmap**", "**The PragyanAI Advantage**", "**Career Outcomes**", "**Ask the AI Advisor**"])

with tab1:
    st.markdown("### The Unbeatable Edge for an Elite AI Career")
    st.markdown('<div class="benefit-card"><strong>Hybrid Learning Model:</strong> Combines intensive, in-person sessions at your college with a structured, live online curriculum for over 1200+ hours of deep learning.</div>', unsafe_allow_html=True)
    st.markdown('<div class="benefit-card"><strong>Outcome-Based Finance:</strong> Our success is tied to yours. With a low semester fee and an income-linked model, we are invested in securing you a high-value placement.</div>', unsafe_allow_html=True)
    st.markdown('<div class="benefit-card"><strong>Massive Project Portfolio:</strong> Graduate with 120+ projects, 4 hackathons, 3 research papers, and multiple real-world capstones. You won\'t just talk about your skills—you\'ll prove them.</div>', unsafe_allow_html=True)
    
    st.markdown("### Program at a Glance")
    cols = st.columns(4)
    highlights = {"🎓 **Level**": "M.Tech Equivalent", "📅 **Duration**": "2 Years", "💻 **Format**": "Hybrid (Online + Offline)", "👥 **Audience**": "Engineering Students"}
    for title, value in highlights.items():
        with cols.pop(0):
            st.markdown(f'<div class="highlight-card"><h4>{title}</h4><p style="font-size: 1.5rem; color: #f97316; font-weight: 600;">{value}</p></div>', unsafe_allow_html=True)

with tab2:
    st.markdown("### Your Semester-wise Roadmap to AI Mastery")
    st.image("https://i.imgur.com/G2Yv4G3.png", caption="The 4-Semester AI Skills Acceleration Roadmap")

    semester_details = {
        "Semester 5: Foundations": ["Skill 1: Python Full Stack & Data Wrangling", "Skill 2: Data Science Foundation & Advanced Wrangling"],
        "Semester 6: Core Machine Learning": ["Skill 3: BI / Data Analytics", "Skill 4: Machine Learning & Advanced ML"],
        "Semester 7: Deep Learning & NLP": ["Skill 5: Deep Learning & Computer Vision (including Advanced DL & CV)", "Skill 6: NLP & AI Bot Building"],
        "Semester 8: Advanced AI & Deployment": ["Skill 7: Generative AI (LLMs, RAG, Fine-Tuning)", "Skill 8: Agentic AI & AI-Powered MVP Development", "Career Prep & Placements"]
    }
    for semester, skills in semester_details.items():
        with st.expander(f"**{semester}**"):
            for skill in skills:
                st.markdown(f"- {skill}")

with tab3:
    st.markdown("### Why PragyanAI Graduates Outperform the Competition")
    st.markdown("We don't just teach—we transform. This program is built to give you a decisive edge in the hyper-competitive tech job market.")
    st.markdown("---")

    # Comparison Section
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="comparison-card"><h4>Typical College Graduate</h4><ul><li><strong>Curriculum:</strong> Foundational AI/ML theory, limited exposure to latest concepts.</li><li><strong>Experience:</strong> 1-2 major academic projects.</li><li><strong>Tools:</strong> Basic ML libraries, often without a deployment focus.</li><li><strong>Support:</strong> Generic placement support.</li></ul></div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="comparison-card" style="border: 2px solid #f97316;"><h4><span class="gradient-text">PragyanAI MCP Graduate</span></h4><ul><li><strong>Curriculum:</strong> 8 core skills including GenAI, Agentic AI, MLOps. Equivalent to 12-15 advanced subjects.</li><li><strong>Experience:</strong> <strong>120+ projects</strong>, 4 hackathons, 3 research papers, multiple real-world capstones.</li><li><strong>Tools:</strong> Mastery of <strong>60+ industry-standard tools</strong> (Hugging Face, PyTorch, LangChain, etc.).</li><li><strong>Support:</strong> <strong>Impact Assured Guarantee</strong> (Targeted ₹50K-1 Lakh higher CTC), with dedicated mentorship.</li></ul></div>', unsafe_allow_html=True)

with tab4:
    st.markdown("### Launch Your Career with Elite Roles")
    st.markdown("This program prepares you for high-impact roles that are defining the future of technology, making you a top candidate for the best companies.")
    
    st.subheader("Target Roles You Will Be Qualified For:")
    roles = ["AI Solution Architect", "MLOps Engineer", "Advanced Data Scientist", "AI Product Manager", "Generative AI Specialist"]
    cols = st.columns(len(roles))
    for i, role in enumerate(roles):
        with cols[i]:
            st.markdown(f'<div class="highlight-card" style="padding: 1.5rem;"><strong>{role}</strong></div>', unsafe_allow_html=True)
            
    st.subheader("Hiring Partners & Target Companies:")
    st.markdown("Graduates with these advanced skills are highly sought after by a wide range of companies, from Big Tech to high-growth AI startups.")
    st.markdown("""
    - **Big Tech (Product-Based Companies):** Google, Microsoft, Amazon, Meta
    - **IT Services & Consulting Giants:** TCS, Infosys, Accenture, Deloitte
    - **Global Capability Centers (GCCs):** Target, Walmart Labs, Major Financial Institutions
    - **High-Growth AI Startups & Unicorns**
    """)

with tab5:
    st.markdown('<h2 class="section-header">Have Questions? Ask our AI Academic Advisor</h2>', unsafe_allow_html=True)
    try:
        st.secrets["GROQ_API_KEY"]
        api_key_present = True
    except (KeyError, FileNotFoundError):
        api_key_present = False

    if not api_key_present:
        st.warning("`GROQ_API_KEY` not found in Streamlit secrets. The Q&A bot is disabled.", icon="⚠️")
    elif st.session_state.vector_store is None:
        st.warning("The knowledge base for the AI Advisor could not be loaded. Please check data sources.", icon="🧠")
    else:
        st.session_state.user_name = st.text_input("Please enter your name to start the chat:", st.session_state.user_name)
        if st.session_state.user_name:
            if st.button("🚀 Launch AI Advisor"):
                st.session_state.chat_dialog_open = True
        
        if st.session_state.chat_dialog_open:
            run_chat_dialog()
            
# --- Call to Action & Footer ---
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown('<div class="highlight-card" style="background: linear-gradient(90deg, #334155, #1e293b);">', unsafe_allow_html=True)
st.markdown('<h2 class="section-header">Secure Your Future in the Age of AI</h2>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center;">Enrollment is now open. Take the decisive step to lead in the new era of technology.</p>', unsafe_allow_html=True)
cta_cols = st.columns([1, 1])
with cta_cols[0]:
    st.link_button("📄 View Program Brochure", "https://drive.google.com/file/d/177VRFLhiyCC_xJ8idVBOhCPYcM6sz6un/view?usp=sharing", use_container_width=True)
with cta_cols[1]:
    st.link_button("📝 Express Interest Now", "https://forms.gle/YLKzVeEPsy685KvJA", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""<div class="footer">
    <p>© 2025 Pragyan AI. All rights reserved.</p>
    <p><strong>Contact:</strong> 
        <a href="mailto:pragyan.ai.school@gmail.com">pragyan.ai.school@gmail.com</a> | +91-9741007422
    </p>
</div>""", unsafe_allow_html=True)
