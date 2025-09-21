import streamlit as st
import base64
import os
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq


# --- Page Configuration ---
st.set_page_config(
    page_title="MCP in AI & Data Science | TJIT + PragyanAI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Styling ---
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")

# --- Helper function for PDF download ---
def get_pdf_as_base64(file_path):
    """Reads a PDF file and returns its base64 encoded string."""
    try:
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return None

# --- RAG Pipeline Setup (Cached for performance) ---
@st.cache_resource
def setup_rag_pipeline(groq_api_key):
    """Loads PDF, splits it, creates embeddings and a vector store, and sets up a RAG QA chain."""
    # 1. Load PDF
    pdf_path = "MCP Program - Grooms Tommorrow's AI Leaders - TJIT -2 Years.pdf"
    if not os.path.exists(pdf_path):
        st.error("Brochure PDF ('MCP Program - Grooms Tommorrow's AI Leaders - TJIT -2 Years.pdf') not found for RAG setup.")
        return None
    
    with st.spinner("Processing program brochure for AI Assistant..."):
        loader = PyPDFLoader(file_path=pdf_path)
        docs = loader.load()

        # 2. Split Text into Chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        # 3. Create Embeddings using a local model
        # Using a local model avoids API calls for embedding and is free
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # 4. Create Vector Store (FAISS)
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)

        # 5. Setup LLM & Retrieval QA Chain
        llm = ChatGroq(api_key=groq_api_key, model_name="llama3-8b-8192")
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )
    return qa_chain


# --- Main App Sections ---

def render_sidebar():
    """Renders the sidebar navigation and key action links."""
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-header">
                <span class="brand-text">TJIT + <span class="gradient-text">PragyanAI</span></span>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.header("Navigation")
        
        # Navigation using a radio button for a cleaner look
        page = st.radio(
            "Go to",
            [
                "Home", "Program Details", "Curriculum", "Investment", 
                "Our Advantage", "About PragyanAI", "AI Assistant", "FAQ"
            ],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        st.header("Get Started")
        
        # Expression of Interest Form
        st.link_button(
            "Expression of Interest Form", 
            "https://forms.gle/your-google-form-link", # Replace with your actual Google Form link
            use_container_width=True, type="primary"
        )
        
        # PDF Download
        pdf_b64 = get_pdf_as_base64("MCP Program - Grooms Tommorrow's AI Leaders - TJIT -2 Years.pdf")
        if pdf_b64:
            st.download_button(
                label="Download Program Brochure (PDF)",
                data=base64.b64decode(pdf_b64),
                file_name="PragyanAI_MCP_Brochure.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        else:
            st.warning("Brochure PDF not found. Please add it to the project folder.")
            
        st.markdown("---")
        st.info(
            """
            **Contact Us:**
            📧 [pragyan.ai.school@gmail.com](mailto:pragyan.ai.school@gmail.com)  
            📞 [+91-9741007422](tel:+919741007422)
            """
        )
        return page

def render_hero():
    """Renders the main hero section."""
    st.markdown(
        """
        <div class="hero-section">
            <h1 class="hero-title">Groom Tomorrow's AI Leaders</h1>
            <p class="hero-subtitle">A 2-Year Master's Certificate Program in Data Science, AI, GenAI & AgenticAI by TJIT and PragyanAI.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    cols = st.columns(3)
    with cols[0]:
        st.markdown('<div class="metric-card">🕒 1200+ Hours</div>', unsafe_allow_html=True)
    with cols[1]:
        st.markdown('<div class="metric-card">🏆 120+ Projects</div>', unsafe_allow_html=True)
    with cols[2]:
        st.markdown('<div class="metric-card">✅ Placement Assured</div>', unsafe_allow_html=True)

def render_program_details():
    """Renders About Program and Roadmap sections."""
    st.markdown("## Transform Your Career with <span class='gradient-text'>Deep AI Skills</span>", unsafe_allow_html=True)
    st.write("This isn't just another certificate. It's a transformational journey designed to counter market headwinds and place you at the forefront of the AI revolution.")
    
    cols = st.columns(3)
    with cols[0]:
        with st.container(border=True):
            st.subheader("💡 Hybrid Learning Model")
            st.write("Combines 400+ hours of intensive offline sessions at TJIT with 600+ hours of live online instruction, offering the best of both worlds.")
    with cols[1]:
        with st.container(border=True):
            st.subheader("📈 Impact Assured")
            st.write("We guarantee you'll earn at least ₹50K-1 Lakh more than your peers, or we refund 50% of the program fee. Your success is our mission.")
    with cols[2]:
        with st.container(border=True):
            st.subheader("🎓 M.Tech Equivalent")
            st.write("With 1200+ hours, 120+ projects, and a curriculum covering 8 key AI domains, the depth is comparable to an M.Tech in AI/ML from a top institution.")
    
    st.markdown("---")
    
    st.markdown("## Your <span class='gradient-text'>AI Skills Acceleration</span> Roadmap", unsafe_allow_html=True)
    st.write("A structured, semester-wise journey from foundational concepts to market-leading AI technologies.")
    
    # Roadmap Timeline
    timeline = {
        "SEMESTER 5": ("Foundations & Data Wrangling", "Master Python Full Stack, Data Science fundamentals, advanced data wrangling, statistics, and visualization."),
        "SEMESTER 6": ("BI & Machine Learning", "Explore Business Intelligence with tools like Tableau & PowerBI. Build, evaluate, and deploy various Machine Learning models."),
        "SEMESTER 7": ("Deep Learning, CV & NLP", "Uncover the power of Deep Learning with TensorFlow & PyTorch. Work on Computer Vision and build advanced NLP-powered AI Bots."),
        "SEMESTER 8": ("Generative & Agentic AI", "Step into the future with Generative AI, LLMs, and Prompt Engineering. Build autonomous AI agents and multi-agent systems.")
    }
    
    for semester, (title, description) in timeline.items():
        st.markdown(f"#### {semester}: {title}")
        st.write(description)
        st.markdown("<br>", unsafe_allow_html=True)


def render_curriculum():
    """Renders the curriculum section with tabs."""
    st.markdown("## An <span class='gradient-text'>Industry-Vetted</span> Curriculum", unsafe_allow_html=True)
    st.write("Explore the 8 key skills you'll master, packed with the most in-demand tools and libraries.")

    sem5, sem6, sem7, sem8 = st.tabs(["Semester 5", "Semester 6", "Semester 7", "Semester 8"])

    with sem5:
        st.subheader("SKILL 1: Python & Data Wrangling")
        st.write("Core Python, data processing, web scraping, and media handling.")
        st.caption("Tools: NumPy, Pandas, OpenCV, Pillow, Scrapy")
        st.subheader("SKILL 2: Data Science Foundation")
        st.write("Statistics, probability, EDA, and data visualization.")
        st.caption("Tools: Matplotlib, Seaborn, Plotly, Statsmodels")

    with sem6:
        st.subheader("SKILL 3: BI / Data Analytics")
        st.write("Business Intelligence, advanced Excel, and dashboarding.")
        st.caption("Tools: Tableau, PowerBI, Google Looker, Metabase")
        st.subheader("SKILL 4: Machine Learning")
        st.write("Supervised/unsupervised learning, model optimization and deployment.")
        st.caption("Tools: Scikit-learn, XGBoost, LightGBM, Streamlit")

    with sem7:
        st.subheader("SKILL 5: Deep Learning & CV")
        st.write("ANN, CNN, RNNs, transformers, and advanced Computer Vision.")
        st.caption("Tools: TensorFlow, Keras, PyTorch, MediaPipe, OpenCV")
        st.subheader("SKILL 6: NLP & AI Bot")
        st.write("Text processing, embeddings, conversational AI, and multimodal bots.")
        st.caption("Tools: Hugging Face, Transformers, NLTK, spaCy")

    with sem8:
        st.subheader("SKILL 7: Generative AI")
        st.write("LLMs, fine-tuning, RAG, and advanced prompt engineering.")
        st.caption("Tools: GPT, LLAMA, LangChain, LlamaIndex, PEFT")
        st.subheader("SKILL 8: Agentic AI")
        st.write("Building single and multi-agent systems for autonomous tasks.")
        st.caption("Tools: CrewAI, AutoGPT, LangChain Agents, AutoGen")

def render_investment():
    """Renders the investment and comparison sections."""
    st.markdown("## A Risk-Free <span class='gradient-text'>Investment In Your Future</span>", unsafe_allow_html=True)
    st.write("Our unique financial model ensures our goals are aligned with yours: your success.")

    cols = st.columns(2)
    with cols[0]:
        with st.container(border=True):
            st.subheader("Program Fee")
            st.markdown("<p class='price-text'>₹90,000</p>", unsafe_allow_html=True)
            st.write("Paid per semester for 4 or 6 semesters.")
    with cols[1]:
        with st.container(border=True):
            st.subheader("Pay After Placement")
            st.write("Only pay after you secure a job. The amount depends on your CTC:")
            st.markdown("""
            - **₹25,000** for CTC < 5 Lakh
            - **₹50,000** for CTC 5-10 Lakh
            - **₹1 Lakh** for CTC > 10 Lakh
            """)
    
    st.markdown("---")

    st.markdown("## The <span class='gradient-text'>PragyanAI Difference</span>", unsafe_allow_html=True)
    st.write("See how our MCP program stacks up against other popular options.")
    
    st.markdown("""
    | Feature               | MCP (This Program)                     | Other AI/ML Programs    | Bootcamps                   |
    |-----------------------|----------------------------------------|-------------------------|-----------------------------|
    | **Duration & Hours** | **1200+ hours** | 250-600 hours           | 200-400 hours               |
    | **Projects** | **100+ projects, 2-5 real capstones** | ~10 projects            | Few basic projects          |
    | **Delivery Mode** | **Blended (Offline + Online)** | Primarily Online        | Mostly Online               |
    | **Cost Structure** | **Pay-per-sem + Pay-after-placement** | High upfront fee        | High upfront fee            |
    | **Placement** | **100% Guarantee with refund policy** | No formal guarantee     | "Placement assistance" only |
    | **Skill Coverage** | **Full stack AI, GenAI, AgenticAI** | Core ML/DS              | ML basics, coding prep      |
    """)


def render_advantage():
    """Renders the 'Why Us' and 'Trainer' sections."""
    st.markdown("## Your <span class='gradient-text'>Unfair Advantage</span> in the AI Era", unsafe_allow_html=True)
    st.write("We've engineered this program to give you a decisive edge in a competitive job market.")
    
    cols = st.columns(4)
    cols[0].metric("📚 Intense Rigor", "1500+ Hours", "8 AI Domains")
    cols[1].metric("💰 Outcome-Based", "Refund Guarantee", "Higher CTC")
    cols[2].metric("🔬 Research Alignment", "3 Papers/Patents", "Hackathons")
    cols[3].metric("🚀 Unmatched Depth", "100+ Projects", "60+ Tools")

    st.markdown("---")
    
    st.markdown("## Meet Your <span class='gradient-text'>Mentor</span>", unsafe_allow_html=True)
    
    cols = st.columns([1, 2])
    with cols[0]:
        st.image("https://placehold.co/200x200/1e293b/f97316?text=SA", caption="Sateesh Ambessange, Lead Trainer", use_column_width=True)
    with cols[1]:
        st.write("""
        Learn from a veteran with **25+ years of industry, academic, and research experience**. 
        He has developed commercial AI products, guided GenAI startups, mentored at hackathons, 
        and helped shape AI curricula for top institutions like Wharton University.
        """)
        st.link_button("View LinkedIn Profile", "https://www.linkedin.com/in/sateesh-ambesange-3020185/")
        
def render_about_pragyanai():
    """Renders the section about PragyanAI's achievements."""
    st.markdown("## Powered by an <span class='gradient-text'>Award-Winning AI Pioneer</span>", unsafe_allow_html=True)
    st.write("PragyanAI is not just an educator but a DIPP-certified AI product and consulting company at the heart of India's innovation ecosystem.")

    cols = st.columns(3)
    with cols[0]:
        with st.container(border=True):
            st.subheader("🏆 National Recognition")
            st.write("""
            - Karnataka Startup Elevate Winner
            - Recipient of Central Govt. Grant
            - Part of Startup India & Karnataka
            """)
    with cols[1]:
        with st.container(border=True):
            st.subheader("🌐 Ecosystem Partners")
            st.write("Proudly associated with leading institutions:")
            st.caption("NASSCOM | IIMB-NSRCEL | GSF Academy | IIITB | DERBI")
    with cols[2]:
        with st.container(border=True):
            st.subheader("🎓 Community & Academia")
            st.write("""
            - Signed MOU with VTU for Internships
            - Conducted 200+ Workshops
            - Jury at Startup India Hackathons
            """)

def render_ai_assistant():
    """Renders the RAG-based AI Assistant."""
    st.markdown("## AI <span class='gradient-text'>Student Assistant (RAG Enabled)</span>", unsafe_allow_html=True)
    st.write("Ask questions about the program. Our AI assistant will find answers directly from the official brochure using a RAG model.")

    # Check for API Key
    try:
        groq_api_key = st.secrets["GROQ_API_KEY"]
    except KeyError:
        st.error("GROQ_API_KEY not found in Streamlit secrets. Please add it to run the AI assistant.")
        st.code(" # .streamlit/secrets.toml \n GROQ_API_KEY = 'YOUR_API_KEY_HERE' ")
        return

    # Setup the RAG pipeline
    qa_chain = setup_rag_pipeline(groq_api_key)
    if qa_chain is None:
        return

    # Initialize chat history
    if "rag_messages" not in st.session_state:
        st.session_state.rag_messages = [
            {"role": "assistant", "content": "How can I help you with the program details today?"}
        ]

    # Display chat messages
    for message in st.session_state.rag_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if prompt := st.chat_input("e.g., How much is the fee after placement?"):
        st.session_state.rag_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching the brochure..."):
                try:
                    response = qa_chain.invoke({"query": prompt})
                    answer = response.get("result", "Sorry, I couldn't find an answer in the document.")
                    st.markdown(answer)
                    st.session_state.rag_messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"An error occurred: {e}")

def render_faq():
    """Renders the FAQ section."""
    st.markdown("## Frequently Asked <span class='gradient-text'>Questions</span>", unsafe_allow_html=True)

    faq_data = {
        "What is the total duration of the program?": "The program is a 2-year comprehensive Master's Certificate Program with over 1200+ hours of learning, spread across semesters 5 through 8 of your engineering degree.",
        "How does the 'Impact Assured' guarantee work?": "We guarantee that graduates will secure a job with a CTC that is at least ₹50,000 to ₹1 Lakh higher than their college peers. If this goal is not met, we will refund 50% of the program fee.",
        "Is this program suitable for beginners?": "Yes. The curriculum is designed to take you from foundational concepts like Python programming in Semester 5 to advanced topics like Agentic AI in Semester 8. It's a structured journey suitable for dedicated learners.",
        "What kind of placement support is provided?": "We provide extensive career support including mock interviews (technical, HR, GD), profile building (Resume, LinkedIn, GitHub), and complementary training on DSA, core CSE concepts, aptitude, and soft skills. The program culminates in a placement drive."
    }

    for question, answer in faq_data.items():
        with st.expander(question):
            st.write(answer)

# --- Main App Execution ---
def main():
    selected_page = render_sidebar()
    
    if selected_page == "Home":
        render_hero()
        st.markdown("---")
        render_program_details()
    elif selected_page == "Program Details":
        render_program_details()
    elif selected_page == "Curriculum":
        render_curriculum()
    elif selected_page == "Investment":
        render_investment()
    elif selected_page == "Our Advantage":
        render_advantage()
        st.markdown("---")
        # Testimonials are a good fit here
        st.markdown("## What Our <span class='gradient-text'>Students Say</span>", unsafe_allow_html=True)
        cols = st.columns(3)
        with cols[0]:
            st.info("""
            "The depth is incredible. I went from Python basics to building complex Agentic AI systems... gave me the confidence to crack interviews at top tech companies."
            **- Arjun Kumar**
            """)
        with cols[1]:
            st.info("""
            "The pay-after-placement model was a game-changer... It showed that PragyanAI was as invested in my career as I was."
            **- Priya Sharma**
            """)
        with cols[2]:
            st.info("""
            "The mentorship from Sateesh sir is unparalleled. His industry experience provides insights you just can't get from textbooks."
            **- Rahul Verma**
            """)
    elif selected_page == "About PragyanAI":
        render_about_pragyanai()
    elif selected_page == "AI Assistant":
        render_ai_assistant()
    elif selected_page == "FAQ":
        render_faq()

if __name__ == "__main__":
    main()
	
