# causal_ai_gradio.py

import os
import tempfile
import shutil
from pathlib import Path
import operator # Added for Annotated example, though not strictly needed by supervisor here
import json # For potential structured output parsing
import pickle # For FAISS saving/loading

# --- Core Typing Imports ---
from typing import TypedDict, Annotated, Sequence, Optional, List, Tuple, Union, Dict

from dotenv import load_dotenv

# --- LangChain / LangGraph Imports ---
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage # Added
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader # Keep for document processing
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import Tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent # Used in notebook, adapting supervisor instead

# Using the specific supervisor import
# Ensure langgraph-supervisor is installed: pip install langgraph-supervisor
try:
    from langgraph_supervisor import create_supervisor
except ImportError:
    print("ERROR: langgraph-supervisor not found. Please install it: pip install langgraph-supervisor")
    exit()


# --- Gradio Import ---
import gradio as gr

# --- Other Imports ---
import markdown2
import datetime

# --- Configuration ---
load_dotenv()

# --- !! Configure Paths !! ---
# Path to your IEC 62740 PDF/Text document for RAG
IEC_62740_PATH = "data/IEC-62740.pdf" # UPDATE THIS PATH if needed
# Path to save/load the persistent FAISS index
FAISS_INDEX_PATH = "faiss_iec62740_index" # Directory to save/load index
# --- End Configuration ---

# --- Initialize LLM and Embeddings ---
# (Keep the initialization code from the previous response, including try-except)
try:
    llm = AzureChatOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.getenv("AZURE_DEPLOYMENT_NAME"),
        api_version="2024-02-01"
    )

    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=os.getenv("EMBEDDING_ENDPOINT"),
        api_key=os.getenv("EMBEDDING_API_KEY"),
        azure_deployment=os.getenv("EMBEDDING_DEPLOYMENT"),
        # Assuming a suitable default API version or specify one like:
        # api_version="2023-05-15"
    )
except Exception as e:
    print(f"ERROR initializing Azure OpenAI clients: {e}")
    print("Please check your .env file and Azure deployment details.")
    exit()

# --- RAG Setup ---
# (Keep the RAG setup code from the previous response, including try-except)
vectorstore = None
if os.path.exists(FAISS_INDEX_PATH):
    print(f"Loading existing FAISS index from {FAISS_INDEX_PATH}")
    try:
        vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        print("FAISS index loaded successfully.")
    except Exception as e:
        print(f"Error loading FAISS index: {e}. Recreating index might be needed if it fails.")
else:
    if os.path.exists(IEC_62740_PATH):
        print(f"Creating new FAISS index from {IEC_62740_PATH}")
        try:
            loader = PyPDFLoader(IEC_62740_PATH)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            texts = text_splitter.split_documents(documents)
            if not texts:
                 raise ValueError("No text chunks generated from the document.")
            vectorstore = FAISS.from_documents(texts, embeddings)
            vectorstore.save_local(FAISS_INDEX_PATH)
            print("FAISS index created and saved.")
        except Exception as e:
             print(f"ERROR creating FAISS index: {e}")
             # Decide how to handle - exit or proceed without RAG?
             # exit() # Exit if RAG is critical
    else:
        print(f"ERROR: IEC 62740 document not found at {IEC_62740_PATH} and index doesn't exist.")
        # exit() # Exit if RAG is critical


# --- Tool Definitions ---
# (Keep the tool definitions: retrieve_rca_techniques, rca_knowledge_tool,
#  process_investigation_report, investigation_tool from the previous response)

# RAG Tool (Advisor)
def retrieve_rca_techniques(query: str) -> str:
    """Search for relevant RCA techniques in the IEC 62740 knowledge base."""
    if vectorstore is None:
        return "ERROR: FAISS vector store not available. Cannot perform RAG."
    try:
        docs = vectorstore.similarity_search(query, k=3)
        if not docs:
            return "No relevant techniques found in the knowledge base for the query."
        return "\n\n".join(doc.page_content for doc in docs)
    except Exception as e:
        return f"Error during RAG retrieval: {e}"

rca_knowledge_tool = Tool(
    name="rca_techniques_lookup",
    description="Search the IEC 62740 standard for Root Cause Analysis (RCA) techniques, methodologies, and their applications based on an incident description or query.",
    func=retrieve_rca_techniques
)

# Investigation Report Tool (Performer)
def process_investigation_report(file_path: str) -> str:
    """Process investigation report (PDF) and extract relevant text content."""
    print(f"Processing investigation report: {file_path}")
    if not file_path or file_path.lower() == 'none' or not os.path.exists(file_path):
        return "Report file path is invalid, not provided, or file does not exist."
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        if not pages:
            return "Could not load any pages from the PDF report."
        # Limit context size if needed
        max_chars = 15000
        content = "\n\n".join(page.page_content for page in pages)
        if len(content) > max_chars:
             content = content[:max_chars] + "\n\n... [Report Content Truncated] ..."
        return content
    except Exception as e:
        print(f"Error processing report PDF at {file_path}: {e}")
        return f"Error processing report: {str(e)}"

investigation_tool = Tool(
    name="investigation_report_lookup",
    description="Reads and extracts text content from a provided investigation report PDF file path. Use this to gather evidence and context for the RCA.",
    func=process_investigation_report
)

# --- Agent Definitions using create_react_agent ---

# Define the RCA Technique Advisor agent prompt (System Message Content)
technique_advisor_system_prompt = """You are an expert in Root Cause Analysis (RCA) techniques based on IEC 62740.
Given an incident description:
1. Use the rca_techniques_lookup tool to search for relevant techniques mentioned in IEC 62740 based on keywords from the incident description.
2. Consider the incident context and apparent complexity.
3. Suggest the single most appropriate RCA technique (e.g., 5 Whys, ECF Charting, Fishbone, CTM, FTA, Tripod Beta, CAST).
4. Provide a concise justification for your choice, referencing the strengths of the technique for this type of incident based on IEC 62740 principles.
5. Format your response clearly:
RECOMMENDED TECHNIQUE: [Technique Name]
JUSTIFICATION: [Your concise justification]
"""

technique_advisor = create_react_agent(
    model=llm,
    tools=[rca_knowledge_tool],
    name="rca_technique_advisor",
    prompt=technique_advisor_system_prompt
)

# Define the RCA Performer agent prompt (Passed Dynamically via Supervisor)
# Base system message for the performer agent
rca_performer_system_prompt = "You are an RCA specialist executing an analysis."

rca_performer = create_react_agent(
    model=llm,
    tools=[investigation_tool],
    name="rca_performer",
    prompt=rca_performer_system_prompt
)


# --- Supervisor Definition ---
members = [technique_advisor, rca_performer]
system_prompt = (
    "You are a supervisor managing an RCA process with two agents: 'rca_technique_advisor' and 'rca_performer'. "
    "Your goal is to orchestrate the process to deliver a final RCA report.\n"
    "1. **Receive Initial Request:** The user input contains the incident description and potentially a report path.\n"
    "2. **Delegate to Advisor:** First, you MUST delegate the task to 'rca_technique_advisor'. Provide the advisor ONLY with the incident description extracted from the user's input. Ask it to recommend the best RCA technique based on IEC 62740 and justify it.\n"
    "3. **Process Advisor Response:** Receive the recommended technique and justification from the advisor.\n"
    "4. **Delegate to Performer:** Next, you MUST delegate the task to 'rca_performer'. Construct a detailed prompt for the performer containing: \n"
    "    - The original incident description.\n"
    "    - The technique recommended by the advisor.\n"
    "    - The EXACT file path of the investigation report if one was provided (pass the path string, e.g., 'C:/path/report.pdf' or 'None').\n"
    "    - Clear instructions to use the 'investigation_report_lookup' tool with the path if provided, perform the analysis using the recommended technique, and structure the output in Markdown (mentioning sections like Overview, Evidence, Analysis, Root Causes, Recommendations).\n"
    "5. **Receive Performer Analysis:** Get the structured analysis from the performer.\n"
    "6. **Final Report:** Format the final response to the user. Combine the advisor's recommendation/justification and the performer's full analysis into a single, coherent Markdown report. Start with the Advisor's part, then the Performer's detailed analysis.\n"
    "**Constraint:** Strictly follow the sequence: Advisor -> Performer -> Final Report. Do not skip steps. Ensure the Performer receives the correct technique name and report path instruction."
)

# Create the supervisor workflow
# Pass agent runnables directly
supervisor_app = create_supervisor(
    model=llm, # Supervisor uses an LLM for routing/prompting decisions
    prompt=system_prompt,
    agents=members
)

# --- Gradio Functionality ---

# (Keep the format_final_response_for_gradio function from the previous response)
def format_final_response_for_gradio(final_markdown_content: str) -> str:
    """Formats the final Markdown content into HTML for Gradio display."""
    if not isinstance(final_markdown_content, str):
        final_markdown_content = "Error: Invalid final content received."
    try:
        markdown_html = markdown2.markdown(
            final_markdown_content,
            extras=['tables', 'fenced-code-blocks', 'code-friendly']
        )
    except Exception as e:
        print(f"Error converting final markdown to HTML: {e}")
        markdown_html = f"<p>Error formatting output. Raw content:</p><pre>{final_markdown_content}</pre>"
    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"rca_analysis_{timestamp}.md"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"# RCA Analysis Results ({timestamp})\n\n")
            f.write(final_markdown_content)
        print(f"Analysis saved to: {filename}")
    except Exception as e:
        print(f"Could not save analysis to file: {e}")

    html_output = f"""
    <div style="font-family: sans-serif; line-height: 1.6; padding: 15px; border: 1px solid #eee; border-radius: 8px; background-color: #f9f9f9;">
        <h2 style="color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 5px;">RCA Analysis Results</h2>
        <div class="markdown-content">
            {markdown_html}
        </div>
    </div>
    <style>
        .markdown-content h1, .markdown-content h2, .markdown-content h3 {{ color: #2980b9; margin-top: 1.2em; margin-bottom: 0.6em; }}
        .markdown-content h1 {{ font-size: 1.6em; }}
        .markdown-content h2 {{ font-size: 1.4em; }}
        .markdown-content h3 {{ font-size: 1.2em; }}
        .markdown-content strong {{ font-weight: 600; }}
        .markdown-content ul, .markdown-content ol {{ margin-left: 20px; margin-bottom: 1em;}}
        .markdown-content li {{ margin-bottom: 0.4em; }}
        .markdown-content pre {{ background-color: #eee; padding: 10px; border-radius: 4px; overflow-x: auto; font-family: monospace;}}
        .markdown-content code {{ background-color: #eee; padding: 2px 4px; border-radius: 3px; font-family: monospace;}}
        .markdown-content table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        .markdown-content th {{ background-color: #3498db; color: white; padding: 10px; border: 1px solid #ddd; text-align: left;}}
        .markdown-content td {{ padding: 8px; border: 1px solid #ddd; background-color: white; }}
        .markdown-content hr {{ border: none; border-top: 1px solid #ddd; margin: 20px 0; }}
    </style>
    """
    return html_output

# Define the Gradio callback function
def run_analysis_gradio(incident_description: str, report_file_obj: Optional[gr.File]): # Use gr.File type hint
    """
    Runs the LangGraph Supervisor App for RCA analysis via Gradio interface.
    """
    print("\n--- Gradio Analysis Request ---")
    print(f"Incident: {incident_description}")

    report_path_for_agent = "None"
    # Gradio provides a temp file object with a '.name' attribute for the path
    if report_file_obj:
        report_path_for_agent = report_file_obj.name
        print(f"Report uploaded: {report_path_for_agent}")
    else:
        print("No report uploaded.")

    # Prepare the initial message for the supervisor
    # Pass the literal report path string. The performer agent's tool handles it.
    initial_supervisor_input = f"""Start RCA Process.
Incident Description: {incident_description}
Investigation Report Path: {report_path_for_agent}
"""
    messages = [HumanMessage(content=initial_supervisor_input)]

    print("Invoking supervisor app...")
    final_result_content = ""
    try:
        # Invoke the supervisor app
        # Note: Streaming might be complex to parse correctly with create_supervisor
        # Using invoke to get the final state is often easier
        
        final_state = supervisor_app.compile().invoke({"messages": messages})

        # Extract the final message from the supervisor (expected to be the report)
        if final_state and final_state.get("messages"):
            final_result_content = final_state["messages"][-1].content
        else:
            final_result_content = "Error: Supervisor did not return expected messages structure."

    except Exception as e:
        print(f"Error invoking supervisor app: {e}")
        final_result_content = f"An error occurred during analysis: {str(e)}\n\nTraceback: {traceback.format_exc()}" # Add traceback for debugging

    print("\n--- Analysis Complete ---")
    # Format the final result for Gradio HTML display
    return format_final_response_for_gradio(final_result_content)



# --- Gradio Interface Definition ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Causal AI - A Root Cause Analysis (RCA) agent")
    gr.Markdown("Enter the incident description below. Optionally, upload a relevant investigation report (PDF only).")

    with gr.Row():
        incident_input = gr.Textbox(label="Incident Description", placeholder="Describe the incident or problem here...", lines=3, scale=3)
        report_upload = gr.File(label="Upload Investigation Report (Optional PDF)", file_types=['.pdf'], scale=1) # Use gr.File type

    # Make the button interactive=True initially
    analyze_button = gr.Button("Analyze Incident", variant="primary", interactive=True)

    # Use HTML component for output, allows embedding spinners or messages
    output_display = gr.HTML(label="Analysis Result")

    # Define the "Processing" state HTML (can include a simple spinner or just text)
    processing_html = """
    <div style='display:flex; justify-content:center; align-items:center; padding: 20px; color: #555;'>
      <svg width="30" height="30" viewBox="0 0 100 100" preserveAspectRatio="xMidYMid">
        <circle cx="50" cy="50" fill="none" stroke="#3498db" stroke-width="10" r="35" stroke-dasharray="164.93361431346415 56.97787143782138">
          <animateTransform attributeName="transform" type="rotate" repeatCount="indefinite" dur="1s" values="0 50 50;360 50 50" keyTimes="0;1"></animateTransform>
        </circle>
      </svg>
      <span style='margin-left: 10px;'>Processing analysis, please wait...</span>
    </div>
    """

    # Link button click to the analysis function with pre- and post-updates
    analyze_button.click(
        # Function to run BEFORE the main analysis function
        fn=lambda: (
            gr.update(value=processing_html), # Show processing message/spinner
            gr.update(interactive=False),     # Disable button
            # Optionally disable inputs too:
            # gr.update(interactive=False),
            # gr.update(interactive=False)
        ),
        inputs=None, # No inputs for this pre-function
        outputs=[
            output_display,
            analyze_button,
            # incident_input, # if disabling inputs
            # report_upload   # if disabling inputs
        ]
    ).then(
        # Main analysis function
        fn=run_analysis_gradio,
        inputs=[incident_input, report_upload], # Pass the actual inputs here
        outputs=[output_display] # Output goes to the HTML display
    ).then(
         # Function to run AFTER the main analysis function completes
        fn=lambda: (
            gr.update(interactive=True),      # Re-enable button
            # Optionally re-enable inputs:
            # gr.update(interactive=True),
            # gr.update(interactive=True)
        ),
        inputs=None, # No inputs for this post-function
        outputs=[
            analyze_button,
            # incident_input, # if re-enabling inputs
            # report_upload   # if re-enabling inputs
        ]
    )


    gr.Examples(
         examples=[
              ["A critical pump failed unexpectedly during operation, causing a production shutdown.", None],
              ["A dropped object (a grating plate) on the Heidrun TLP, on 22 September 2015.", None],
              # Add example with file path if needed for testing, Gradio will handle UI upload
              # ["A dropped object (a grating plate) on the Heidrun TLP, on 22 September 2015.", "data/investigation_reports/investigation-report---statoil---heidrun.pdf"],
         ],
         inputs=[incident_input, report_upload],
         outputs=output_display,
         # Note: Examples running directly won't show the intermediate processing state easily
         # They still call the main function
         fn=run_analysis_gradio,
         cache_examples=False
    )

if __name__ == "__main__":
    # Add a traceback import for better error reporting
    import traceback
    # Check if vector store loaded ok before launch
    if vectorstore is None:
         print("\nWARNING: FAISS Vector Store for IEC 62740 could not be loaded/created.")
         print("The RCA Technique Advisor's RAG tool will not function correctly.")

    print("\nLaunching Gradio Interface...")
    demo.launch() # share=True if needed