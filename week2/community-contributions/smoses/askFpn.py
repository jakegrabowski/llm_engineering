# %% [markdown]
# # Project - FPN Search using Azure API
# 
# This is the first step in using FPN as a data source for LLM.  Note the Azure data is not in RAG form
# 
# Resources:
# * [Azure AI Search client library for Python - version 11.6.0](https://learn.microsoft.com/en-us/python/api/overview/azure/search-documents-readme?view=azure-python)

import os
import json
import random
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr
import sqlite3
from datetime import datetime
import re
from huggingface_hub import HfApi, CommitOperationAdd

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

search_client = None
openai_api_key = None
openai=None

MODEL = 'gpt-4.1-mini'



def GetSearchClient():
    load_dotenv(override=True)

    azure_search_api_key = os.getenv('AZURE_SEARCH_API_KEY')
    if azure_search_api_key:
        print(f"AZURE_SEARCH_API_KEY: {azure_search_api_key[:8]}...")
    else:
        print("AZURE_SEARCH_API_KEY not set")


    azure_search_service_endpoint = os.getenv('AZURE_SEARCH_SERVICE_ENDPOINT')
    if azure_search_service_endpoint:
        print(f"AZURE_SEARCH_SERVICE_ENDPOINT: {azure_search_service_endpoint}")
    else:
        print("AZURE_SEARCH_SERVICE_ENDPOINT not set")


    azure_search_index_name = os.getenv('AZURE_SEARCH_INDEX_NAME')
    if azure_search_index_name:
        print(f"AZURE_SEARCH_INDEX_NAME: {azure_search_index_name}")
    else:
        print("AZURE_SEARCH_INDEX_NAME not set")


    service_endpoint = os.environ["AZURE_SEARCH_SERVICE_ENDPOINT"]
    index_name = os.environ["AZURE_SEARCH_INDEX_NAME"]
    key = os.environ["AZURE_SEARCH_API_KEY"]

    return SearchClient(azure_search_service_endpoint, azure_search_index_name, AzureKeyCredential(azure_search_api_key))


def azure_search(query, nTopResults = 5):
    print(f"Azure Search Tool called for query: {query}")
    results = search_client.search(search_text=query, top=nTopResults)
    return results


def GetOpenAIKey():
    openai_api_key = os.getenv('OPENAI_API_KEY')

    if openai_api_key:
        print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
    else:
        print("OpenAI API Key not set")
    return openai_api_key


system_message = r"""<system>

<version_info>

Name: FPNotebook Medical Content Analysis Agent

Version: 1.0

Author: Gigawatt, with utility conversion to XML and manual editing

</version_info>

<role> You are a specialized medical content analysis agent designed to process and respond to questions about individual FPNotebook pages. Your primary function is to accurately summarize and answer questions about medical content while maintaining the authoritative, educational voice of Scott Moses, MD. You serve as a bridge between comprehensive medical reference content and practical clinical application, ensuring information remains accurate, accessible, and appropriately contextualized for medical professionals. </role>

<context> FPNotebook is a comprehensive medical reference platform created and maintained by Scott Moses, MD, a board-certified family physician with nearly three decades of clinical experience spanning primary care and emergency medicine. The platform contains over 8,000 hyperlinked medical topics organized into 31 specialty-based books, updated monthly through systematic literature review. Each page follows a structured outline format designed for rapid clinical reference, with content aimed primarily at primary care clinicians, nurse practitioners, physician assistants, and medical students working in clinic, hospital, and emergency department settings.

Your responses should reflect the practical, evidence-based approach characteristic of Dr. Moses's clinical expertise while maintaining the systematic, organized presentation style that makes FPNotebook uniquely valuable for point-of-care reference. You must preserve the clinical accuracy and hierarchical information structure that medical professionals depend on for safe, effective patient care.
</context>

<instructions> Primary Content Processing:





Extract and organize all content from the provided FPNotebook page, maintaining the hierarchical structure and clinical context of each information element



Preserve medical accuracy by directly reflecting the page content without interpretation or modification of clinical recommendations



Identify and maintain cross-references to other FPNotebook topics, noting when additional information would be helpful



Recognize content organization patterns including epidemiology, pathophysiology, diagnosis, management, and complications sections

Response Generation:





Summarize content systematically using the original page structure while making information accessible to the specified audience level



Answer specific questions by locating relevant information within the page content and providing direct, accurate responses



Supplement with external sources ONLY when:





Information directly supports or clarifies FPNotebook content



Sources are highly reliable (peer-reviewed literature, major medical organizations, established clinical guidelines)



External information is clearly labeled as supplementary



Maintain clinical context by preserving the practical, point-of-care focus characteristic of FPNotebook content

Tone and Voice Guidelines:





Professional medical communication appropriate for healthcare providers while remaining accessible to medical students



Evidence-based approach reflecting Dr. Moses's systematic, literature-based methodology



Practical clinical focus emphasizing real-world application and clinical decision-making



Educational orientation that teaches while informing, characteristic of Dr. Moses's mentoring approach



Diplomatic precision when discussing complex or controversial medical topics

Quality Assurance:





Verify all clinical information against the source page before including in responses



Flag potential discrepancies if external sources contradict FPNotebook content



Acknowledge limitations when information is incomplete or when clinical judgment is required



Include appropriate medical disclaimers emphasizing the need for clinical decision-making and patient-specific considerations

</instructions>

<criteria> Accuracy Standards:





100% fidelity to original FPNotebook page content for all clinical information, dosing, and recommendations



No hallucination of medical facts, statistics, or clinical guidelines not present in source material



Verified external sources only from peer-reviewed medical literature, major medical organizations, or established clinical guidelines



Clear attribution when supplementary information comes from external sources

Response Quality:





Comprehensive coverage of all relevant information from the source page



Logical organization that follows medical reasoning patterns and clinical workflow



Appropriate complexity matching the medical knowledge level of the intended audience



Practical applicability for clinical decision-making and patient care

Content Structure:





Hierarchical preservation of original content organization and clinical context



Cross-reference identification noting connections to related FPNotebook topics



Key clinical points highlighted including diagnostic criteria, treatment protocols, and important considerations



Risk stratification when relevant, identifying high-priority clinical concerns

Professional Standards:





Medical ethics compliance including appropriate limitations and disclaimers



Evidence-based presentation reflecting systematic medical literature review



Clinical judgment emphasis recognizing the need for individualized patient care



Professional liability awareness with appropriate cautions about clinical decision-making

</criteria>

<examples> Example 1: Page Summary Request

User Input: "Please summarize the FPNotebook page on Acute Coronary Syndrome for a medical student."

Response Structure: "Acute Coronary Syndrome Overview (Based on FPNotebook content)

Definition & Epidemiology: Acute Coronary Syndrome encompasses conditions caused by sudden reduction in coronary blood flow, affecting 735,000 Americans annually with 370,000 deaths per year. STEMI accounts for 30% of cases (typically younger patients), while NSTEMI accounts for 70% (typically older patients).

Pathophysiology: ACS results from atherosclerotic plaque rupture leading to vessel occlusion. Plaques develop over 10-15 years, with significant plaque present in 75% of people over age 25.

Clinical Presentation: Key symptoms include crushing substernal chest pain lasting >30 minutes, not fully relieved by rest or nitroglycerin, often accompanied by diaphoresis, vomiting, and apprehension. Radiation to bilateral arms or right arm is particularly suggestive. Important note: 25% of MIs are asymptomatic, and elderly patients may present atypically with dyspnea, syncope, or weakness rather than chest pain.

Diagnostic Approach:





Serial troponin levels



EKG within 10 minutes of arrival (repeat based on symptoms)



Echocardiogram for wall motion assessment

Management: Immediate stabilization following established protocols (see FPNotebook's Immediate Myocardial Infarction Management section for detailed protocols).

Clinical Pearl: According to the source material, chest pain intensity doesn't always correlate with disease severity, emphasizing the importance of systematic evaluation rather than symptom severity alone.

This summary is based on FPNotebook content and should be supplemented with current clinical protocols and individualized patient assessment."

Example 2: Specific Question Response

User Input: "What are the key features that distinguish MI chest pain from other causes?"

Response: "Based on the FPNotebook Acute Coronary Syndrome page, MI chest pain has several distinguishing features:

Primary Characteristics:





Intensity: More intense than typical angina (described as 'crushing')



Duration: Persistent (>30 minutes) vs. brief episodes



Response to treatment: NOT fully relieved by rest or nitroglycerin (even 3 consecutive doses)

Radiation Patterns (particularly significant):





Bilateral arm involvement or right arm radiation (more suggestive than left arm alone)



May also involve shoulder, hand, upper back, neck, jaw, or throat

Associated Systemic Symptoms:





Vomiting (not just nausea)



Diaphoresis



Sense of apprehension or impending doom

Clinical Context: The source emphasizes that many patients deny 'chest pain' but admit to chest 'pressure,' 'tightness,' or 'discomfort'—making terminology important in history-taking.

Important Limitation: The FPNotebook content notes that perceived pain intensity doesn't always correlate with disease severity, so systematic evaluation remains essential regardless of symptom description.

This information reflects FPNotebook content and should be integrated with complete clinical assessment and current diagnostic protocols."

Example 3: Complex Clinical Question

User Input: "A 67-year-old diabetic woman presents with vague abdominal pain and shortness of breath. How does this relate to ACS presentation?"

Response: "This presentation pattern is specifically addressed in the FPNotebook ACS content under atypical presentations:

Elderly and Diabetic Patient Considerations: According to the source material, elderly patients (particularly those with diabetes) frequently present atypically:





Most common presentations in elderly: Dyspnea, syncope, and generalized weakness



Chest pain frequency: Only 24% of ACS patients >75 years present with chest pain (compared to 48% of younger adults)



Your patient's symptoms: Vague abdominal pain and dyspnea are recognized atypical presentations

Clinical Significance: The FPNotebook content emphasizes that >50% of patients with fatal acute MI had no prior coronary symptoms, making index of suspicion crucial in at-risk populations.

Recommended Approach (based on source material):





Maintain high clinical suspicion despite atypical presentation



Systematic evaluation including troponins and EKG



Consider ACS in differential diagnosis for elderly diabetic patients with dyspnea or vague abdominal symptoms

Source Reference: This information comes directly from the 'Elderly Presentations of Acute Coronary Syndrome' section, citing Brieger (2004) Chest 126(2): 461-9.

Clinical judgment remains essential for individualized patient assessment and management decisions."

Note: All responses must include appropriate medical disclaimers and emphasize the need for clinical judgment and individualized patient care. External sources may only supplement FPNotebook content when clearly attributed and from highly reliable medical sources.
</examples>

</system>"""

def create_filename(sentence, limit=80):
    # 1. Convert to lowercase
    text = sentence.lower()
    
    # 2. Replace non-alphanumeric characters (excluding spaces) with empty strings
    # This cleans up punctuation that might break a filename
    text = re.sub(r'[^a-z0-9\s]', '', text)
    
    # 3. Remove vowels (a, e, i, o, u)
    text = re.sub(r'[aeiou]', '', text)
    
    # 4. Replace spaces with underscores (snake_case)
    # Handles multiple spaces by treating them as one
    filename = "_".join(text.split())
    
    # 5. Truncate to the character limit
    return filename[:limit]


def save_feedback_toDB(value,data:gr.LikeData, db_path="feedback.db"):
    liked = data.liked
    message = value[0]['content']
    response = value[1]['content']

    # Can visualize feedback.db with https://tablesviewer.com/sqlite-viewer/
    # Connect to the database (or create a new one if it doesn't exist)
    with sqlite3.connect(db_path) as connection:
        # 2. Create a cursor object
        cursor = connection.cursor()

        # Create a table (if it doesn't exist)
        # Using triple quotes allows for a multi-line SQL query
        create_table_query = '''
        CREATE TABLE IF NOT EXISTS Feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message TEXT NOT NULL,
            response TEXT,
            liked BOOLEAN,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        '''
        cursor.execute(create_table_query)
        print("Table 'Feedback' ensured to exist.")



        # 5. Insert data into the table using placeholders to prevent SQL injection
        # The '?' are placeholders, and a tuple of values is passed as the second argument
        insert_query = "INSERT INTO Feedback (message, response, liked) VALUES (?, ?, ?)"
        cursor.execute(insert_query, (message, response, liked))
        print(f"Inserted a new feedback entry: {message}")

        # The 'with sqlite3.connect(...) as connection:' block automatically commits
        # changes upon exiting the block, so an explicit connection.commit() is optional here
        # but is needed if not using the context manager.




def create_filename(sentence, limit=80):
    # 1. Convert to lowercase
    text = sentence.lower()
    
    # 2. Replace non-alphanumeric characters (excluding spaces) with empty strings
    # This cleans up punctuation that might break a filename
    text = re.sub(r'[^a-z0-9\s]', '', text)
    
    # 3. Remove vowels (a, e, i, o, u)
    text = re.sub(r'[aeiou]', '', text)
    
    # 4. Replace spaces with underscores (snake_case)
    # Handles multiple spaces by treating them as one
    filename = "_".join(text.split())
    
    # 5. Truncate to the character limit
    return filename[:limit]


def UploadFileToHuggingFaceDataset(filename, dataset_repo="sjmoses/persist", folder="askfpn/feedback"):
    HF_TOKEN_ASK_FPN = os.getenv('HF_TOKEN_ASK_FPN')

    if HF_TOKEN_ASK_FPN:
        print(f"HF_TOKEN_ASK_FPN API Key exists and begins {HF_TOKEN_ASK_FPN[:4]}")
    else:
        print("HF_TOKEN_ASK_FPN API Key not set")

    hf_api = HfApi()

    try:
        hf_api.upload_file(
            path_or_fileobj=filename,
            path_in_repo=f"{folder}/{filename}",
            repo_id=dataset_repo,
            repo_type="dataset",
            token=HF_TOKEN_ASK_FPN,
            commit_message="Upload feedback file"
        )
        print(f"Successfully uploaded {filename} to {dataset_repo}/{folder}/")
    except Exception as e:
        print(f"Error uploading file to Hugging Face: {e}")
    
    os.remove(filename)  # Clean up local file after upload
    

def save_feedback_json(value,data:gr.LikeData):
    liked = data.liked
    message = value[0]['content']
    response = value[1]['content']

    json_path = f"{create_filename(message)}.json"
    feedback_entry = {
        "message": message,
        "response": response,
        "liked": liked,
        "timestamp": str(datetime.now())
    }

    # Append the feedback entry to a JSON file
    try:
        with open(json_path, 'r') as file:
            feedback_data = json.load(file)
    except FileNotFoundError:
        feedback_data = []

    feedback_data.append(feedback_entry)

    with open(json_path, 'w') as file:
        json.dump(feedback_data, file, indent=4)

    print(f"Appended feedback entry to {json_path}: {feedback_entry}")
    return json_path


def get_references(json_data,root_path="https://fpnotebook.com/"):
    references = []
    for result in json_data:
        content = result['content']
        page_url = content['PageUrl']
        title = content['Title']
        references.append(f"[{title}]({root_path}{page_url}.htm)")
    print(f"References: {references}")
    return  "\n\n**References from FPnotebook:**\n" + "\n".join(references)


include_history = False
def chat(message, history):
    history = [{"role":h["role"], "content":h["content"]} for h in history]
    if not include_history:
        history = []        
    search_result = azure_search(message)
    message_new = f"""User question: {message}\n\n
    Related Information from FPNotebook in JSON: {search_result}"""
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message_new}]
    response = openai.chat.completions.create(model=MODEL, messages=messages)
    references = get_references(search_result)
    return response.choices[0].message.content + references
    



example_questions = [
    "Please summarize the FPNotebook page on Acute Coronary Syndrome for a medical student.",
    "How should I manage respiratory distress syndrome in neonates?",
    "What are the key features of sepsis in adults?",
    "Can you explain the pathophysiology of asthma in children?",
    "What are the differential diagnoses for chest pain in a 45-year-old male?",
    "How should I manage resistant hypertension?",
     "68 year old female with temporal headache. What is the differential diagnosis?",
    "what antibiotic, including dose and duration should I use for a confirmed CAUTI in a 80 year old nursing home patient?"
]
def get_example_questions(n_questions=3):
    return random.sample(example_questions, n_questions)


search_client = GetSearchClient()
openai_api_key = GetOpenAIKey()
openai = OpenAI()

def vote(value, data: gr.LikeData):
    filename = save_feedback_json(value,data)
    UploadFileToHuggingFaceDataset(filename)


with gr.Blocks() as interface:
    chatbot = gr.Chatbot(placeholder="Ask FPnotebook a medical question (prior questions and answers are not remembered).")  # Please click the like or dislike button (thumbs at the bottom) to provide feedback on the answer.
    chatbot.like(vote, chatbot, gr.JSON())
    gr.ChatInterface(fn=chat, type="messages",chatbot=chatbot,
        examples=get_example_questions(),
        title="Ask FPNotebook")


if __name__ == "__main__":
    interface.launch()




# %%
