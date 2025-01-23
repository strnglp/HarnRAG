import os
import sys
import logging
import warnings

from llama_index.core import Settings
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from embedding import get_query_engine
from history import query_with_follow_up
################# AI MODEL SETTINGS ###################
# This project can use various embedding models for
# encoding semantic meaning of documents into vectors
# https://huggingface.co/BAAI/bge-base-en-v1.5
# Baseline best performance I've seen
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
# This project uses Ollama for summarization of answers
# https://ollama.com/search
LLM_MODEL = "llama3.2"
#LLM_MODEL = "phi3:medium"
#LLM_MODEL = "gemma2:27b"
# Chunk size wound up being very important to my RAG usecase
# When it is too small llama-index will chunk Documents, which
# I was creating at the page level, into multiple Documents.
# This meant relevancy of each of these smaller Documents may
# be computed without important context (from the rest of the page)
# It's now set at a size that seems to give each page a Document
Settings.chunk_size = 2048
Settings.overlap = 512
Settings.context_window = 16384
Settings.num_output = 512
Settings.embed_model = HuggingFaceEmbedding(
    model_name=EMBEDDING_MODEL,   
    device="cuda", 
    max_length=512,
    trust_remote_code=True
)
Settings.llm = Ollama(
    model=LLM_MODEL,
    temperature=0,
    request_timeout=60.0,
    context_window=Settings.context_window,
    num_output=Settings.num_output,
)
#######################################################
####
#### Logging, output & warning configuration
####
#logging.basicConfig(level=logging.CRITICAL)
warnings.filterwarnings("ignore") # pytorch can be spammy
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1" # amd ROCm support related
###
### Load or create embeddings
###
data_path = "./data/" # source data
storage_path = "./storage_" + EMBEDDING_MODEL.split("/")[1] # computed data
####
#### User input loop for conversation
####
# TODO: swap in a custom prompt when "lists" or quantities (all) are requested
custom_qa_prompt_str = """
You are an expert at retrieving and organizing information from the provided context.
Analyze the context carefully and answer the query comprehensively, using all relevant details.
When relevant "spells" are called "ritual invocations" in the context of clerics/gods.
Do not reference the context or keywords in your response.
Always provide a complete and direct answer without notes or commentary.
Prefer tables or lists when providing collections and respond using markdown.

Context:

[Acronyms start]
FRP=Fantasy Role-Playing
GM=Gamemaster
NPC=Non-Player Character
PC=Player Character
AGL=Agility
AUR=Aura
CML=Comeliness or Convocation Mastery Level (when in the context of spells)
DEX=Dexterity
END=Endurance
EYE=Eyesight
HRG=Hearing
INT=Intelligence
MOR=Morality
MOV=Move
SML=Smell
STA=Stamina
STR=Strength
VOI=Voice
WIL=Will
ML=Mastery Level
EML=Effective Mastery Level
OML=Opening Mastery Level
SB=Skill Base
SI=Skill Index
CS=Critical Success
MS=Marginal Success
MF=Marginal Failure
CF=Critical Failure
IP=Injury Penalty
FP=Fatigue Penalty
EP=Encumbrance Penalty
UP=Universal Penalty
IL=Injury Level
FL=Fatigue Level
B=Blunt Aspect
E=Edge Aspect
P=Point Aspect
F=Fire/Frost Aspect
PP=Piety Point
RML=Ritual Mastery Level
RTL=Ritual Target Level in the context of Gods, Religion, and Rituals, or Research Target Level in the context of spells
CSB=Convocation Skill Base
CSI=Convocation Skill Index
OP=Option Point
TR=Tuzyn Reckoning
COL=Columbia Games, Inc.
Ahn=Ahnu
Hir=Hirin
Ang=Angberelius
Lad=Lado
Ara=Aralius
Mas=Masara
Fen=Feneri
Nad=Nadai
Sko=Skorus
Tai=Tai
Tar=Tarael
Ula=Ulandus
SMP=Skill Maintenance Points
WAC=Weapon Attack Class
WDC=Weapon Defense Class
AML=Attack Mastery Level
DML=Defense Mastery Level
WQ=Weapon Quality
WT=Weight
A/D=Attack/Defense Classes
HM=Hand Mode
PR=Price
Ab=Abdomen
Bk=Back (Rear Tx Ab)
Ca=Calves
Ch=Chest(Front Tx Ab)
El=Elbow
Fa=Face
Fo=Forearms
Ft=Feet
Gr=Groin
Ha=Hands
Hp=Hips
Kn=Knees
Nk=Neck
Sh=Shoulders
Sk=Skull
Th=Thighs
Tx=Thorax
Ua=Upper Arms
LQ=Land Quality
AC=Gross Acres
HD=Households
[Acronyms end]

{context_str}

Query:
{query_str}

Answer:
"""
custom_qa_prompt = PromptTemplate(custom_qa_prompt_str)
query_engine = get_query_engine(data_path, storage_path)
query_engine.update_prompts({"response_synthesizer:text_qa_template": custom_qa_prompt})
