# NitiAI-AI-guide-to-Government-Schemes-and-Acts

## NitiAI: RAG + LangGraph Chatbot for Government Schemes and Acts
 
This repository contains the implementation of **NitiAI**, a Retrieval-Augmented Generation (RAG) based chatbot orchestrated with **LangGraph**.  
The project was developed as part of **DS246: Agentic & Generative AI** coursework.
 
NitiAI is designed to simplify access to information about Indian government schemes and acts. It provides reliable, grounded answers to factual queries and interpretive advice, especially for users with limited digital literacy.
 
---
 
### 📌 Problem Statement
 
Government schemes and acts are often complex, scattered across multiple portals, and difficult to understand.  
Challenges include:
- Long, jargon-heavy official documents.
- Language barriers and low digital literacy.
- Difficulty in finding eligibility, required documents, deadlines, and compliance steps.
- Frequent updates in rules (agriculture, taxation, etc.) making static resources unreliable.
 
NitiAI addresses these challenges by combining **RAG pipelines** with **LangGraph agent orchestration** to deliver simple, accurate, and contextual responses.
 
---
 
### 🎯 Objectives
 
- Build a RAG-based agent to answer queries on government schemes and acts.
- Use LangGraph for multi-agent workflows with routing, guardrails, and advisory support.
- Experiment with two RAG setups:
  - **Pipeline 1:** ChromaDB + BGE-large
  - **Pipeline 2:** Weaviate + BGE-M3 hybrid search
- Evaluate system performance on:
  - Answer relevancy
  - Faithfulness
  - Hallucination
  - Guardrail accuracy
 
---
 
### ⚙️ Key Features
 
- **Base Model:** Llama-3.1-8b-instruct  
- **Vector Databases:** ChromaDB and Weaviate  
- **Embeddings:** BGE-large / BGE-M3  
- **Rerankers:** BGE Reranker Large / BGE Reranker v2 M3  
- **LangGraph Agents:**
  - Orchestrator (routing + guardrails)
  - Q&A Agent (short factual answers with citations)
  - Advisor Agent (scenario-based interpretive advice)  
- **Evaluation Metrics:** Answer relevancy, faithfulness, hallucination, guardrail accuracy
 
---
 
### 📂 Repository Structure
 
#### 1. **Dataset**
Contains data collected from [india.gov.in](https://india.gov.in/my-government/schemes) and [acts portal](https://india.gov.in/my-government/acts-and-rule).  
- Scraped scheme pages, act sections, FAQs.  
- Cleaned and chunked into JSON format with metadata (scheme name, source file, URL, department).  
- JSON fields for chunks: `doc_type`, `doc_id`, `chunk_id`, `text`, `metadata`.
 
---
 
#### 2. **RAG Pipeline 1 (ChromaDB)**
Implements the first RAG setup using **ChromaDB**.  
Includes scripts for:
- Creating embeddings with **BGE-large**.  
- Storing embeddings into ChromaDB.  
- Retrieval using cosine similarity.  
- Re-ranking with **BGE Reranker Large**.  
- Grounded generation with **Llama-3.1-8b**.
 
This pipeline is optimized for simple, fast local setups with straightforward Python integration.
 
---
 
#### 3. **RAG Pipeline 2 (Weaviate)**
Implements the second RAG setup using **Weaviate**.  
Includes scripts for:
- Creating embeddings with **BGE-M3**.  
- Storing embeddings into Weaviate via API.  
- Hybrid retrieval (semantic + keyword/BM25) with adaptive alpha weighting.  
- Re-ranking with **BGE Reranker v2 M3**.  
- Grounded generation with **Llama-3.1-8b**.
 
This pipeline is optimized for multilingual retrieval and better hybrid recall.
 
---
 
#### 4. **LangGraph**
Implements the agent workflow using **LangGraph**.  
- **Retrieved_chunks file:** Handles query input and fetches relevant chunks from vector DBs.  
- **Agents inside LangGraph:**
  - **Orchestrator:** Routes queries to appropriate agents and enforces guardrails.  
  - **Q&A Agent:** Provides short factual answers with citations.  
  - **Advisor Agent:** Provides interpretive, legal, or practical suggestions.  
- Graph workflow ensures safe routing and multi-agent collaboration.
 
---
 
#### 5. **Evaluation**
Contains test datasets and evaluation prompt script.  
- **Test data JSON files:**  
  - 194 factual query-answer pairs for Q&A agent.  
  - 94 scenario-based query-answer pairs for Advisor agent.  
- **Prompt file:** Defines the evaluation prompt for LLM-as-judge.  
- Metrics evaluated: Answer relevancy, hallucination, faithfulness, guardrail accuracy.  
- Comparison between base LLM vs RAG-enhanced pipelines.
 
---
 
### 📊 Evaluation Results (Summary)
 
| Metric                | Base LLM | RAG (Chroma) | RAG (Weaviate) |
|------------------------|----------|--------------|----------------|
| Answer Relevancy (%)   | 34–55    | 71–78        | 65–77          |
| Hallucination (%)      | 18–40    | 25–45        | 9–28           |
| Faithfulness (%)       | 34–55    | 55–70        | 65–75          |
| Guardrail Accuracy (%) | 88–100   | 88–100       | 88–100         |
 
- RAG pipelines significantly improve relevancy and faithfulness.  
- Weaviate setup performs best in **Advisor mode**, with lowest hallucination rates.  
- Retrieval imperfections remain a challenge.
 
---
 
### 🚀 Future Work
 
- Integrate **Tavily MCP tool** for web search when data is missing.  
- Expand multilingual support for regional languages.  
- Build a **Streamlit UI** demo for chatbot interaction.
 
---
 
### 📝 How to Use
 
1. Clone the repository.  
2. Create HF token and Weaviate API key.
3. Run either **RAG Pipeline 1** or **RAG Pipeline 2** to build the vector database.  
4. Use **LangGraph** workflow to query via agents.  
5. Evaluate responses using files in **Evaluation/**.
 
---
 
### 📚 References
 
- [India.gov.in Schemes](https://india.gov.in/my-government/schemes)  
- [India.gov.in Acts & Rules](https://india.gov.in/my-government/acts-and-rule)  
- LangGraph documentation  
- Hugging Face Llama-3.1-8b-instruct  
 
---
 
### License
This project is for academic purposes under DS246 coursework.  
Please cite appropriately if reusing components.
