# RCA agent
This project utilizes a hierarchical multi-agent RAG architecture for developing a Root cause analysis agent. 

It is based on LangGraph's supervisor framework model consisting of a supervisor agent which coordinates two specialized agents.​ The supervisor controls all communication flow and task delegation, making decisions about which agent to invoke based on the current context and task requirements. 
The two specialized agents are:​
1. RCA Technique Advisor agent: Agent specializes in suggesting the most appropriate RCA technique based on the event provided by the user. Has access to a knowledge base (here IEC-62740) stored in a vector store. ​
2. RCA Performer agent: Agent specializes in performing the analysis based on the event provided by the user and technique recommended by RCA Technique agent. Has access to files to collect evidence (here investigation reports).
![alt text](https://github.com/AbhilashAnand06/hackathon_rca_agent/blob/main/hackathon_architecture.png?raw=true)
