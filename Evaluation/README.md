# Overview

This repository contains various Python scripts used to generate and analyze data visualizations for our research. Each script corresponds to a specific figure in the research paper. Below, we detail how to configure and run the code for each figure, along with the corresponding Python scripts.

# Configuration

To run the code and generate the corresponding figures, please ensure the following:

1. **Install the required dependencies**:
   You can install the required dependencies by running the following command:

   ```bash
   pip install matplotlib==3.9.2
   pip install numpy==1.26.4
   pip install pandas==2.2.2
Ensure your environment is set up: Make sure you have **Python 3.11** installed along with the necessary libraries. You can use virtual environments to manage dependencies.


# Code and Figure Correspondence

## Fig.2
### Script: 
`generation.py`
### Description: 
This script generates the data and visualizations used in Figure 2.This is the reasoning data arrangement and visualization script of 15 Instance in the main experiment under different LLMs.

You can modify the `llmlist` to select different LLMs, and you can modify the `datasetlist` to choose different datasets.

Structure-Based Extraction
## Fig.3, Fig.4 and Fig.5
### Scripts:
`TokenMemory_SE.py`, `TokenMemory_PR.py`,`TokenMemory_Combine.py`,`GenerationTime.py` ,`Foldedhistogram-TokenTime.py`,`Foldedhistogram-Memory.py`
### Description: 
These scripts generate **Figures 3, Figures 4 and Figure 5**, which involves time,token and memory analysis across different Instances. Run the following scripts as needed:
`TokenMemory_SE.py`: For analyzing token memory in the Subgraph-Extraction Module.
`TokenMemory_PR.py`: For analyzing token memory in the Path-Retrieval Module.
`TokenMemory_Combine.py`: For combining Subgraph-Extraction Module and Path-Retrieval Module token and memory data of different Instances.
`GenerationTime.py`:For analyzing time of different Instances.
`Foldedhistogram-TokenTime.py`:For visualizing the token and time data.
`Foldedhistogram-Memory`:For visualizing the memory data.

## Fig.6
### Script:
`generationHop.py`
### Description: 
This script generates the data and visualizations used in **Figure 6**. Compare the influence of single-hop, two-hop and multi-hop data on Instance reasoning results.Ensure all necessary input data files are available before running the script.

## Fig.7
### Script:
`SE_structModel.py`
### Description: 
This script generates **Figure 7**, focusing on the Structure-Based Extraction used for analysis,such as PPR,RandWalk and KSE.

## Fig.8
### Script: 
`SE_smallModel.py`
### Description: 
This script generates **Figure 8**, related to the Semantic-Augmented Extraction,such as BM25, BGE and ST.

## Fig.9
### Script: 
`SE_LLMModel.py`
### Description: 
This script generates **Figure 9**, based on the results of the LLM model.
## Fig.10
### Script:
`PR.py`
Description: This script generates Figure 10, focusing on Path-Retrieval Module.
## Fig.11
Script: 
`PR_Scale_INone.py`
Description: This script generates Figure 11, exploring different scales in the Path-Retrieval Module.
## Fig.12
### Script: 
`PRTime.py`
### Description: 
This script generates Figure 12, focusing on the time-related analysis of Path-Retrieval Module results.
