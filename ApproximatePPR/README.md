



## Overview

This repository contains the implementation of the Fora, TopPPR and OurPPR algorithm, which is designed to compare and evaluate the effectiveness of three different Personalized PageRank (PPR) algorithms. In this repository, we present the following:

- **OurPPR.py**: Our implementation of the Personalized PageRank algorithm.
- **Fora**: The original implementation of the **Fora** algorithm.
- **TopPPR**: The original implementation of the **TopPPR** algorithm.

The experiment results will provide a detailed comparison of these three algorithms in terms of runtime and accuracy.

## Prerequisites

Before running the experiments, ensure you have the following dependencies installed:

- Python 3.11
- Required Python libraries:

```txt
pip install igraph
```

## Usage

### Running the experiment

To run the experiment and compare the results of the three PPR algorithms, simply execute the following command:


* **Fora**

Refer to the [Fora](https://github.com/wangsibovictor/fora) user manual to configure the dataset format, and then run the `.sh` script to calculate the results.
```bash
bash all_query.sh
```

* **TopPPR**

Similarly, refer to the [Topppr](https://github.com/wzskytop/TopPPR/tree/master) user manual for environment setup and dataset format conversion, then run the `.sh` script to calculate the results.

```bash
bash all_query.sh
```

* **OurPPR**

You can directly run the `.py` script to obtain the PPR results.

```bash
python OurPPR.py
```

## Results

Calculate the time and accuracy of different PPR through recall.py, and output it into a table.
```bash
python Recall.py
```