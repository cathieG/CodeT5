# CodeT5
Project Objective: Fine tune CodeT5 for predicting if statements

Overview:
* [1 Introduction](#1-introduction)  
* [2 Getting Started](#2-getting-started)  
  * [2.1 Preparations](#21-preparations)  
  * [2.2 Install All Dependencies](#22-install-all-dependencies)  
* [3 Run Files](#3-run-files)
  * [3.1 Preprocessing.py](#31-preprocessing.py)
  * [3.2 Model.py](32-model.py)
  * [3.3 Evaluation.py](33-evaluation.py)
* [4 Other Documentations](#4-other-documentations)
---

# **1. Introduction**  
This project fine-tunes CodeT5, a pre-trained transformer model, to predict if statements in source code. 

We used the CodeXGLUE dataset, focusing on extracting and preprocessing code data relevant to conditional statements. After tokenizing the data with RobertaTokenizer, we fine-tuned CodeT5 using the T5ForConditionalGeneration model. The training process was managed using the Hugging Face Trainer API.

The trained model was evaluated on a test set using BLEU and CodeBLEU metrics to measure its prediction accuracy. Finally, the model was saved and zipped for easy distribution.

This project improves CodeT5's ability to handle conditional code structures, advancing automated code generation and analysis tools.

---

# **2. Getting Started**  

This project is implemented in **Python 3.9+** and is compatible with **macOS, Linux, and Windows**.  

## **2.1 Preparations**  

(1) Clone the repository to your workspace:  
```shell
git clone https://github.com/cathieG/CodeT5.git
```
(2) Navigate into the repository: (change according to your directory structure)
```shell
cd CodeT5
```
(3) Set up a virtual environment and activate it:

For macOS/Linux:
```shell
python -m venv ./venv/
source venv/bin/activate
```
For Windows:
```shell
python -m venv venv
venv\Scripts\activate
```

To deactivate the virtual environment, use the command:
```shell
deactivate
```
## **2.2 Install All Dependencies**

Install the required dependencies:
```shell
pip install -r requirements.txt
```
## **2.3 Clone CodeXGLUE Repository**

The CodeXGLUE dataset is required for this project. Clone the repository containing the dataset:
```shell
git clone https://github.com/microsoft/CodeXGLUE.git
```

# **3. Run Files**

The main steps of the project are as follows:

3.1 Preprocessing

3.2 Model Training

3.3 Evaluation

You have the option to run the project from the preprocessing or model training stage.

## **3.1 Preprocessing.py**
To run the file: 
```shell
python preprocessing.py
```
## **3.2 Model.py**

Run this script using the following command:
```shell
python model.py
```

## **3.3 Evaluation.py**
After running Model.py, the model is saved in a zipped folder. Therefore, run the following command to unzip it first:

On macOS/Linux:
```shell
unzip output_model.zip -d output-model
```
On windows:
```shell
Expand-Archive -Path "output_model.zip" -DestinationPath "output_model"
```
Then run the file:
```shell
python evaluation.py
```

# **4. Other Documentations**

- testset-results.csv: Contains the desired output acquired by running the scripts.

- CSCI_420_Assignment2_Report.pdf: Contains the write-up for this assignment.

- In "colab/" folder, we've included the google colab files for reference.

- In "data/" folder, "data/raw/" contains the training, testing, and validation data that the professor provided us.
"data/processed/" contains the flattened validation and test data that we preprocessed. Due to the flattened training data being too large, we couldn't include it in the repository.

- Additionally, we couldn't include the checkpoint (fine-tuned model) also because it was too large.






















