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
* [4 Report](#4-report)  

---

# **1. Introduction**  
Will add some introductions later

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

You have the option to run the project from different stages. The main steps of the project are as follows:

3.1 Preprocessing

3.2 Model Training

3.3 Evaluation

This repository includes all intermediate results, allowing you to start from any stage of the project without interruption.

Now, jump to the section that you wish to start from!

## **3.1 Preprocessing.py**
To run the file: 
```shell
python preprocessing.py
```
## **3.2 Model.py**

**Note:** If you previously ran `preprocessing.py` and generated your own flattened files, please follow the instructions in this file to update the file paths accordingly so you can proceed with training using your data. 

Otherwise, you can run this script directly using the following command:
```shell
python model.py
```

## **3.3 Evaluation.py**




























