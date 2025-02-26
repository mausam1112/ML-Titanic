![](https://github.com/mausam1112/ML-Titanic/blob/main/data/images/Titanics4.png)

# **Titanic Survival Prediction**

This repository contains the implementation of a machine learning model for predicting Titanic survival outcomes. The project follows a structured approach using configuration-based model selection, training, and evaluation.

## **Setup Instructions**

### 1. **Create a Virtual Environment**
Before installing dependencies, it's recommended to create a virtual environment:

```bash
python -m venv venv
```

### 2. **Activate the Virtual Environment**
- Windows
    ```bash
    .venv\Scripts\activate
    ```

- Mac/Linux
    ```bash
    source venv/bin/activate
    ```

### 3. **Install Dependencies**
```bash
pip install -r requirements.txt
```
Or
```bash
pip3 install -r requirements.txt
```

### 4. **Select the model to run train or eval action**
- Edit the file `src/configs/configs.py` to select the model name.
- Note: Only one of the `model_name` class variable must be uncommented.


### 5. **Train the model**
```bash
python run_train.py
```
Or
```bash
python3 run_train.py
```
Trained model are automatically saved under directory `saved_models/{Name}/v{version number}.{extension}`


### 5. **Evaluation**
- **Model Selection**
  - By default, the model version with the highest numeric suffix value will be choosen.
  - Distinct model version can be provided in `run_eval.py` file
  
- **Run Evaluation**
```bash
python run_eval.py
```
Or
```bash
python3 run_eval.py
```


