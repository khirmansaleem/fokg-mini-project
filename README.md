Copy **exactly this** into your `README.md` file:

````markdown
# FOKG Mini Project: Fact Checking over Knowledge Graphs

This project is my solution for the mini project in **Foundations of Knowledge Graphs**.

The task is to assign a truth score between `0` and `1` to RDF statements. A score close to `1` means the fact is likely true, while a score close to `0` means the fact is likely false.

The final output file is:

```text
outputs/result.ttl
```

This file is used for GERBIL evaluation on **SW 2022 Test**.

---
## How to Run Quickly

The project can be run directly from the terminal.

Clone the repository:

```bash
git clone https://github.com/khirmansaleem/fokg-mini-project.git
cd fokg-mini-project
```

Install the required packages:

```bash
python -m pip install -r requirements.txt
```

Run the project:

```bash
python main.py
```

After the script finishes, the final result file will be generated here:

```text
outputs/result.ttl
```

## Project Structure

```text
fokg-mini-project/
│
├── data/
│   ├── train.nt.txt
│   └── test.nt.txt
│
├── outputs/
│   ├── result.ttl
│   ├── train_result.ttl
│   ├── kg_features_train.csv
│   └── kg_features_test.csv
│
├── src/
│   ├── __init__.py
│   ├── parser.py
│   ├── baseline.py
│   ├── improved_model.py
│   ├── tuning.py
│   ├── kg_features.py
│   ├── kg_rules.py
│   ├── writer.py
│   └── validator.py
│
├── main.py
├── requirements.txt
└── README.md
```

---

## Requirements

The project uses Python 3.

The required packages are listed in `requirements.txt`:

```text
rdflib
pandas
scikit-learn
requests
```

---

## Input Files

The input files must be placed inside the `data/` folder:

```text
data/train.nt.txt
data/test.nt.txt
```

The training file contains RDF statements with truth values.

The test file contains RDF statements without truth values.

---

## How to Run

Run all commands from the main project folder.

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

Install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run the project:

```powershell
python main.py
```

After the script finishes, the final result file will be created here:

```text
outputs/result.ttl
```

For GERBIL test evaluation, upload:

```text
outputs/result.ttl
```

and select:

```text
SW 2022 Test
```

For GERBIL training evaluation, upload:

```text
outputs/train_result.ttl
```

and select:

```text
SW 2022 Train
```

---

## Method

The solution uses three main steps.

First, a predicate-based baseline is created. For each predicate, the average truth value in the training data is calculated and used as a basic prediction score.

Second, a lightweight pattern-based model is trained. It uses general features such as predicate type, object shape, object category clues, and simple predicate-object mismatch patterns.

Third, DBpedia is used as an external knowledge graph. The system checks whether the exact candidate triple exists in DBpedia. It also uses related evidence such as subject-predicate and predicate-object counts.

The strongest signal was whether the exact triple already exists in DBpedia. This knowledge graph evidence improved the result clearly.

---

## Output Format

The generated result file follows this format:

```ttl
<Fact-URI> <http://swc2017.aksw.org/hasTruthValue> "score"^^<http://www.w3.org/2001/XMLSchema#double> .
```

Example:

```ttl
<http://swc2017.aksw.org/task2/dataset/3417193> <http://swc2017.aksw.org/hasTruthValue> "0.876543"^^<http://www.w3.org/2001/XMLSchema#double> .
```

The script validates the generated RDF output before finishing.

---

## DBpedia Feature Cache

The first run may take some time because the script queries DBpedia.

The extracted DBpedia features are cached in the `outputs/` folder:

```text
outputs/kg_features_train.csv
outputs/kg_features_test.csv
```

If these files already exist, the script loads them directly instead of querying DBpedia again.

---

## GERBIL Result

Final GERBIL result on **SW 2022 Test**:

```text
ROC AUC: 0.8444
```
````
