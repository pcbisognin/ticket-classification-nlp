# Ticket Classification with NLP

This project builds a supervised NLP model to automatically classify customer support tickets (free-text chat complaints) into predefined categories, enabling faster routing to the correct specialist team.

The notebook explores multiple NLP representations (n-grams and embeddings) and compares model performance using **weighted F1 score** on a **75/25 train-test split** with `random_state=42`. The final (“champion”) pipeline achieves **F1 > 0.75**.

## Dataset

- Source CSV: `tickets_reclamacoes_classificados.csv`
- URL: https://dados-ml-pln.s3.sa-east-1.amazonaws.com/tickets_reclamacoes_classificados.csv

The dataset contains (among other fields) a text description of the complaint and its category label.  
This project focuses on:
- `descricao_reclamacao` (text)
- `categoria` (label)

## What was done

### Text preprocessing
The notebook applies practical cleaning steps to improve signal-to-noise ratio, including:
- lowercasing
- removal of digits and special characters
- removal of placeholder tokens (e.g., `xx`/`xxxx` used to mask personal data)
- stopword removal (Portuguese), with a small customization to keep relevant tokens (e.g., “não”)

### Modeling approaches tested
All approaches use **Logistic Regression** as a strong, interpretable baseline classifier:

1. **Bag-of-Words (CountVectorizer)**
   - unigrams
   - unigrams + bigrams

2. **TF-IDF (TfidfVectorizer)**
   - unigrams
   - unigrams + bigrams

3. **Word Embeddings**
   - pre-trained Word2Vec (document vector as the mean of word vectors)

4. **Sentence Embeddings**
   - SentenceTransformer multilingual model (sentence-level embeddings)

## Best model (champion pipeline)

**CountVectorizer (1–2 grams) + Logistic Regression**  
This configuration produced the best overall results on the test set (weighted F1 above the required threshold).

> Note: In this dataset, the simpler n-gram approach outperformed more complex embedding-based approaches, which can happen when categories are strongly driven by local keywords and short phrases.

## Results

Evaluation is reported using:
- accuracy
- precision / recall / F1 per class
- **weighted F1** (main metric)

The champion model exceeds **F1 > 0.75** on the test set.

## How to run

### Option 1: Run the notebook
1. Clone the repository
2. Install dependencies
3. Run `NLP_final.ipynb` in Jupyter/VS Code


pip install -r requirements.txt
