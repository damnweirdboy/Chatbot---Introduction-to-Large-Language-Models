# NLP Chatbot – Course Assistant 🤖

## 📌 Description

This project is an NLP-based chatbot designed to answer questions related to a university course.
It uses classical Natural Language Processing techniques implemented from scratch without external ML libraries.

---

## 🧠 Techniques Used

* Text Normalization (lowercasing, punctuation removal)
* Stemming
* Regular Expressions (Intent Detection)
* ELIZA-style conversational rules
* TF-IDF with N-grams
* Cosine Similarity
* Edit Distance (Levenshtein)
* Confidence Threshold with fallback responses

---

## ⚙️ How It Works

1. User input is normalized
2. Intent is detected using regex
3. TF-IDF vectorization is applied
4. Cosine similarity finds the best match
5. Edit distance handles typos
6. ELIZA rules handle conversational inputs
7. If confidence is low → fallback response

---

## ▶️ How to Run

```bash
python chatbot.py
```

---

## 💬 Example

```
You: what is NLP?
Bot: NLP stands for Natural Language Processing...
```

---

## 📂 Project Structure

```
chatbot.py
data.json
README.md
```

---

## 🎯 Notes

This chatbot is implemented from scratch to demonstrate fundamental NLP concepts without relying on external libraries like scikit-learn.

---
