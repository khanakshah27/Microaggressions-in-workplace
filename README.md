
# Microaggressions in Workplace Detector

This project is a machine learning-powered web application that identifies microaggressions in workplace communication, with a particular focus on language that marginalizes women. It demonstrates how NLP and ML can be used to foster inclusivity and support human resource decision-making in real time.

## Features

* Detects microaggressive language in real-time using a trained logistic regression model
* Built with Flask for a smooth user experience and fast backend integration
* Provides a clean, Bootstrap-styled frontend interface for ease of use
* High model accuracy (91%) on validation data
* Practical implications for HR teams, diversity/inclusion training, and moderation tools

## Tech Stack

* **Frontend**: HTML, CSS, Bootstrap
* **Backend**: Python, Flask
* **Libraries**: Scikit-learn, Pandas, SpaCy, Jinja2

## Machine Learning Pipeline

1. **Data Preprocessing**

   * Cleaned real-world workplace communication samples
   * Performed lemmatization, stopword removal, and lowercasing using SpaCy

2. **Feature Engineering**

   * Employed TF-IDF vectorization to convert text into numerical feature vectors

3. **Model Training**

   * Trained a logistic regression classifier
   * Achieved 91% validation accuracy

4. **Web Integration**

   * Embedded the trained model into a Flask app
   * Enabled real-time prediction via a simple text input form

## Getting Started

### Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/your-username/Microaggressions-in-Workplace-Detector.git
cd Microaggressions-in-Workplace-Detector
pip install -r requirements.txt
```

### Run the App

```bash
python app.py
```

Visit `http://127.0.0.1:5000` in your browser to use the application.

## Folder Structure

```
├── app.py
├── model/
│   └── trained_model.pkl
├── templates/
│   └── index.html
├── static/
│   └── styles.css
├── requirements.txt
└── README.md
```

## Potential Use Cases

* HR review systems for flagging problematic communication
* Internal company tools for promoting respectful workplace culture
* Educational platforms focused on inclusivity and bias awareness

## Future Work

* Expand model to detect other forms of microaggressions (race, age, disability, etc.)
* Add support for multilingual detection
* Deploy on a cloud platform for broader accessibility

