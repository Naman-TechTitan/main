import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import tkinter as tk
from tkinter import messagebox
import pyttsx3
import webbrowser

class MedicalChatbot:
    def __init__(self, root):
        self.root = root
        self.create_gradient_background()

    def __init__(self, data_path='medical_data.csv'):
        self.data_path = data_path
        self.label_encoder = LabelEncoder()
        self.question_bank = self.load_questions()
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('clf', MultinomialNB())
        ])
        self.load_data()
        self.skip_remaining_questions = False  # To manage skipping questions
        self.engine = pyttsx3.init()  # Text-to-speech engine

    def load_questions(self):
        return [
            "Are you experiencing fever?",
            "Do you feel fatigue or weakness?",
            "Do you have a headache?",
            "Do you have a cough?",
            "Is there any difficulty breathing?",
            "Do you have a runny nose or sore throat?",
            "Do you have any stomach pain?",
            "Have you experienced nausea or vomiting?",
            "Any changes in appetite?",
            "Are you experiencing chest pain?",
            "Do you have shortness of breath?",
            "Any rapid or irregular heartbeat?",
            "Do you have any rash?",
            "Is there itching or swelling?",
            "Have you noticed any skin changes?"
        ]

    def load_data(self):
        try:
            df = pd.read_csv(self.data_path)
            symptoms_cols = [col for col in df.columns if col != 'disease']
            self.symptoms_data = [
                ' '.join([symptom for symptom in row.index if row[symptom] == 1 or row[symptom] == '1'])
                for _, row in df[symptoms_cols].iterrows()
        ]
            self.diagnoses = self.label_encoder.fit_transform(df['disease'])
            self.train_model()
        except FileNotFoundError:
            raise FileNotFoundError("The required 'medical_data.csv' file is missing. Please provide the file and restart the application.")

    def train_model(self):
        X_train, X_test, y_train, y_test = train_test_split(self.symptoms_data, self.diagnoses, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)

    def speak(self, text):
        self.engine.say(text)
        self.engine.runAndWait()

    def get_user_symptoms_voice(self):
        self.symptoms = []
        self.question_index = 0
        self.skip_remaining_questions = False
        self.ask_question_voice()

    def ask_question_voice(self):
        if self.question_index < len(self.question_bank):
            question = self.question_bank[self.question_index]
            self.speak(question)
            self.show_voice_symptom_prompt(question)
        else:
            self.get_diagnosis_and_recommend()

    def show_voice_symptom_prompt(self, question):
        dialog = tk.Toplevel()
        dialog.title("Voice Symptom Inquiry")
        dialog.geometry("1600x900")
        dialog.configure(bg='black')

        label = tk.Label(dialog, text=question, font=("Arial", 40), bg='black', fg='white')
        label.pack(pady=50)

        button_frame = tk.Frame(dialog, bg='black')
        button_frame.pack(pady=50)

        yes_button = tk.Button(button_frame, text="Yes", command=lambda: self.record_symptom_voice(dialog, True), bg="#4CAF50", fg="white", font=("Arial", 25))
        yes_button.pack(side=tk.LEFT, padx=20)

        no_button = tk.Button(button_frame, text="No", command=lambda: self.record_symptom_voice(dialog, False), bg="#ec0000", fg="white", font=("Arial", 25))
        no_button.pack(side=tk.LEFT, padx=20)

        skip_button = tk.Button(button_frame, text="Skip rest of the questions", command=lambda: self.skip_voice_questions(dialog), bg="#FFA500", fg="white", font=("Arial", 25))
        skip_button.pack(side=tk.LEFT, padx=20)

        dialog.wait_window()

    def skip_voice_questions(self, dialog):
        dialog.destroy()
        self.skip_remaining_questions = True  # Set the flag to skip remaining questions
        self.get_diagnosis_and_recommend()  # Directly get diagnosis

    def record_symptom_voice(self, dialog, answer):
        if answer:
            symptom = self.question_bank[self.question_index]
            self.symptoms.append(symptom)
        dialog.destroy()
        self.question_index += 1
        self.ask_question_voice()

    def get_user_symptoms(self):
        self.symptoms = []
        self.question_index = 0
        self.skip_remaining_questions = False  # Reset skip flag for new session
        self.ask_question()

    def ask_question(self):
        if self.question_index < len(self.question_bank):
            question = self.question_bank[self.question_index]
            self.show_symptom_prompt(question)
        else:
            self.get_diagnosis_and_recommend()

    def show_symptom_prompt(self, question):
        dialog = tk.Toplevel()
        dialog.title("Symptom Inquiry")
        dialog.geometry("1600x900")
        dialog.configure(bg='black')

        label = tk.Label(dialog, text=question, font=("Arial", 40), bg='black', fg='white')
        label.pack(pady=20)

        button_frame = tk.Frame(dialog, bg='black')
        button_frame.pack(pady=10)

        yes_button = tk.Button(button_frame, text="Yes", command=lambda: self.record_symptom(dialog, True), bg="#4CAF50", fg="white", font=("Arial", 25))
        yes_button.pack(side=tk.LEFT, padx=5)

        no_button = tk.Button(button_frame, text="No", command=lambda: self.record_symptom(dialog, False), bg="#ec0000", fg="white", font=("Arial", 25))
        no_button.pack(side=tk.LEFT, padx=5)

        skip_button = tk.Button(button_frame, text="Skip rest of the questions", command=lambda: self.skip_questions(dialog), bg="#FFA500", fg="white", font=("Arial", 25))
        skip_button.pack(side=tk.LEFT, padx=5)

        dialog.wait_window()

    def skip_questions(self, dialog):
        dialog.destroy()
        self.skip_remaining_questions = True  # Set the flag to skip remaining questions
        self.get_diagnosis_and_recommend()  # Directly get diagnosis

    def record_symptom(self, dialog, answer):
        if answer:
            symptom = self.question_bank[self.question_index]
            self.symptoms.append(symptom)
        dialog.destroy()
        self.question_index += 1
        self.ask_question()

    def get_diagnosis_and_recommend(self):
        symptoms_str = ' '.join(self.symptoms)
        prediction = self.model.predict([symptoms_str])
        predicted_disease = self.label_encoder.inverse_transform(prediction)[0]
        confidence = np.max(self.model.predict_proba([symptoms_str])) * 100
        messagebox.showinfo("Diagnosis", f"Predicted Disease: {predicted_disease}\nConfidence: {confidence:.2f}%")
        self.give_recommendations(predicted_disease)

    def give_recommendations(self, disease):
        recommendations = {
            'Common Cold': ["Stay hydrated.", "Rest well.", "Use a humidifier."],
            'Bronchitis': ["Avoid smoking.", "Use a cough suppressant.", "Drink warm fluids."],
            'Gastroenteritis': ["Stay hydrated.", "Eat bland foods.", "Avoid dairy."],
            'Respiratory Infection': ["Use a vaporizer.", "Stay warm.", "Consult a doctor if symptoms worsen."],
            'Allergic Reaction': ["Avoid allergens.", "Use antihistamines.", "Consult a doctor if needed."]
        }
        recs = recommendations.get(disease, ["Consult a doctor for proper treatment."])
        messagebox.showinfo("Recommendations", "\n".join(recs))

    def open_feedback_form(self):
        feedback_url = "https://forms.gle/WCGzW9eX5ooBZz1b7" 
        webbrowser.open(feedback_url)

    def search_nearby(self):
        query = "https://www.google.com/maps/search/hospitals+or+pharmacies+near+me"
        webbrowser.open(query)

# Main GUI Setup
if __name__ == "__main__":
    chatbot = MedicalChatbot()

    root = tk.Tk()
    root.title("Vital Vision Chatbot")
    root.geometry("1600x900")

    # Header
    header_frame = tk.Frame(root, bg="#2196F3", pady=20)
    header_frame.pack(fill=tk.X)
    header_label = tk.Label(header_frame, text="Vital Vision Chatbot", font=("Times New Roman", 40), bg="#2196F3", fg="white")
    header_label.pack(side=tk.TOP)
    
    # Right corner buttons in header
    header_buttons_frame = tk.Frame(header_frame, bg="#2196F3")
    header_buttons_frame.pack(side=tk.RIGHT, padx=20)
    
    about_button = tk.Button(header_buttons_frame, text="About Us", command=lambda: messagebox.showinfo("About Us", "Vital Vision is a medical chatbot..."), bg="#4CAF50", fg="white", font=("Arial", 12))
    about_button.pack(side=tk.LEFT, padx=10)
    
    contact_button = tk.Button(header_buttons_frame, text="Contact Us", command=lambda: messagebox.showinfo("Contact Us", "Email: contact@vitalvision.com"),  bg="#4CAF50", fg="white", font=("Arial", 12))
    contact_button.pack(side=tk.LEFT, padx=10)
    
    feedback_button = tk.Button(header_buttons_frame, text="Feedback", command=chatbot.open_feedback_form, bg="#4CAF50", fg="white", font=("Arial", 12))
    feedback_button.pack(side=tk.LEFT, padx=10)
    
    # Main Buttons
    text_button = tk.Button(root, text="Text-based Diagnosis", command=chatbot.get_user_symptoms, bg="#FF5722", fg="white", font=("Arial", 20), pady=20,padx=5)
    text_button.pack(pady=70,padx=100)
    
    voice_button = tk.Button(root, text="Voice-based Diagnosis", command=chatbot.get_user_symptoms_voice, bg="#FFA500", fg="white", font=("Arial", 20), pady=20,padx=20)
    voice_button.pack(pady=70,padx=100)

    map_button = tk.Button(root, text="Find Hospitals/Pharmacies nearby", command=chatbot.search_nearby, bg="#006400", fg="white", font=("Arial", 20), pady=20,padx=10)
    map_button.pack(pady=70,padx=100)

    # Footer
    footer_frame = tk.Frame(root, bg="#2196F3", pady=10)
    footer_frame.pack(side=tk.BOTTOM, fill=tk.X)
    footer_label = tk.Label(footer_frame, text="© 2024 Vital Vision. All rights reserved.", font=("Arial", 10), bg="#2196F3", fg="white")
    footer_label.pack()

    root.mainloop()
