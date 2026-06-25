import firebase_admin
from firebase_admin import credentials, db
from datetime import datetime
import os

class FirebaseClinicalLogger:
    def __init__(self, cert_path="firebase-credentials.json"):
        self.enabled = False
        
        # Guard clause: check if credentials file exists
        if os.path.exists(cert_path):
            try:
                cred = credentials.Certificate(cert_path)
                # Initialize the app if it hasn't been initialized yet
                if not firebase_admin._apps:
                    firebase_admin.initialize_app(cred, {
                        'databaseURL': 'https://your-project-id-default-rtdb.firebaseio.com/' # Replace with your actual database URL if you set one up
                    })
                self.enabled = True
                print("⚡ Firebase Cloud Integration active. Realtime database connected.")
            except Exception as e:
                print(f"⚠️ Firebase initialization failed: {e}. Running in local-only fallback mode.")
        else:
            print("ℹ️ firebase-credentials.json not found. Application running in local-only fallback mode.")

    def log_patient_transaction(self, notes, metrics, risk_score, status):
        if not self.enabled:
            return "Local Save Only"
            
        try:
            ref = db.reference('clinical_logs')
            transaction_data = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'clinical_text_narrative': notes,
                'vitals': metrics,
                'diagnostic_risk_percentage': round(risk_score, 2),
                'triage_status': status
            }
            # Securely push as a new transaction node
            new_log_ref = ref.push(transaction_data)
            return new_log_ref.key
        except Exception as e:
            print(f"Error syncing transaction to Firebase cloud: {e}")
            return None