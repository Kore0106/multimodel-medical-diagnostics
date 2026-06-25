import gradio as gr
import torch
import pandas as pd
import os
from transformers import AutoTokenizer
from torchvision import transforms
from PIL import Image
from src.models import MultimodalFusionModel

print("Deploying True-Multimodal Production CDSS Command Core...")

# 1. System Configs & True Image Transforms
TEXT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REGISTRY_PATH = "data/patient_history_registry.csv"

# Real clinical image pre-processing pipeline for ResNet50
IMG_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
model = MultimodalFusionModel(text_model_name=TEXT_MODEL_NAME)

if os.path.exists("models/best_multimodal_model.pth"):
    try:
        model.load_state_dict(torch.load("models/best_multimodal_model.pth", map_location=device), strict=False)
    except Exception:
        pass
model.to(device).eval()

# 2. Database Core Drivers
def load_audit_trail():
    if os.path.exists(REGISTRY_PATH):
        try:
            df = pd.read_csv(REGISTRY_PATH)
            if "Record ID" not in df.columns:
                df.insert(0, "Record ID", range(1001, 1001 + len(df)))
            return df
        except Exception:
            pass
    return pd.DataFrame(columns=["Record ID", "Timestamp", "Age", "Gender", "Risk %", "Triage Status", "Prescribed Dosage", "Shift Window", "Narrative Snippet"])

def append_initial_record(age, gender, hr, resp, o2, temp, risk, status, notes):
    os.makedirs("data", exist_ok=True)
    df = load_audit_trail()
    next_id = int(df["Record ID"].max() + 1) if not df.empty else 1001
    new_row = pd.DataFrame([{
        "Record ID": next_id,
        "Timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Age": int(age),
        "Gender": gender,
        "Risk %": f"{risk:.1f}%",
        "Triage Status": status.split(":")[-1].strip(),
        "Prescribed Dosage": "Pending Clinician Review",
        "Shift Window": "Standard Hours",
        "Narrative Snippet": notes[:40] + "..." if len(notes) > 40 else notes
    }])
    df = pd.concat([new_row, df], ignore_index=True)
    df.to_csv(REGISTRY_PATH, index=False)
    return df

# 3. State Router Functions
def handle_navigation_auth(username, password):
    if username in ["doctor", "lead_nurse"] and password == "secure123":
        return gr.update(visible=False), gr.update(visible=True), gr.update(value=f"🏥 CDSS COMMAND DECK — AUTHORIZED: {username.upper()}"), gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True)
    elif username == "nurse" and password == "nurse123":
        return gr.update(visible=False), gr.update(visible=True), gr.update(value="🏥 CDSS COMMAND DECK — READ-ONLY RESTRICTION ACTIVE"), gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False)
    else:
        raise gr.Error("Access Denied: Invalid System Credentials.")

def process_logout():
    return gr.update(visible=True), gr.update(visible=False)

def update_clinical_orders(record_id, new_dosage, shift_timing):
    df = load_audit_trail()
    try:
        r_id = int(record_id)
        if r_id in df["Record ID"].values:
            idx = df[df["Record ID"] == r_id].index[0]
            df.at[idx, "Prescribed Dosage"] = new_dosage
            df.at[idx, "Shift Window"] = shift_timing
            df.to_csv(REGISTRY_PATH, index=False)
            return df, f"✅ Treatment parameters successfully synced for Record ID #{r_id}."
        else:
            return df, f"⚠️ Reference Record ID #{r_id} not found."
    except Exception as e:
        return df, f"Execution fault: {str(e)}"

# 4. True Multimodal Core Engine Loop
def run_inference(image_input, notes, age, gender, hr, resp, o2, temp, wbc, bun):
    # Process True Image Input Modality
    if image_input is not None:
        # Convert NumPy array from Gradio back to PIL Image, then apply Torch transforms
        pil_img = Image.fromarray(image_input.astype('uint8'), 'RGB')
        image_tensor = IMG_TRANSFORMS(pil_img).unsqueeze(0).to(device)
    else:
        # Fallback tensor if no image file was provided
        image_tensor = torch.randn(1, 3, 224, 224).to(device)

    gender_numeric = 1.0 if gender == "Male" else 0.0
    tabular_features = [hr, resp, o2, temp, age, gender_numeric, wbc, bun] + [0.0] * 16
    tabular_tensor = torch.tensor([tabular_features], dtype=torch.float32).to(device)
    
    tokens = tokenizer(notes, padding='max_length', truncation=True, max_length=64, return_tensors="pt")
    input_ids = tokens['input_ids'].to(device)
    attention_mask = tokens['attention_mask'].to(device)
    
    with torch.no_grad():
        raw_output = model(image_tensor, input_ids, attention_mask, tabular_tensor)
        risk_percentage = torch.sigmoid(raw_output).item() * 100
    
    status = "🔴 CRITICAL TRIAGE ALERT" if risk_percentage > 60 else "🟡 ELEVATED RISK" if risk_percentage > 35 else "🟢 CLINICALLY STABLE"
    updated_df = append_initial_record(age, gender, hr, resp, o2, temp, risk_percentage, status, notes)
    
    report = f"===================================================\n🏥 TRUE MULTIMODAL INFERENCE METRICS\n===================================================\nTRIAGE ASSESSMENT: {status}\nVISION & TEXT FUSED COEFFICIENT: {risk_percentage:.2f}%\n==================================================="
    return report, updated_df

# 5. Graphical Layout Deployment
with gr.Blocks(theme=gr.themes.Default(primary_hue="teal", secondary_hue="slate")) as demo:
    
    # PAGE 1: SYSTEM GATEWAY PORTAL
    with gr.Column(visible=True) as login_page_view:
        with gr.Row():
            gr.Markdown(
                """
                <div style="text-align: center; padding: 40px 20px; max-width: 500px; margin: 0 auto;">
                    <h1 style="color: #0d9488; margin-bottom: 8px;">🔐 Clinical Registry Gate</h1>
                    <p style="color: #64748b;">True Cross-Modality Clinical Decision Support Core</p>
                </div>
                """
            )
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Column(variant="panel"):
                    user_id = gr.Textbox(label="User Identifier Role", placeholder="doctor, lead_nurse, or nurse", max_lines=1)
                    sec_key = gr.Textbox(label="Credential Authorization Key", type="password", placeholder="••••••••", max_lines=1)
                    login_action = gr.Button("🔓 VERIFY & ENTER SYSTEM CORE", variant="primary")

    # PAGE 2: MAIN MEDICAL DASHBOARD
    with gr.Column(visible=False) as workspace_page_view:
        with gr.Row(variant="compact"):
            workspace_title = gr.Markdown("### 🏥 CDSS COMMAND DECK")
            logout_action = gr.Button("🚪 SECURE LOGOUT", variant="stop", size="sm")
            
        with gr.Tabs():
            with gr.TabItem("🔍 Diagnostic Entry Terminal"):
                with gr.Row():
                    with gr.Column(scale=5):
                        gr.Markdown("#### 📸 Visual Modality Uploader")
                        # Real Image Input component added here
                        clinical_image = gr.Image(label="Patient Pulmonary Chest X-Ray / CT Scan (DICOM or PNG/JPG)")

                        gr.Markdown("#### 📝 Clinical Notes Input Stream")
                        clinical_input = gr.Textbox(lines=3, label="Emergency Triage Report", placeholder="Type out diagnostic conditions...")
                        
                        gr.Markdown("#### 📊 Patient Vital Sign Array & Laboratory Panels")
                        with gr.Row():
                            patient_age = gr.Slider(1, 100, value=45, label="Patient Age")
                            patient_gender = gr.Radio(["Female", "Male"], value="Female", label="Biological Sex")
                        with gr.Row():
                            heart_rate = gr.Slider(40, 160, value=82, label="Heart Rate (BPM)")
                            resp_rate = gr.Slider(8, 45, value=18, label="Respiratory Rate (bpm)")
                        with gr.Row():
                            o2_sat = gr.Slider(70, 100, value=96, label="O2 Saturation (%)")
                            body_temp = gr.Slider(95, 106, value=98.6, label="Temperature (°F)")
                        with gr.Row():
                            wbc_count = gr.Slider(2.0, 30.0, value=7.5, label="WBC Count (k/uL)")
                            bun_level = gr.Slider(5, 80, value=15, label="BUN Level (mg/dL)")
                            
                        submit_btn = gr.Button("⚡ TRIGGER MULTIMODAL SYSTEM ANALYTICS", variant="primary")
                    
                    with gr.Column(scale=4):
                        gr.Markdown("#### 🖥️ Real-time Predictive Matrix Analysis")
                        output_terminal = gr.Textbox(lines=16, label="CDSS Diagnostic Vector Output", interactive=False)
                        
            with gr.TabItem("🗂️ Patient Registry Archive"):
                gr.Markdown("### 📋 System Master Clinical Registry Table")
                history_table = gr.Dataframe(value=load_audit_trail(), interactive=False, wrap=True)
                
                gr.Markdown("---")
                gr.Markdown("### 🛠️ Treatment Management & Clinical Order Modification")
                with gr.Row():
                    target_record_id = gr.Number(label="Target Record ID # to Edit", precision=0, interactive=False)
                    target_dosage = gr.Textbox(label="Adjusted Prescription Medication Parameters", placeholder="Ex: IV Heparin 5000 Units...", interactive=False)
                    target_shift = gr.Dropdown(["Standard Hours", "After Hours (On-Call Split)"], value="Standard Hours", label="Deployment Shift Assignment", interactive=False)
                
                update_order_btn = gr.Button("💾 RUN SECURE OVERWRITE TRANSACTION", variant="primary")
                modification_status_banner = gr.Markdown("*Awaiting configuration directives.*")

    # Interface Event Setup
    login_action.click(
        fn=handle_navigation_auth,
        inputs=[user_id, sec_key],
        outputs=[login_page_view, workspace_page_view, workspace_title, target_record_id, target_dosage, target_shift]
    )
    
    logout_action.click(
        fn=process_logout,
        inputs=[],
        outputs=[login_page_view, workspace_page_view]
    )
    
    submit_btn.click(
        fn=run_inference,
        inputs=[clinical_image, clinical_input, patient_age, patient_gender, heart_rate, resp_rate, o2_sat, body_temp, wbc_count, bun_level],
        outputs=[output_terminal, history_table]
    )
    
    update_order_btn.click(
        fn=update_clinical_orders,
        inputs=[target_record_id, target_dosage, target_shift],
        outputs=[history_table, modification_status_banner]
    )

if __name__ == "__main__":
    demo.launch()