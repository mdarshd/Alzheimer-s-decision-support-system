import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageEnhance
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18
from prediction.csv_prediction import predict_from_clinical_form
from recommendation.recommendation_engine import get_recommendation
from llm.llm_interface import explain_clinical_decision, explain_mri_stage
WINDOW_W, WINDOW_H = 1100, 750
BORDER_COLOR = "#1E90FF"
BG_COLOR = "#0E0E0E"
FG_COLOR = "#F0F6FF"
BTN_BG = "#1C1C1C"
BTN_ACTIVE = "#2E6FFF"
LIGHT_GREEN_BORDER = "#7CFF7C"
BTN_FONT = ("Segoe UI", 12)
BTN_PADX = 28
BTN_PADY = 12
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
MRI_MODEL_PATH = os.path.join(BASE_DIR, "models", "mri_resnet18_stage.pth")
ORIGINAL_LABELS = {
    0: "Mild_Impairment",
    1: "Moderate_Impairment",
    2: "No_Impairment",
    3: "Very_Mild_Impairment",
}
UPDATED_STAGE_MAP = {
    "No_Impairment": "No Impairment",
    "Very_Mild_Impairment": "Mild Impairment",
    "Mild_Impairment": "Moderate Impairment",
    "Moderate_Impairment": "Severe Impairment",
}
MRI_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
frames = {}
clinical_data = {}
clinical_result = None
final_stage = None
def load_mri_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(ORIGINAL_LABELS))
    model.load_state_dict(
        torch.load(MRI_MODEL_PATH, map_location=device, weights_only=True)
    )
    model.eval().to(device)
    return model, device
def show_frame(name):
    for f in frames.values():
        f.pack_forget()
    frames[name].pack(fill="both", expand=True)
def load_bg(path, w, h):
    img = Image.open(path).resize((w, h), Image.Resampling.LANCZOS)
    img = ImageEnhance.Brightness(img).enhance(0.55)
    return ImageTk.PhotoImage(img)
def create_bg_frame(root, image_name):
    frame = tk.Frame(root, bg=BG_COLOR)
    canvas = tk.Canvas(frame, highlightthickness=0, bd=0)
    canvas.pack(fill="both", expand=True)
    def redraw(event):
        bg = load_bg(os.path.join(ASSETS_DIR, image_name), event.width, event.height)
        canvas.bg = bg
        canvas.delete("all")
        canvas.create_image(0, 0, image=bg, anchor="nw")
    canvas.bind("<Configure>", redraw)
    return frame
def blue_box(parent, width=800, rely=0.45):
    outer = tk.Frame(parent, bg=BORDER_COLOR)
    outer.place(relx=0.5, rely=rely, anchor="center", width=width)
    inner = tk.Frame(outer, bg=BG_COLOR)
    inner.pack(fill="both", expand=True, padx=2, pady=2)
    return inner
def themed_text(parent):
    t = tk.Text(
        parent, wrap="word", bg=BG_COLOR, fg=FG_COLOR,
        insertbackground="white", relief="flat", bd=0,
        highlightthickness=0, font=("Segoe UI", 13),
    )
    t.tag_config("heading", foreground="#66D9FF", font=("Segoe UI", 14, "bold"))
    t.tag_config("body", foreground=FG_COLOR)
    t.pack(fill="both", expand=True, padx=15, pady=15)
    return t
def build_welcome_frame(root):
    frame = create_bg_frame(root, "welcome.png")
    frames["welcome"] = frame

    tk.Button(
        frame, text="Start Clinical Assessment",
        font=BTN_FONT, padx=BTN_PADX, pady=BTN_PADY,
        command=lambda: show_frame("clinical"),
    ).place(relx=0.5, rely=0.7, anchor="center")


def build_clinical_frame(root):
    frame = create_bg_frame(root, "clinical.png")
    frames["clinical"] = frame
    box = blue_box(frame, width=520)
    box.grid_columnconfigure(1, weight=0)
    age = tk.StringVar()
    bmi = tk.StringVar()
    mmse = tk.StringVar()
    gender = tk.StringVar(value="")

    fields = [
        "Age", "BMI", "MMSE Score", "Gender",
        "Behavioral Problems", "Disorientation",
        "Difficulty Completing Tasks", "Forgetfulness",
        "Head Injury", "Family History Alzheimers",
    ]

    vars_map = {
        "Age": age,
        "BMI": bmi,
        "MMSE Score": mmse,
        "Behavioral Problems": tk.IntVar(),
        "Disorientation": tk.IntVar(),
        "Difficulty Completing Tasks": tk.IntVar(),
        "Forgetfulness": tk.IntVar(),
        "Head Injury": tk.IntVar(),
        "Family History Alzheimers": tk.IntVar(),
    }

    for i, field in enumerate(fields):
        tk.Label(
            box, text=field, bg=BG_COLOR, fg=FG_COLOR,
            font=("Segoe UI", 12), anchor="w",
        ).grid(row=i, column=0, sticky="w", padx=20, pady=10)

        if field in ["Age", "BMI", "MMSE Score"]:
            tk.Entry(
                box, textvariable=vars_map[field], width=25
            ).grid(row=i, column=1, sticky="w")

        elif field == "Gender":
            gender_frame = tk.Frame(box, bg=BG_COLOR)
            gender_frame.grid(row=i, column=1, sticky="w", pady=5)

            for val in ["Male", "Female"]:
                tk.Radiobutton(
                    gender_frame, text=val,
                    variable=gender, value=val,
                    indicatoron=False, width=12,
                    bg=BTN_BG, fg=FG_COLOR,
                    activebackground=BTN_BG,
                    activeforeground=FG_COLOR,
                    selectcolor=BTN_ACTIVE,
                    bd=0, relief="flat",
                ).pack(side="left", padx=6)

        else:
            tk.Checkbutton(
                box, variable=vars_map[field],
                bg=BG_COLOR, activebackground=BG_COLOR,
            ).grid(row=i, column=1, sticky="w")

    def submit_clinical():
        """Validate the form, run the clinical prediction model, and advance the UI."""
        if not all([age.get(), bmi.get(), mmse.get(), gender.get()]):
            messagebox.showerror("Error", "Please fill the clinical form completely.")
            return

        global clinical_data, clinical_result

        gender_text = gender.get()
        gender_numeric = 0 if gender_text == "Male" else 1

        clinical_data = {
            "Age": int(age.get()),
            "BMI": float(bmi.get()),
            "MMSE": int(mmse.get()),
            "Gender": gender_numeric,
            "GenderText": gender_text,
            **{
                k.replace(" ", ""): v.get()
                for k, v in vars_map.items() if isinstance(v, tk.IntVar)
            },
        }

        clinical_result = predict_from_clinical_form(clinical_data)
        show_frame("clinical_result")

    tk.Button(
        frame, text="Next",
        font=BTN_FONT, padx=BTN_PADX, pady=BTN_PADY,
        command=submit_clinical,
    ).place(relx=0.5, rely=0.9, anchor="center")
def build_clinical_result_frame(root):
    frame = create_bg_frame(root, "clinical.png")
    frames["clinical_result"] = frame

    box_res = blue_box(frame)
    text_res = themed_text(box_res)

    btn_mri = tk.Button(
        frame, text="Proceed to MRI",
        font=BTN_FONT, padx=BTN_PADX, pady=BTN_PADY,
        command=lambda: show_frame("mri"),
    )

    btn_finish = tk.Button(
        frame, text="Finish",
        font=BTN_FONT, padx=BTN_PADX, pady=BTN_PADY,
        command=root.quit,
    )

    def render_result(event=None):
        text_res.config(state="normal")
        text_res.delete("1.0", tk.END)

        diagnosis = clinical_result["Diagnosis"]

        text_res.insert(tk.END, "Diagnosis:\n", "heading")
        text_res.insert(tk.END, diagnosis + "\n\n", "body")

        text_res.insert(tk.END, "AI Explanation:\n", "heading")
        text_res.insert(
            tk.END,
            explain_clinical_decision(clinical_result, clinical_data),
            "body",
        )

        text_res.config(state="disabled")

        if "no" in diagnosis.lower():
            btn_finish.place(relx=0.5, rely=0.93, anchor="center")
        else:
            btn_mri.place(relx=0.5, rely=0.93, anchor="center")

    frame.bind("<Visibility>", render_result)


def build_mri_frame(root, model, device):
    """Build the MRI upload screen."""
    frame = create_bg_frame(root, "mri.png")
    frames["mri"] = frame

    def upload_mri():
        global final_stage
        path = filedialog.askopenfilename(
            filetypes=[("MRI Images", "*.jpg *.png")]
        )
        if not path:
            return

        img = MRI_TRANSFORM(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            idx = torch.argmax(model(img), dim=1).item()

        final_stage = UPDATED_STAGE_MAP[ORIGINAL_LABELS[idx]]
        show_frame("mri_result")

    tk.Button(
        frame, text="Upload MRI & Analyze",
        font=BTN_FONT,
        padx=BTN_PADX, pady=BTN_PADY,
        highlightbackground=LIGHT_GREEN_BORDER,
        highlightthickness=3, bd=0,
        command=upload_mri,
    ).place(relx=0.5, rely=0.6, anchor="center")


def build_mri_result_frame(root):
    frame = create_bg_frame(root, "mri.png")
    frames["mri_result"] = frame

    box_mri_res = blue_box(frame)
    text_mri = themed_text(box_mri_res)

    def render_mri(event=None):
        text_mri.config(state="normal")
        text_mri.delete("1.0", tk.END)

        text_mri.insert(tk.END, "MRI Stage:\n", "heading")
        text_mri.insert(tk.END, f"{final_stage}\n\n", "body")

        text_mri.insert(tk.END, "Recommendations:\n", "heading")
        for r in get_recommendation(final_stage)["Recommendations"]:
            text_mri.insert(tk.END, f"- {r}\n", "body")

        text_mri.insert(tk.END, "\nAI Explanation:\n", "heading")
        text_mri.insert(
            tk.END,
            explain_mri_stage(final_stage, clinical_data),
            "body",
        )

        text_mri.config(state="disabled")

    frame.bind("<Visibility>", render_mri)

    tk.Button(
        frame, text="Finish",
        font=BTN_FONT, padx=BTN_PADX, pady=BTN_PADY,
        command=root.quit,
    ).place(relx=0.5, rely=0.93, anchor="center")


def main():
    root = tk.Tk()
    root.title("Alzheimer's Clinical Decision Support System")
    root.geometry(f"{WINDOW_W}x{WINDOW_H}")
    root.configure(bg=BG_COLOR)

    model, device = load_mri_model()

    build_welcome_frame(root)
    build_clinical_frame(root)
    build_clinical_result_frame(root)
    build_mri_frame(root, model, device)
    build_mri_result_frame(root)

    show_frame("welcome")
    root.mainloop()


if __name__ == "__main__":
    main()
