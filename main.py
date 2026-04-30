from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
import customtkinter as ctk

from capture import capture_faces
from recognize import run_recognition
from train import train_model
from utils import (
    ATTENDANCE_FILE,
    LABELS_FILE,
    TRAINER_DIR,
    get_next_person_id,
    list_registered_students,
    normalize_video_source,
    read_attendance_records,
)

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class SmartAttendanceDashboard:
    def __init__(self, root: ctk.CTk) -> None:
        self.root = root
        self.root.title("Smart Attendance Dashboard")
        self.root.geometry("1200x800")
        self.root.minsize(1080, 680)

        self.status_var = tk.StringVar(value="Dashboard ready.")
        self.busy_var = tk.StringVar(value="Idle")
        self.name_var = tk.StringVar()
        self.person_id_var = tk.StringVar(value=str(get_next_person_id()))
        self.samples_var = tk.StringVar(value="45")
        self.camera_var = tk.StringVar(value="0")
        self.source_var = tk.StringVar()
        self.backend_var = tk.StringVar(value="default")

        self._build_styles()
        self._build_layout()
        self.refresh_data()

    def _build_styles(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        
        # Dark theme colors for Treeview
        bg_color = "#2b2b2b"
        fg_color = "#dce4ee"
        selected_bg = "#1f538d"
        
        style.configure(
            "Treeview",
            background=bg_color,
            foreground=fg_color,
            fieldbackground=bg_color,
            rowheight=35,
            font=("Segoe UI", 11),
            borderwidth=0
        )
        style.configure(
            "Treeview.Heading",
            background="#1f1f1f",
            foreground=fg_color,
            font=("Segoe UI", 12, "bold"),
            borderwidth=0
        )
        style.map(
            "Treeview",
            background=[("selected", selected_bg)],
            foreground=[("selected", "white")]
        )
        style.map(
            "Treeview.Heading",
            background=[("active", "#333333")]
        )

    def _build_layout(self) -> None:
        # Main container
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # Hero Header
        hero = ctk.CTkFrame(self.root, corner_radius=0, fg_color="#1f538d")
        hero.grid(row=0, column=0, sticky="ew", ipadx=20, ipady=20)
        
        ctk.CTkLabel(
    hero,
    text="Face Recognition Attendance System",
    font=ctk.CTkFont(family="Segoe UI", size=25, weight="bold"),
    text_color="white",
    anchor="center",
    justify="center"
).pack(fill="x", padx=20, pady=(10, 0))
        ctk.CTkLabel(
    hero,
    text="Register students, collect datasets, train the recognizer, and monitor live attendance.",
    font=ctk.CTkFont(family="Segoe UI", size=14),
    text_color="#dce4ee",
    anchor="center",
    justify="center"
).pack(fill="x", padx=20, pady=(0,10))
        # Content area
        content = ctk.CTkFrame(self.root, fg_color="transparent")
        content.grid(row=1, column=0, sticky="nsew", padx=20, pady=20)
        content.grid_columnconfigure(0, weight=3)
        content.grid_columnconfigure(1, weight=5)
        content.grid_rowconfigure(1, weight=1)

        # Stats Row
        stats_row = ctk.CTkFrame(content, fg_color="transparent")
        stats_row.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 20))
        stats_row.grid_columnconfigure((0, 1, 2, 3), weight=1)
        
        self.student_count_label = self._build_stat_card(stats_row, 0, "Registered Students")
        self.image_count_label = self._build_stat_card(stats_row, 1, "Dataset Images")
        self.attendance_count_label = self._build_stat_card(stats_row, 2, "Attendance Records")
        self.model_status_label = self._build_stat_card(stats_row, 3, "Model Status")

        # Left Panel
        left_panel = ctk.CTkScrollableFrame(content, corner_radius=10, fg_color="#2b2b2b")
        left_panel.grid(row=1, column=0, sticky="nsew", padx=(0, 10))

        # Right Panel
        right_panel = ctk.CTkFrame(content, corner_radius=10, fg_color="#2b2b2b")
        right_panel.grid(row=1, column=1, sticky="nsew")
        right_panel.grid_rowconfigure(1, weight=1)
        right_panel.grid_rowconfigure(3, weight=1)
        right_panel.grid_columnconfigure(0, weight=1)

        self._build_registration_panel(left_panel)
        self._build_actions_panel(left_panel)
        self._build_status_panel(left_panel)
        
        self._build_students_table(right_panel)
        self._build_attendance_table(right_panel)

    def _build_stat_card(self, parent, col, caption):
        card = ctk.CTkFrame(parent, corner_radius=10, fg_color="#2b2b2b")
        card.grid(row=0, column=col, sticky="ew", padx=10)
        
        value_label = ctk.CTkLabel(card, text="0", font=ctk.CTkFont(size=28, weight="bold"), text_color="#1f538d")
        value_label.pack(anchor="w", padx=20, pady=(20, 5))
        
        caption_label = ctk.CTkLabel(card, text=caption, font=ctk.CTkFont(size=12))
        caption_label.pack(anchor="w", padx=20, pady=(0, 20))
        return value_label

    def _build_registration_panel(self, parent):
        title = ctk.CTkLabel(parent, text="Student Registration", font=ctk.CTkFont(size=18, weight="bold"))
        title.pack(anchor="w", padx=15, pady=(15, 5))
        
        desc = ctk.CTkLabel(parent, text="Add a profile and collect face dataset.", font=ctk.CTkFont(size=12), text_color="gray")
        desc.pack(anchor="w", padx=15, pady=(0, 15))

        form = ctk.CTkFrame(parent, fg_color="transparent")
        form.pack(fill="x", padx=15)
        form.grid_columnconfigure(1, weight=1)

        self._add_form_row(form, 0, "Student ID", self.person_id_var, readonly=True)
        self._add_form_row(form, 1, "Full Name", self.name_var)
        self._add_form_row(form, 2, "Samples", self.samples_var)
        self._add_form_row(form, 3, "Camera Index", self.camera_var)
        self._add_form_row(form, 4, "Source URL", self.source_var)

        ctk.CTkLabel(form, text="Backend").grid(row=5, column=0, sticky="w", pady=8, padx=(0, 10))
        backend_combo = ctk.CTkComboBox(
            form, 
            variable=self.backend_var, 
            values=["default", "msmf", "dshow", "auto"],
            state="readonly"
        )
        backend_combo.grid(row=5, column=1, sticky="ew", pady=8)

        btn = ctk.CTkButton(parent, text="Capture Dataset", command=self.start_capture, font=ctk.CTkFont(weight="bold"))
        btn.pack(fill="x", padx=15, pady=(20, 10))

    def _build_actions_panel(self, parent):
        title = ctk.CTkLabel(parent, text="System Actions", font=ctk.CTkFont(size=18, weight="bold"))
        title.pack(anchor="w", padx=15, pady=(25, 5))

        ctk.CTkButton(parent, text="Train Model", command=self.start_training, fg_color="#10b981", hover_color="#059669", font=ctk.CTkFont(weight="bold")).pack(fill="x", padx=15, pady=8)
        ctk.CTkButton(parent, text="Start Recognition", command=self.start_recognition, fg_color="#8b5cf6", hover_color="#7c3aed", font=ctk.CTkFont(weight="bold")).pack(fill="x", padx=15, pady=8)
        ctk.CTkButton(parent, text="Refresh Dashboard", command=self.refresh_data, fg_color="transparent", border_width=1, text_color=("gray10", "gray90")).pack(fill="x", padx=15, pady=8)

    def _build_status_panel(self, parent):
        title = ctk.CTkLabel(parent, text="Status", font=ctk.CTkFont(size=18, weight="bold"))
        title.pack(anchor="w", padx=15, pady=(25, 5))
        
        status_lbl = ctk.CTkLabel(parent, textvariable=self.busy_var, font=ctk.CTkFont(size=12, weight="bold"), text_color="#10b981")
        status_lbl.pack(anchor="w", padx=15)

        self.status_box = ctk.CTkTextbox(parent, height=100, font=ctk.CTkFont(family="Consolas", size=12), wrap="word", corner_radius=5)
        self.status_box.pack(fill="x", padx=15, pady=10)
        self.status_box.insert("1.0", self.status_var.get())
        self.status_box.configure(state="disabled")

    def _build_students_table(self, parent):
        lbl = ctk.CTkLabel(parent, text="Registered Students", font=ctk.CTkFont(size=18, weight="bold"))
        lbl.grid(row=0, column=0, sticky="w", padx=20, pady=(20, 10))

        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 20))
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        columns = ("person_id", "name", "images")
        tree = ttk.Treeview(frame, columns=columns, show="headings", height=8)
        for column, title, width in (("person_id", "ID", 70), ("name", "Name", 200), ("images", "Images", 100)):
            tree.heading(column, text=title)
            tree.column(column, width=width, anchor="center")
        tree.grid(row=0, column=0, sticky="nsew")
        
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        tree.configure(yscrollcommand=scrollbar.set)
        
        self.students_tree = tree

    def _build_attendance_table(self, parent):
        lbl = ctk.CTkLabel(parent, text="Attendance Records", font=ctk.CTkFont(size=18, weight="bold"))
        lbl.grid(row=2, column=0, sticky="w", padx=20, pady=(10, 10))

        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.grid(row=3, column=0, sticky="nsew", padx=20, pady=(0, 20))
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        columns = ("person_id", "name", "timestamp")
        tree = ttk.Treeview(frame, columns=columns, show="headings", height=10)
        for column, title, width in (("person_id", "ID", 70), ("name", "Name", 200), ("timestamp", "Timestamp", 250)):
            tree.heading(column, text=title)
            tree.column(column, width=width, anchor="center")
        tree.grid(row=0, column=0, sticky="nsew")
        
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        tree.configure(yscrollcommand=scrollbar.set)
        
        self.attendance_tree = tree

    def _add_form_row(self, parent, row, label, variable, readonly=False):
        lbl = ctk.CTkLabel(parent, text=label, font=ctk.CTkFont(size=12))
        lbl.grid(row=row, column=0, sticky="w", pady=8, padx=(0, 10))
        
        state = "readonly" if readonly else "normal"
        # CTkEntry behaves slightly differently with state
        entry = ctk.CTkEntry(parent, textvariable=variable, state=state)
        entry.grid(row=row, column=1, sticky="ew", pady=8)
        if readonly:
            entry.configure(text_color="gray")

    def set_status(self, message: str, busy: str = "Idle") -> None:
        self.status_var.set(message)
        self.busy_var.set(busy)
        self.status_box.configure(state="normal")
        self.status_box.delete("1.0", "end")
        self.status_box.insert("1.0", message)
        self.status_box.configure(state="disabled")

    def run_in_thread(self, work, success_message: str) -> None:
        def worker() -> None:
            try:
                result = work()
            except Exception as exc:
                error_message = str(exc)
                self.root.after(0, lambda msg=error_message: self.set_status(f"Error: {msg}", "Error"))
                self.root.after(0, lambda msg=error_message: messagebox.showerror("Operation Failed", msg))
                return

            final_message = success_message.format(result=result)
            self.root.after(0, lambda: self.set_status(final_message, "Completed"))
            self.root.after(0, lambda msg=final_message: messagebox.showinfo("Success", msg))
            self.root.after(0, self.refresh_data)

        threading.Thread(target=worker, daemon=True).start()

    def start_capture(self) -> None:
        try:
            samples = int(self.samples_var.get().strip())
            camera_index = int(self.camera_var.get().strip())
        except ValueError:
            messagebox.showerror("Invalid Input", "Samples and Camera Index must be numbers.")
            return

        name = self.name_var.get().strip()
        if not name:
            messagebox.showerror("Missing Name", "Enter the student name before capturing.")
            return

        person_id = get_next_person_id()
        self.person_id_var.set(str(person_id))
        source = normalize_video_source(self.source_var.get())
        if source:
            self.source_var.set(source)
        backend = self.backend_var.get().strip() or "default"
        source_text = source or "webcam"
        self.set_status(
            f"Capturing dataset for {name} with auto-generated ID {person_id}. Source: {source_text}",
            "Running capture",
        )
        self.run_in_thread(
            lambda: capture_faces(person_id, name, samples, camera_index, source, backend),
            "Dataset capture finished. Images saved to {result}",
        )

    def start_training(self) -> None:
        self.set_status("Training the LBPH model. This may take a moment.", "Training")
        self.run_in_thread(lambda: train_model(), "Model training finished. Saved to {result}")

    def start_recognition(self) -> None:
        try:
            camera_index = int(self.camera_var.get().strip())
        except ValueError:
            messagebox.showerror("Invalid Input", "Camera Index must be a number.")
            return

        model_path = TRAINER_DIR / "trainer.yml"
        if not model_path.exists():
            messagebox.showerror("Model Missing", "Train the model before starting recognition.")
            return

        source = normalize_video_source(self.source_var.get())
        if source:
            self.source_var.set(source)
        backend = self.backend_var.get().strip() or "default"
        source_text = source or "webcam"
        self.set_status(f"Recognition started from {source_text}. Press q in the OpenCV window to stop monitoring.", "Monitoring")

        threading.Thread(
            target=self._run_recognition_worker,
            args=(camera_index, source, backend),
            daemon=True,
        ).start()

    def _run_recognition_worker(self, camera_index: int, source: str | None, backend: str) -> None:
        try:
            run_recognition(camera_index, source, backend)
        except Exception as exc:
            error_message = str(exc)
            self.root.after(0, lambda msg=error_message: self.set_status(f"Recognition error: {msg}", "Error"))
            self.root.after(0, lambda msg=error_message: messagebox.showerror("Recognition Failed", msg))
            return

        self.root.after(0, lambda: self.set_status("Recognition stopped. Dashboard remains ready.", "Idle"))
        self.root.after(0, self.refresh_data)

    def refresh_data(self) -> None:
        students = list_registered_students()
        attendance = read_attendance_records()

        self._replace_tree_rows(self.students_tree, students)
        self._replace_tree_rows(
            self.attendance_tree,
            [(row["person_id"], row["name"], row["timestamp"]) for row in reversed(attendance)],
        )

        self.student_count_label.configure(text=str(len(students)))
        self.image_count_label.configure(text=str(sum(image_count for _, _, image_count in students)))
        self.attendance_count_label.configure(text=str(len(attendance)))
        model_ready = (TRAINER_DIR / "trainer.yml").exists()
        labels_ready = LABELS_FILE.exists()
        
        status_text = "Ready" if model_ready and labels_ready else "Pending"
        color = "#10b981" if status_text == "Ready" else "#f59e0b"
        self.model_status_label.configure(text=status_text, text_color=color)

        self.person_id_var.set(str(get_next_person_id()))

    def _replace_tree_rows(self, tree: ttk.Treeview, rows) -> None:
        for item in tree.get_children():
            tree.delete(item)
        for row in rows:
            tree.insert("", "end", values=row)


def main() -> None:
    root = ctk.CTk()
    dashboard = SmartAttendanceDashboard(root)
    if not ATTENDANCE_FILE.exists():
        dashboard.refresh_data()
    root.mainloop()


if __name__ == "__main__":
    main()
