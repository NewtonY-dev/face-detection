from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

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


class SmartAttendanceDashboard:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Smart Attendance Dashboard")
        self.root.geometry("1180x760")
        self.root.minsize(1080, 680)
        self.root.configure(bg="#f4efe6")

        self.status_var = tk.StringVar(value="Dashboard ready.")
        self.busy_var = tk.StringVar(value="Idle")
        self.name_var = tk.StringVar()
        self.person_id_var = tk.StringVar(value=str(get_next_person_id()))
        self.samples_var = tk.StringVar(value="30")
        self.camera_var = tk.StringVar(value="0")
        self.source_var = tk.StringVar()
        self.backend_var = tk.StringVar(value="default")

        self._build_styles()
        self._build_layout()
        self.refresh_data()

    def _build_styles(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("App.TFrame", background="#f4efe6")
        style.configure("Panel.TFrame", background="#fffaf2")
        style.configure("Hero.TFrame", background="#1f3a5f")
        style.configure("Title.TLabel", background="#1f3a5f", foreground="#fff8ea", font=("Segoe UI", 24, "bold"))
        style.configure("Subtitle.TLabel", background="#1f3a5f", foreground="#dce7f5", font=("Segoe UI", 11))
        style.configure("PanelTitle.TLabel", background="#fffaf2", foreground="#22324a", font=("Segoe UI", 13, "bold"))
        style.configure("Body.TLabel", background="#fffaf2", foreground="#334155", font=("Segoe UI", 10))
        style.configure("StatValue.TLabel", background="#dbe9f6", foreground="#12324a", font=("Segoe UI", 20, "bold"))
        style.configure("StatCaption.TLabel", background="#dbe9f6", foreground="#37516d", font=("Segoe UI", 10))
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"))
        style.configure("Treeview", rowheight=28, font=("Segoe UI", 10))
        style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"))

    def _build_layout(self) -> None:
        root_frame = ttk.Frame(self.root, style="App.TFrame", padding=18)
        root_frame.pack(fill="both", expand=True)

        hero = ttk.Frame(root_frame, style="Hero.TFrame", padding=20)
        hero.pack(fill="x")
        ttk.Label(hero, text="Smart Attendance and Classroom Monitoring", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            hero,
            text="Register students, collect datasets, train the recognizer, and monitor live attendance from one dashboard.",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(6, 0))

        stats_row = ttk.Frame(root_frame, style="App.TFrame")
        stats_row.pack(fill="x", pady=(16, 16))
        self.student_count_label = self._build_stat_card(stats_row, "Registered Students")
        self.image_count_label = self._build_stat_card(stats_row, "Dataset Images")
        self.attendance_count_label = self._build_stat_card(stats_row, "Attendance Records")
        self.model_status_label = self._build_stat_card(stats_row, "Model Status")

        content = ttk.Frame(root_frame, style="App.TFrame")
        content.pack(fill="both", expand=True)
        content.columnconfigure(0, weight=3)
        content.columnconfigure(1, weight=4)
        content.rowconfigure(0, weight=1)

        left_panel = ttk.Frame(content, style="Panel.TFrame", padding=16)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        right_panel = ttk.Frame(content, style="Panel.TFrame", padding=16)
        right_panel.grid(row=0, column=1, sticky="nsew")
        right_panel.rowconfigure(1, weight=1)
        right_panel.rowconfigure(3, weight=1)

        self._build_registration_panel(left_panel)
        self._build_actions_panel(left_panel)
        self._build_status_panel(left_panel)
        self._build_students_table(right_panel)
        self._build_attendance_table(right_panel)

    def _build_stat_card(self, parent: ttk.Frame, caption: str) -> ttk.Label:
        card = tk.Frame(parent, bg="#dbe9f6", bd=0, highlightthickness=0)
        card.pack(side="left", fill="x", expand=True, padx=(0, 10))
        value_label = ttk.Label(card, text="0", style="StatValue.TLabel")
        value_label.pack(anchor="w", padx=16, pady=(16, 4))
        ttk.Label(card, text=caption, style="StatCaption.TLabel").pack(anchor="w", padx=16, pady=(0, 16))
        return value_label

    def _build_registration_panel(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Student Registration", style="PanelTitle.TLabel").pack(anchor="w")
        ttk.Label(parent, text="Add a student profile and collect their face dataset.", style="Body.TLabel").pack(anchor="w", pady=(4, 12))

        form = ttk.Frame(parent, style="Panel.TFrame")
        form.pack(fill="x")
        form.columnconfigure(1, weight=1)

        self._add_form_row(form, 0, "Student ID", self.person_id_var, readonly=True)
        self._add_form_row(form, 1, "Full Name", self.name_var)
        self._add_form_row(form, 2, "Samples", self.samples_var)
        self._add_form_row(form, 3, "Camera Index", self.camera_var)
        self._add_form_row(form, 4, "Source URL", self.source_var)

        ttk.Label(form, text="Backend", style="Body.TLabel").grid(row=5, column=0, sticky="w", pady=6)
        backend_combo = ttk.Combobox(
            form,
            textvariable=self.backend_var,
            values=["default", "msmf", "dshow", "auto"],
            state="readonly",
        )
        backend_combo.grid(row=5, column=1, sticky="ew", pady=6)

        ttk.Button(parent, text="Capture Dataset", style="Accent.TButton", command=self.start_capture).pack(fill="x", pady=(16, 0))

    def _build_actions_panel(self, parent: ttk.Frame) -> None:
        section = ttk.Frame(parent, style="Panel.TFrame")
        section.pack(fill="x", pady=(20, 0))
        ttk.Label(section, text="System Actions", style="PanelTitle.TLabel").pack(anchor="w")
        ttk.Label(section, text="Train the model and launch live monitoring from here.", style="Body.TLabel").pack(anchor="w", pady=(4, 12))

        ttk.Button(section, text="Train Model", style="Accent.TButton", command=self.start_training).pack(fill="x")
        ttk.Button(section, text="Start Recognition", style="Accent.TButton", command=self.start_recognition).pack(fill="x", pady=(10, 0))
        ttk.Button(section, text="Refresh Dashboard", command=self.refresh_data).pack(fill="x", pady=(10, 0))

    def _build_status_panel(self, parent: ttk.Frame) -> None:
        section = ttk.Frame(parent, style="Panel.TFrame")
        section.pack(fill="both", expand=True, pady=(20, 0))
        ttk.Label(section, text="Status", style="PanelTitle.TLabel").pack(anchor="w")
        ttk.Label(section, textvariable=self.busy_var, style="Body.TLabel").pack(anchor="w", pady=(6, 2))
        message = tk.Text(section, height=10, wrap="word", bg="#f7f1e7", fg="#22324a", font=("Consolas", 10), relief="flat")
        message.pack(fill="both", expand=True, pady=(8, 0))
        message.insert("1.0", self.status_var.get())
        message.configure(state="disabled")
        self.status_box = message

    def _build_students_table(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Registered Students", style="PanelTitle.TLabel").grid(row=0, column=0, sticky="w")
        columns = ("person_id", "name", "images")
        tree = ttk.Treeview(parent, columns=columns, show="headings", height=8)
        for column, title, width in (("person_id", "ID", 70), ("name", "Name", 180), ("images", "Images", 90)):
            tree.heading(column, text=title)
            tree.column(column, width=width, anchor="center")
        tree.grid(row=1, column=0, sticky="nsew", pady=(10, 18))
        self.students_tree = tree

    def _build_attendance_table(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Attendance Records", style="PanelTitle.TLabel").grid(row=2, column=0, sticky="w")
        columns = ("person_id", "name", "timestamp")
        tree = ttk.Treeview(parent, columns=columns, show="headings", height=10)
        for column, title, width in (
            ("person_id", "ID", 70),
            ("name", "Name", 170),
            ("timestamp", "Timestamp", 220),
        ):
            tree.heading(column, text=title)
            tree.column(column, width=width, anchor="center")
        tree.grid(row=3, column=0, sticky="nsew", pady=(10, 0))
        self.attendance_tree = tree

    def _add_form_row(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        variable: tk.StringVar,
        readonly: bool = False,
    ) -> None:
        ttk.Label(parent, text=label, style="Body.TLabel").grid(row=row, column=0, sticky="w", pady=6, padx=(0, 10))
        state = "readonly" if readonly else "normal"
        ttk.Entry(parent, textvariable=variable, state=state).grid(row=row, column=1, sticky="ew", pady=6)

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
        self.model_status_label.configure(text="Ready" if model_ready and labels_ready else "Pending")

        self.person_id_var.set(str(get_next_person_id()))

    def _replace_tree_rows(self, tree: ttk.Treeview, rows) -> None:
        for item in tree.get_children():
            tree.delete(item)
        for row in rows:
            tree.insert("", "end", values=row)


def main() -> None:
    root = tk.Tk()
    dashboard = SmartAttendanceDashboard(root)
    if not ATTENDANCE_FILE.exists():
        dashboard.refresh_data()
    root.mainloop()


if __name__ == "__main__":
    main()
