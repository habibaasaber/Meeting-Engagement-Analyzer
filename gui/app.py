import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import logging
from prediction.predict import Predictor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EngagementApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Meeting Engagement Analyzer")
        self.root.geometry("1200x850")
        
        # Initialize predictor
        try:
            self.predictor = Predictor()
        except Exception as e:
            messagebox.showwarning("Warning", f"Models not found or error loading models: {e}\nPlease train models first.")
            self.predictor = None

        self.df = None
        self.results_df = None
        self.filter_low = False
        
        self.setup_ui()

    def setup_ui(self):
        # Main Layout
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header / Controls
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        
        ttk.Button(control_frame, text="Upload Meeting Report", command=self.load_file).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Export Results to CSV", command=self.export_csv).pack(side=tk.LEFT, padx=5)
        
        self.filter_btn = ttk.Button(control_frame, text="Show Low Engagement Students", command=self.toggle_filter)
        self.filter_btn.pack(side=tk.LEFT, padx=5)
        
        self.status_label = ttk.Label(control_frame, text="Status: Ready", font=("Segoe UI", 9, "italic"))
        self.status_label.pack(side=tk.RIGHT, padx=5)
        
        # Content Frame (Table and Charts)
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Table (Left Side)
        table_container = ttk.Frame(content_frame)
        table_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        table_label_frame = ttk.LabelFrame(table_container, text="Student Engagement Overview", padding="5")
        table_label_frame.pack(fill=tk.BOTH, expand=True)
        
        columns = ("Name", "Duration", "Chat", "Score", "Level", "Cluster")
        self.tree = ttk.Treeview(table_label_frame, columns=columns, show='headings', selectmode="browse")
        
        self.tree.heading("Name", text="Student Name")
        self.tree.heading("Duration", text="Time (min)")
        self.tree.heading("Chat", text="Chat")
        self.tree.heading("Score", text="Eng. Score")
        self.tree.heading("Level", text="Level")
        self.tree.heading("Cluster", text="Style")
        
        # Adjust column widths
        self.tree.column("Name", width=150)
        self.tree.column("Duration", width=80)
        self.tree.column("Chat", width=60)
        self.tree.column("Score", width=80)
        self.tree.column("Level", width=100)
        self.tree.column("Cluster", width=100)
        
        scrollbar = ttk.Scrollbar(table_label_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Define tags for colors
        self.tree.tag_configure('High', background='#d4edda')   # Soft Green
        self.tree.tag_configure('Medium', background='#fff3cd') # Soft Yellow
        self.tree.tag_configure('Low', background='#f8d7da')    # Soft Red
        
        # Bind double click
        self.tree.bind("<Double-1>", self.show_student_details)
        
        # Charts (Right Side)
        charts_container = ttk.Frame(content_frame)
        charts_container.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False, padx=(10, 0))
        
        self.pie_frame = ttk.LabelFrame(charts_container, text="Engagement Distribution", padding="5")
        self.pie_frame.pack(fill=tk.X, expand=False)
        
        self.bar_frame = ttk.LabelFrame(charts_container, text="Top Student Participation", padding="5")
        self.bar_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV/Excel files", "*.csv *.xlsx")])
        if not file_path:
            return
            
        try:
            logger.info(f"Loading file: {file_path}")
            if file_path.endswith('.csv'):
                self.df = pd.read_csv(file_path)
            else:
                self.df = pd.read_excel(file_path)
                
            if self.predictor:
                self.results_df = self.predictor.predict_all(self.df)
                self.update_table()
                self.update_charts()
                self.status_label.config(text=f"Loaded: {os.path.basename(file_path)}")
            else:
                messagebox.showerror("Error", "Predictor not initialized. Train models first.")
        except Exception as e:
            logger.error(f"Failed to load file: {e}")
            messagebox.showerror("Error", f"Failed to load file: {e}")

    def toggle_filter(self):
        if self.results_df is None:
            return
        self.filter_low = not self.filter_low
        self.filter_btn.config(text="Show All Students" if self.filter_low else "Show Low Engagement Students")
        self.update_table()

    def update_table(self):
        # Clear existing
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        if self.results_df is None:
            return
            
        data_to_show = self.results_df
        if self.filter_low:
            data_to_show = self.results_df[self.results_df['Engagement_Level'] == 'Low']
            
        # Add data
        for _, row in data_to_show.iterrows():
            level = row['Engagement_Level']
            # Safely get columns
            duration = round(row.get('Duration (minutes)', 0), 1)
            chat = row.get('Chat Messages Count', 0)
            score = round(row.get('engagement_score', 0), 1)
            cluster = row.get('Participation_Cluster', 'N/A')
            
            self.tree.insert("", tk.END, values=(
                row['Name'], 
                duration, 
                chat,
                score,
                level,
                cluster
            ), tags=(level,))

    def update_charts(self):
        # Clear existing charts
        for frame in [self.pie_frame, self.bar_frame]:
            for widget in frame.winfo_children():
                widget.destroy()
            
        if self.results_df is None:
            return
            
        # 1. Pie Chart
        counts = self.results_df['Engagement_Level'].value_counts()
        labels = counts.index
        sizes = counts.values
        colors = {'High': '#28a745', 'Medium': '#ffc107', 'Low': '#dc3545'}
        pie_colors = [colors.get(l, '#6c757d') for l in labels]
        
        fig1, ax1 = plt.subplots(figsize=(4, 3))
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', colors=pie_colors, startangle=140)
        ax1.set_title("Engagement Distribution")
        
        canvas1 = FigureCanvasTkAgg(fig1, master=self.pie_frame)
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 2. Bar Chart (Top 10 Participations)
        top_students = self.results_df.sort_values(by='participation_score', ascending=False).head(10)
        
        fig2, ax2 = plt.subplots(figsize=(4, 4))
        ax2.barh(top_students['Name'], top_students['participation_score'], color='#007bff')
        ax2.set_xlabel("Participation Score")
        ax2.set_title("Top 10 Active Students")
        plt.tight_layout()
        
        canvas2 = FigureCanvasTkAgg(fig2, master=self.bar_frame)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def show_student_details(self, event):
        item = self.tree.selection()
        if not item:
            return
            
        student_name = self.tree.item(item, "values")[0]
        student_data = self.results_df[self.results_df['Name'] == student_name].iloc[0]
        
        # Create popup
        details_win = tk.Toplevel(self.root)
        details_win.title(f"Student Profile: {student_name}")
        details_win.geometry("500x550")
        details_win.resizable(False, False)
        
        frame = ttk.Frame(details_win, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Header
        ttk.Label(frame, text=student_name, font=("Segoe UI", 16, "bold")).pack(pady=(0, 10))
        
        # Main Metrics
        metrics_frame = ttk.LabelFrame(frame, text="Engagement Metrics", padding="10")
        metrics_frame.pack(fill=tk.X, pady=5)
        
        rows = [
            ("Engagement Score:", f"{student_data.get('engagement_score', 0):.1f} / 100"),
            ("Engagement Level:", student_data.get('Engagement_Level', 'N/A')),
            ("Participation Style:", student_data.get('Participation_Cluster', 'N/A')),
            ("Participation Score:", f"{student_data.get('participation_score', 0):.1f}"),
            ("Duration Ratio:", f"{student_data.get('duration_ratio', 0)*100:.1f}%")
        ]
        
        for label, val in rows:
            row_frame = ttk.Frame(metrics_frame)
            row_frame.pack(fill=tk.X, pady=2)
            ttk.Label(row_frame, text=label, width=20).pack(side=tk.LEFT)
            ttk.Label(row_frame, text=val, font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT)
            
        # Similar Students
        similar_frame = ttk.LabelFrame(frame, text="Similar Students (KNN Analysis)", padding="10")
        similar_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        similar_list = self.predictor.get_similar_students(self.results_df, student_name)
        
        if similar_list:
            for s in similar_list:
                s_frame = ttk.Frame(similar_frame)
                s_frame.pack(fill=tk.X, pady=1)
                ttk.Label(s_frame, text=s['Name'], width=20).pack(side=tk.LEFT)
                ttk.Label(s_frame, text=f"{s['Engagement']} ({s['Score']})").pack(side=tk.LEFT)
        else:
            ttk.Label(similar_frame, text="No similar students found.").pack()
            
        ttk.Button(frame, text="Close", command=details_win.destroy).pack(side=tk.BOTTOM, pady=10)

    def export_csv(self):
        if self.results_df is None:
            messagebox.showwarning("Warning", "No data to export.")
            return
            
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.results_df.to_csv(file_path, index=False)
            messagebox.showinfo("Success", f"Results saved to {file_path}")

def run_app():
    root = tk.Tk()
    # Apply a modern theme if available
    style = ttk.Style()
    if "vista" in style.theme_names():
        style.theme_use("vista")
    
    app = EngagementApp(root)
    root.mainloop()

if __name__ == "__main__":
    run_app()

