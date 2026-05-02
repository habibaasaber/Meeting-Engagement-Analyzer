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
        self.root.title("Meeting Engagement Analyzer - AI Dashboard")
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
        # Apply Notebook for Tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # TAB 1: Analyzer Dashboard
        self.analyzer_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.analyzer_frame, text=" 📊 Student Analyzer ")
        self.setup_analyzer_tab()

        # TAB 2: Model Performance
        self.performance_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.performance_frame, text=" ⚙️ Model Performance ")
        self.setup_performance_tab()

    def setup_analyzer_tab(self):
        # Header / Controls & Search
        control_frame = ttk.LabelFrame(self.analyzer_frame, text="Controls & Search", padding="10")
        control_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        
        # File Actions
        file_frame = ttk.Frame(control_frame)
        file_frame.pack(side=tk.LEFT, padx=5)
        ttk.Button(file_frame, text="Upload Report", command=self.load_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(file_frame, text="Export CSV", command=self.export_csv).pack(side=tk.LEFT, padx=2)
        
        # Search Bar
        search_frame = ttk.Frame(control_frame)
        search_frame.pack(side=tk.LEFT, padx=20)
        ttk.Label(search_frame, text="Search Student:").pack(side=tk.LEFT, padx=2)
        self.search_var = tk.StringVar()
        self.search_var.trace_add("write", lambda *args: self.update_table()) # Real-time search
        self.search_entry = ttk.Entry(search_frame, textvariable=self.search_var, width=25)
        self.search_entry.pack(side=tk.LEFT, padx=2)
        ttk.Button(search_frame, text="Clear", command=lambda: self.search_var.set("")).pack(side=tk.LEFT, padx=2)

        # Filters
        filter_frame = ttk.Frame(control_frame)
        filter_frame.pack(side=tk.LEFT, padx=5)
        self.filter_btn = ttk.Button(filter_frame, text="Show Low Engagement", command=self.toggle_filter)
        self.filter_btn.pack(side=tk.LEFT, padx=2)
        
        self.status_label = ttk.Label(control_frame, text="Status: Ready", font=("Segoe UI", 9, "italic"))
        self.status_label.pack(side=tk.RIGHT, padx=5)
        
        # Content Frame (Table and Charts)
        content_frame = ttk.Frame(self.analyzer_frame)
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
        
        self.tree.tag_configure('High', background='#d4edda')
        self.tree.tag_configure('Medium', background='#fff3cd')
        self.tree.tag_configure('Low', background='#f8d7da')
        self.tree.bind("<Double-1>", self.show_student_details)
        
        # Charts (Right Side)
        charts_container = ttk.Frame(content_frame)
        charts_container.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        self.pie_frame = ttk.LabelFrame(charts_container, text="Engagement Distribution", padding="5")
        self.pie_frame.pack(fill=tk.BOTH, expand=True)
        
        self.cluster_frame = ttk.LabelFrame(charts_container, text="Participation Style (K-Means)", padding="5")
        self.cluster_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.bar_frame = ttk.LabelFrame(charts_container, text="Top Student Participation", padding="5")
        self.bar_frame.pack(fill=tk.BOTH, expand=True)

    def setup_performance_tab(self):
        # Horizontal Split: Top for Report, Bottom for Chart
        self.performance_frame.columnconfigure(0, weight=1)
        self.performance_frame.rowconfigure(0, weight=1) # Report row
        self.performance_frame.rowconfigure(1, weight=2) # Chart row (Takes more space)

        # TOP: Statistics Text
        stats_frame = ttk.LabelFrame(self.performance_frame, text="Evaluation Report", padding="10")
        stats_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 5))
        
        self.stats_text = tk.Text(stats_frame, wrap=tk.WORD, font=("Consolas", 10), height=10)
        stats_scroll = ttk.Scrollbar(stats_frame, command=self.stats_text.yview)
        self.stats_text.configure(yscrollcommand=stats_scroll.set)
        
        self.stats_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        stats_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # BOTTOM: Comparison Chart
        chart_frame = ttk.LabelFrame(self.performance_frame, text="Model Comparison (All Algorithms)", padding="10")
        chart_frame.grid(row=1, column=0, sticky="nsew")
        
        self.comparison_canvas_frame = ttk.Frame(chart_frame)
        self.comparison_canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Button(stats_frame, text="Refresh Performance Data", command=self.load_performance_data).pack(side=tk.RIGHT, padx=5)
        
        self.load_performance_data()

    def load_performance_data(self):
        report_path = 'models/evaluation_report.txt'
        self.stats_text.delete('1.0', tk.END)
        if os.path.exists(report_path):
            with open(report_path, 'r', encoding='utf-8') as f:
                self.stats_text.insert(tk.END, f.read())
        else:
            self.stats_text.insert(tk.END, "Performance report not found. Please train models.")

        for widget in self.comparison_canvas_frame.winfo_children():
            widget.destroy()

        plot_path = 'models/model_comparison.png'
        if os.path.exists(plot_path):
            try:
                img = tk.PhotoImage(file=plot_path)
                img_label = tk.Label(self.comparison_canvas_frame, image=img)
                img_label.image = img
                img_label.pack(expand=True)
            except Exception as e:
                ttk.Label(self.comparison_canvas_frame, text=f"Could not load chart image: {e}").pack()
        else:
            ttk.Label(self.comparison_canvas_frame, text="Comparison chart not found.").pack()

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV/Excel files", "*.csv *.xlsx")])
        if not file_path:
            return
            
        try:
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
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        if self.results_df is None:
            return
            
        data_to_show = self.results_df
        
        # Apply Search Filter
        query = self.search_var.get().strip().lower()
        if query:
            data_to_show = data_to_show[data_to_show['Name'].str.lower().str.contains(query)]

        # Apply Low Engagement Filter
        if self.filter_low:
            data_to_show = data_to_show[data_to_show['Engagement_Level'] == 'Low']
            
        for _, row in data_to_show.iterrows():
            level = row['Engagement_Level']
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
        for frame in [self.pie_frame, self.bar_frame, self.cluster_frame]:
            for widget in frame.winfo_children():
                widget.destroy()
            
        if self.results_df is None:
            return
            
        # 1. Engagement Pie Chart
        counts = self.results_df['Engagement_Level'].value_counts()
        labels = counts.index
        sizes = counts.values
        colors = {'High': '#28a745', 'Medium': '#ffc107', 'Low': '#dc3545'}
        pie_colors = [colors.get(l, '#6c757d') for l in labels]
        
        fig1, ax1 = plt.subplots(figsize=(4, 2.2))
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', colors=pie_colors, startangle=140, textprops={'fontsize': 9})
        ax1.set_title("Engagement Levels", fontsize=10, fontweight='bold')
        plt.tight_layout()
        
        canvas1 = FigureCanvasTkAgg(fig1, master=self.pie_frame)
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 2. Cluster Bar Chart (K-Means Result)
        cluster_counts = self.results_df['Participation_Cluster'].value_counts()
        
        fig3, ax3 = plt.subplots(figsize=(4, 2.2))
        cluster_counts.plot(kind='bar', ax=ax3, color='#9467bd')
        ax3.set_title("K-Means Participation Styles", fontsize=10, fontweight='bold')
        ax3.set_ylabel("Student Count", fontsize=8)
        ax3.tick_params(axis='x', labelsize=8)
        plt.tight_layout()
        
        canvas3 = FigureCanvasTkAgg(fig3, master=self.cluster_frame)
        canvas3.draw()
        canvas3.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 3. Participation Bar Chart
        top_students = self.results_df.sort_values(by='participation_score', ascending=False).head(10)
        
        fig2, ax2 = plt.subplots(figsize=(4, 3.5))
        ax2.barh(top_students['Name'], top_students['participation_score'], color='#007bff')
        ax2.set_xlabel("Participation Score", fontsize=9)
        ax2.set_title("Top 10 Active Students", fontsize=10, fontweight='bold')
        ax2.tick_params(axis='y', labelsize=9)
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
        
        details_win = tk.Toplevel(self.root)
        details_win.title(f"Student Profile: {student_name}")
        details_win.geometry("500x550")
        details_win.resizable(False, False)
        
        frame = ttk.Frame(details_win, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(frame, text=student_name, font=("Segoe UI", 16, "bold")).pack(pady=(0, 10))
        
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
    style = ttk.Style()
    if "vista" in style.theme_names():
        style.theme_use("vista")
    
    app = EngagementApp(root)
    root.mainloop()

if __name__ == "__main__":
    run_app()
