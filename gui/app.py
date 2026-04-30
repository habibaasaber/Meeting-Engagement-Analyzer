import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
from prediction.predict import Predictor

class EngagementApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Meeting Engagement Analyzer")
        self.root.geometry("10000x800")
        
        # Initialize predictor
        try:
            self.predictor = Predictor()
        except Exception as e:
            messagebox.showwarning("Warning", f"Models not found or error loading models: {e}\nPlease train models first.")
            self.predictor = None

        self.df = None
        self.results_df = None
        
        self.setup_ui()

    def setup_ui(self):
        # Main Layout
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header / Controls
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.pack(side=tk.TOP, fill=tk.X)
        
        ttk.Button(control_frame, text="Upload Meeting Report", command=self.load_file).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Export Results to CSV", command=self.export_csv).pack(side=tk.LEFT, padx=5)
        
        self.status_label = ttk.Label(control_frame, text="No file loaded")
        self.status_label.pack(side=tk.RIGHT, padx=5)
        
        # Content Frame (Table and Chart)
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Table (Left Side)
        table_frame = ttk.LabelFrame(content_frame, text="Student Engagement Data", padding="5")
        table_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.tree = ttk.Treeview(table_frame, columns=("Name", "Duration", "Chat", "Engagement", "Binary", "Cluster"), show='headings')
        self.tree.heading("Name", text="Student Name")
        self.tree.heading("Duration", text="Duration (min)")
        self.tree.heading("Chat", text="Chat Count")
        self.tree.heading("Engagement", text="Engagement Level")
        self.tree.heading("Binary", text="Status")
        self.tree.heading("Cluster", text="Style")
        
        # Scrollbar for table
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Define tags for colors
        self.tree.tag_configure('High', background='#d4edda')   # Green
        self.tree.tag_configure('Medium', background='#fff3cd') # Yellow
        self.tree.tag_configure('Low', background='#f8d7da')    # Red
        
        # Chart (Right Side)
        self.chart_frame = ttk.LabelFrame(content_frame, text="Engagement Distribution", padding="5")
        self.chart_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

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
                self.update_chart()
                self.status_label.config(text=f"Loaded: {os.path.basename(file_path)}")
            else:
                messagebox.showerror("Error", "Predictor not initialized. Train models first.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {e}")

    def update_table(self):
        # Clear existing
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        # Add data
        for _, row in self.results_df.iterrows():
            level = row['Engagement_Level']
            self.tree.insert("", tk.END, values=(
                row['Name'], 
                row['Duration (minutes)'], 
                row['Chat Messages Count'],
                level,
                row['Binary_Engagement'],
                row['Participation_Cluster']
            ), tags=(level,))

    def update_chart(self):
        # Clear existing chart
        for widget in self.chart_frame.winfo_children():
            widget.destroy()
            
        if self.results_df is None:
            return
            
        counts = self.results_df['Engagement_Level'].value_counts()
        labels = counts.index
        sizes = counts.values
        colors = ['#d4edda' if l == 'High' else '#fff3cd' if l == 'Medium' else '#f8d7da' for l in labels]
        
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140)
        ax.set_title("Engagement Distribution")
        
        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

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
    app = EngagementApp(root)
    root.mainloop()

if __name__ == "__main__":
    run_app()
