import matplotlib.pyplot as plt
import numpy as np
import time
import tkinter as tk
from tkinter import scrolledtext, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class SessionReporter:
    def __init__(self):
        pass

    def generate_report(self, scoring_engine, timeline):
        print("\n[Report] Generating post-session graphical report...")
        
        averages = scoring_engine.session_averages
        exp_engine = scoring_engine.explanation_engine
        
        if not averages["attention"]:
            print("[Report] No data collected during session.")
            return

        frames = len(averages["attention"])
        time_axis = np.linspace(0, frames / 10.0, frames)

        inf_conf_history = averages.get("inference_confidence", [1.0] * frames)
        if not inf_conf_history:
            inf_conf_history = [1.0] * frames
        elif len(inf_conf_history) < frames:
            inf_conf_history += [1.0] * (frames - len(inf_conf_history))
            
        inf_conf_arr = np.array(inf_conf_history)
        attn_arr = np.array(averages["attention"])

        gaze_deviation_ratio = (exp_engine.gaze_left_count + exp_engine.gaze_right_count) / max(1, exp_engine.total_frames)
        session_conf = np.mean(averages["confidence"])
        session_attn = np.mean(averages["attention"])
        session_foc = np.mean(averages["focus"])
        rolling_str = np.mean(averages["stress"])
        final_integrity = scoring_engine.integrity_score
        
        final_session_score, contributors = scoring_engine.trace.calculate_session_trace(
            session_confidence=session_conf,
            session_attention=session_attn,
            session_focus=session_foc,
            rolling_stress=rolling_str,
            integrity_score=final_integrity,
            gaze_deviation_ratio=gaze_deviation_ratio
        )
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 6.5))
        fig.suptitle("Interview Behavioral Analysis Curves", fontsize=14, fontweight='bold', color='#1A202C')

        error_margin = (1.0 - inf_conf_arr) * 0.4
        lower_bound = np.clip(attn_arr - error_margin, 0.0, 1.0)
        upper_bound = np.clip(attn_arr + error_margin, 0.0, 1.0)
        
        ax1.plot(time_axis, averages["attention"], label="Attention (Score)", color="#3182CE", linewidth=2.0)
        ax1.fill_between(time_axis, lower_bound, upper_bound, color="#3182CE", alpha=0.15, label="Confidence of Inference Interval")
        ax1.plot(time_axis, averages["focus"], label="Focus (Head Pose)", color="#00B5D8", alpha=0.7, linestyle="--")
        
        ax1.set_title("Candidate Attention & Focus over Time", fontsize=11, fontweight='semibold')
        ax1.set_ylabel("Score (0.0 - 1.0)", fontsize=9)
        ax1.set_ylim(-0.05, 1.05)
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(loc="upper right", fontsize=8)

        ax2.plot(time_axis, averages["confidence"], label="Communication Confidence", color="#38A169", linewidth=2.0)
        ax2.plot(time_axis, averages["stress"], label="Stress Indicators", color="#E53E3E", alpha=0.8)
        
        ax2.set_title("Communication Confidence vs. Stress Indicators", fontsize=11, fontweight='semibold')
        ax2.set_ylabel("Score (0.0 - 1.0)", fontsize=9)
        ax2.set_ylim(-0.05, 1.05)
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(loc="upper right", fontsize=8)

        ax3.set_title("Behavioral & Integrity Events Timeline", fontsize=11, fontweight='semibold')
        ax3.set_xlabel("Time (Seconds)", fontsize=9)
        ax3.set_yticks([])
        ax3.set_ylim(0, 1.2)
        ax3.set_xlim(-5, time_axis[-1] + 5)
        ax3.grid(True, axis='x', linestyle="--", alpha=0.5)

        colors = {"Behavior": "#DD6B20", "Integrity": "#E53E3E"}
        
        for event in timeline.events:
            t = event["timestamp"]
            e_type = event["type"]
            name = event["name"]
            
            c = colors.get(e_type, "#3182CE")
            ax3.axvline(x=t, color=c, linestyle="-.", alpha=0.8, linewidth=1.5)
            y_pos = 0.5 if t % 10 < 5 else 0.8
            ax3.text(t, y_pos, f" {name} ({event['time_str']})", rotation=0, 
                     verticalalignment='center', color=c, fontsize=7, fontweight='bold',
                     bbox=dict(facecolor='white', alpha=0.9, edgecolor=c, boxstyle='round,pad=0.2'))

        plt.tight_layout()

        report_text = self._build_report_string(final_session_score, final_integrity, contributors, exp_engine, timeline)

        self._print_console_report(final_session_score, final_integrity, contributors, exp_engine, timeline)
        self._write_markdown_report(final_session_score, final_integrity, contributors, exp_engine, timeline)
        self._launch_tkinter_report(fig, report_text)

    def _build_report_string(self, score, integrity, contributors, exp_engine, timeline):
        divider = "=" * 80 + "\n"
        sub_divider = "-" * 80 + "\n"
        
        r_str = divider
        r_str += "                    AI INTERVIEW EXPLAINABILITY & SESSION EVALUATION\n"
        r_str += divider + "\n"
        r_str += f"  • Final Session Score: {score}/100\n"
        r_str += f"  • Overall Integrity Level: {integrity:.1f}%\n"
        r_str += f"  • Average Inference Confidence: {int(np.mean(exp_engine.unstable_landmarks_frames / max(1, exp_engine.total_frames)) * 100)}%\n\n"
        
        r_str += "1. SCORING TRACEBACK ANALYSIS\n"
        r_str += sub_divider
        for factor, val in contributors.items():
            sign = "+" if val >= 0 else ""
            r_str += f"   - {factor:32s} : {sign}{val}\n"
        r_str += f"   - {'Final Session Score':32s} : {score}\n\n"
        
        r_str += "2. BEHAVIORAL EVIDENCE & EXPLANATIONS\n"
        r_str += sub_divider
        
        attn_exp = exp_engine.get_attention_explanation(score, 1.0)
        r_str += f"   [Attention Score Explanation]\n"
        for line in attn_exp["explanation"]:
            r_str += f"     • {line}\n"
        r_str += "\n"
            
        conf_exp = exp_engine.get_confidence_explanation(score, 1.0)
        r_str += f"   [Confidence Score Explanation]\n"
        for line in conf_exp["explanation"]:
            r_str += f"     • {line}\n"
        r_str += "\n"
            
        integ_exp = exp_engine.get_integrity_explanation(integrity, 1.0)
        r_str += f"   [Integrity Explanation]\n"
        for line in integ_exp["explanation"]:
            r_str += f"     • {line}\n"
        r_str += "\n"

        r_str += "3. RECRUITER INSIGHTS SUMMARY\n"
        r_str += sub_divider
        insights = exp_engine.generate_recruiter_summary(score, integrity)
        for insight in insights:
            r_str += f"   • {insight}\n"
        r_str += "\n"

        r_str += "4. TIMELINE BEHAVIORAL EVENTS & CAUSAL REASONING\n"
        r_str += sub_divider
        if not timeline.events:
            r_str += "   No major behavioral fluctuations detected.\n"
        else:
            for event in timeline.events:
                reasons = ", ".join(event["reason_data"].get("reason", []))
                r_str += f"   • {event['time_str']} -> {event['name']} (Confidence: {event['reason_data']['confidence']})\n"
                r_str += f"     Causal Chain: {reasons}\n"
                
        r_str += "\n" + divider
        return r_str

    def _launch_tkinter_report(self, fig, report_text):
        root = tk.Tk()
        root.title("AI Interview Session Report")
        root.geometry("1100x920")
        root.config(bg="#1A202C")

        title_lbl = tk.Label(root, text="AI Interview Session Evaluation Report", 
                             fg="#FFFFFF", bg="#1A202C", font=("Helvetica", 16, "bold"), pady=10)
        title_lbl.pack(side=tk.TOP, fill=tk.X)

        plot_frame = tk.Frame(root, bg="#1A202C")
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)

        section_lbl = tk.Label(root, text="Detailed Recruiter Explainability & Insights", 
                               fg="#FFB832", bg="#1A202C", font=("Helvetica", 12, "bold"), anchor="w", padx=20, pady=5)
        section_lbl.pack(side=tk.TOP, fill=tk.X)

        text_frame = tk.Frame(root, bg="#1A202C")
        text_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=20, pady=5)

        txt_widget = scrolledtext.ScrolledText(text_frame, wrap=tk.WORD, height=12, 
                                               bg="#2D3748", fg="#E2E8F0", font=("Consolas", 10))
        txt_widget.pack(fill=tk.BOTH, expand=True)
        txt_widget.insert(tk.END, report_text)

        # User Experience / Product Review panel in Tkinter
        ux_frame = tk.Frame(root, bg="#1A202C", pady=5)
        ux_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=20, pady=2)
        
        ux_lbl = tk.Label(ux_frame, text="Rate Platform Experience (1-5):", fg="#FFB832", bg="#1A202C", font=("Helvetica", 10, "bold"))
        ux_lbl.pack(side=tk.LEFT, padx=5)
        
        rating_var = tk.StringVar(value="5")
        rating_opt = tk.OptionMenu(ux_frame, rating_var, "1", "2", "3", "4", "5")
        rating_opt.config(bg="#2D3748", fg="#E2E8F0", activebackground="#4A5568", activeforeground="#FFFFFF", highlightthickness=0, bd=0)
        rating_opt.pack(side=tk.LEFT, padx=5)
        
        comments_lbl = tk.Label(ux_frame, text="Review Comments:", fg="#FFFFFF", bg="#1A202C", font=("Helvetica", 10))
        comments_lbl.pack(side=tk.LEFT, padx=10)
        
        comments_entry = tk.Entry(ux_frame, bg="#2D3748", fg="#E2E8F0", insertbackground="#E2E8F0", font=("Helvetica", 10), width=45)
        comments_entry.pack(side=tk.LEFT, padx=5)

        # Report Finalize and Save Panel
        save_frame = tk.Frame(root, bg="#1A202C", pady=8)
        save_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=20, pady=2)

        def _on_save_report():
            edited_text = txt_widget.get("1.0", tk.END).strip()

            ux_rating = rating_var.get()
            ux_comments = comments_entry.get().strip() or "None"

            # Format markdown addition and append it to report code
            feedback_md = f"\n\n## User Experience & Product Review\n"
            feedback_md += f"- **Platform Rating**: {'★' * int(ux_rating)}{'☆' * (5 - int(ux_rating))} ({ux_rating}/5)\n"
            feedback_md += f"- **User Comments**: {ux_comments}\n"
            
            final_report_md = edited_text + feedback_md

            # Ask user for confirmation
            confirm = messagebox.askyesno(
                "Save Report",
                f"Are you sure you want to write the final changes to 'session_report.md' including your Platform Rating of {ux_rating}/5?"
            )
            if not confirm:
                return

            # Write updated report content back to file
            try:
                with open("session_report.md", "w", encoding="utf-8") as f:
                    f.write(final_report_md)
                print("[Report] Written edited markdown with UX feedback to session_report.md")
                messagebox.showinfo("Save Report", "Successfully finalized and saved the report to session_report.md!")
            except Exception as ex:
                messagebox.showerror("Save Report", f"Failed to save session_report.md: {ex}")
                return

        save_btn = tk.Button(save_frame, text="Save & Finalize Report", fg="#FFFFFF", bg="#3182CE", activebackground="#2B6CB0",
                             activeforeground="#FFFFFF", font=("Helvetica", 10, "bold"), padx=15, command=_on_save_report)
        save_btn.pack(side=tk.RIGHT, padx=5)

        def _on_close():
            plt.close(fig)
            root.destroy()

        root.protocol("WM_DELETE_WINDOW", _on_close)
        root.mainloop()

    def _print_console_report(self, score, integrity, contributors, exp_engine, timeline):
        print("\n" + "=" * 60)
        print("          AI INTERVIEW EXPLAINABILITY REPORT (RECRUITER GRADE)")
        print("=" * 60)
        print(f"Final Session Score: {score}/100")
        print(f"Overall Integrity:   {integrity:.1f}%")
        print(f"Average Inference Confidence: {int(np.mean(exp_engine.unstable_landmarks_frames / max(1, exp_engine.total_frames)) * 100)}%")
        print("-" * 60)
        print("SCORING TRACEBACK CONTRIBUTIONS:")
        for factor, val in contributors.items():
            sign = "+" if val >= 0 else ""
            print(f"  {factor:30s} : {sign}{val}")
        print("-" * 60)
        print("METRIC-SPECIFIC EXPLANATIONS:")
        
        attn_exp = exp_engine.get_attention_explanation(score, 1.0)
        print(f"\n[Attention Score: {attn_exp['score']}%]")
        for line in attn_exp["explanation"]:
            print(f"  • {line}")
            
        conf_exp = exp_engine.get_confidence_explanation(score, 1.0)
        print(f"\n[Confidence Score: {conf_exp['score']}%]")
        for line in conf_exp["explanation"]:
            print(f"  • {line}")
            
        integ_exp = exp_engine.get_integrity_explanation(integrity, 1.0)
        print(f"\n[Integrity Score: {integ_exp['score']}%]")
        for line in integ_exp["explanation"]:
            print(f"  • {line}")
            
        print("-" * 60)
        print("CHRONOLOGICAL BEHAVIORAL EVENTS TIMELINE:")
        if not timeline.events:
            print("  No major behavioral events logged during the session.")
        else:
            for event in timeline.events:
                reasons = ", ".join(event["reason_data"].get("reason", []))
                print(f"  • {event['time_str']} -> {event['name']} (Confidence: {event['reason_data']['confidence']})")
                print(f"    Causal Chain: {reasons}")
                
        print("-" * 60)
        print("RECRUITER INSIGHTS:")
        insights = exp_engine.generate_recruiter_summary(score, integrity)
        for insight in insights:
            print(f"  • {insight}")
        print("=" * 60 + "\n")

    def _write_markdown_report(self, score, integrity, contributors, exp_engine, timeline):
        report_path = "session_report.md"
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(f"# AI Interview Session Explainability Report\n\n")
                f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write(f"## Executive Summary\n")
                f.write(f"- **Final Session Score**: {score}/100\n")
                f.write(f"- **Overall Integrity**: {integrity:.1f}%\n\n")
                
                f.write(f"## Scoring Traceback Breakdown\n")
                f.write(f"Every final evaluation score is traceable to its constituent behavioral variables. The contributors sum up to the final score:\n\n")
                f.write(f"| Contributor Factor | Impact on Score |\n")
                f.write(f"| :--- | :--- |\n")
                for factor, val in contributors.items():
                    sign = "+" if val >= 0 else ""
                    f.write(f"| {factor} | {sign}{val} |\n")
                f.write(f"| **Final Session Score** | **{score}** |\n\n")
                
                f.write(f"## Structured Metric Explanations\n\n")
                
                attn_exp = exp_engine.get_attention_explanation(score, 1.0)
                f.write(f"### Attention Score\n")
                f.write(f"**Evidence & Observations**:\n")
                for line in attn_exp["explanation"]:
                    f.write(f"- {line}\n")
                f.write(f"\n")
                
                conf_exp = exp_engine.get_confidence_explanation(score, 1.0)
                f.write(f"### Confidence Score\n")
                f.write(f"**Evidence & Observations**:\n")
                for line in conf_exp["explanation"]:
                    f.write(f"- {line}\n")
                f.write(f"\n")
                
                integ_exp = exp_engine.get_integrity_explanation(integrity, 1.0)
                f.write(f"### Integrity Evaluation\n")
                f.write(f"**Evidence & Observations**:\n")
                for line in integ_exp["explanation"]:
                    f.write(f"- {line}\n")
                f.write(f"\n")
                
                f.write(f"## Behavioral & Integrity Timeline Events\n")
                if not timeline.events:
                    f.write(f"No major behavioral anomalies or warnings were logged during the session.\n\n")
                else:
                    f.write(f"| Timestamp | Event Name | Confidence | Contributing Causes |\n")
                    f.write(f"| :--- | :--- | :--- | :--- |\n")
                    for event in timeline.events:
                        reasons = ", ".join(event["reason_data"].get("reason", []))
                        f.write(f"| {event['time_str']} | {event['name']} | {event['reason_data']['confidence']} | {reasons} |\n")
                    f.write(f"\n")
                    
                f.write(f"## Recruiter Insights & Recommendations\n")
                insights = exp_engine.generate_recruiter_summary(score, integrity)
                for insight in insights:
                    f.write(f"- {insight}\n")
                    
            print(f"[Report] Markdown session report successfully written to {report_path}")
        except Exception as e:
            print(f"[Report] Warning: Could not write markdown report: {e}")
