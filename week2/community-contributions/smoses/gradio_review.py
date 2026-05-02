import gradio as gr
import json
import os
from pathlib import Path

def load_json_file(filepath):
    """Load JSON file and return its contents"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}

def save_json_file(filepath, data):
    """Save data back to JSON file"""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return "✅ Saved successfully!"
    except Exception as e:
        return f"❌ Error saving: {str(e)}"

def calculate_total(acc, use, comp, conc, clar):
    """Calculate total score from individual scores"""
    scores = [acc, use, comp, conc, clar]
    # Only sum if all scores are provided (not None and > 0)
    if all(s is not None and s > 0 for s in scores):
        return sum(scores)
    return 0

def create_model_evaluation_ui(model_name, model_data, data_store):
    """Create evaluation UI components for a single model"""
    with gr.Group():
        gr.Markdown(f"### 🤖 Model: `{model_name}`")
        
        # Display model's answer
        gr.Markdown("**Answer:**")
        gr.Markdown(model_data.get("answer", "No answer provided"))
        
        # Display LLM evaluations if they exist
        if "evals" in model_data and model_data["evals"]:
            gr.Markdown("---")
            gr.Markdown("**🔍 LLM Evaluations:**")
            for eval_model, eval_data in model_data["evals"].items():
                with gr.Accordion(f"Evaluation by {eval_model}", open=False):
                    eval_text = f"""
- **Accuracy:** {eval_data.get('accuracy', 'N/A')}
- **Usefulness:** {eval_data.get('usefulness', 'N/A')}
- **Completeness:** {eval_data.get('completeness', 'N/A')}
- **Conciseness:** {eval_data.get('conciseness', 'N/A')}
- **Clarity:** {eval_data.get('clarity', 'N/A')}
- **Total:** {eval_data.get('total', 'N/A')}

**Comments:** {eval_data.get('comments', 'No comments')}
"""
                    gr.Markdown(eval_text)
        
        # Human evaluation section
        gr.Markdown("---")
        gr.Markdown("**👤 Your Evaluation:**")
        
        # Get existing human eval if it exists
        human_eval = model_data.get("human_eval", {})
        
        with gr.Row():
            accuracy = gr.Slider(1, 5, step=1, label="Accuracy", 
                                value=human_eval.get("accuracy", 3))
            usefulness = gr.Slider(1, 5, step=1, label="Usefulness", 
                                  value=human_eval.get("usefulness", 3))
            completeness = gr.Slider(1, 5, step=1, label="Completeness", 
                                    value=human_eval.get("completeness", 3))
        
        with gr.Row():
            conciseness = gr.Slider(1, 5, step=1, label="Conciseness", 
                                   value=human_eval.get("conciseness", 3))
            clarity = gr.Slider(1, 5, step=1, label="Clarity", 
                               value=human_eval.get("clarity", 3))
            total = gr.Number(label="Total Score", value=human_eval.get("total", 15), 
                            interactive=False)
        
        comments = gr.Textbox(label="Comments", lines=3, 
                             value=human_eval.get("comments", ""),
                             placeholder="Enter your evaluation comments here...")
        
        # Update total when any score changes
        for score_slider in [accuracy, usefulness, completeness, conciseness, clarity]:
            score_slider.change(
                fn=calculate_total,
                inputs=[accuracy, usefulness, completeness, conciseness, clarity],
                outputs=total
            )
        
        # Store references to components for this model
        return {
            "model_name": model_name,
            "accuracy": accuracy,
            "usefulness": usefulness,
            "completeness": completeness,
            "conciseness": conciseness,
            "clarity": clarity,
            "total": total,
            "comments": comments
        }

def create_ui_for_file(filepath):
    """Create the complete UI for a loaded JSON file"""
    data = load_json_file(filepath)
    
    if "error" in data:
        return [gr.Markdown(f"# ❌ Error loading file: {data['error']}")], data, []
    
    components = []
    model_evals = []
    
    # Display question and references
    components.append(gr.Markdown(f"# 📋 Medical Question"))
    components.append(gr.Markdown(f"**Question:** {data.get('question', 'No question found')}"))
    components.append(gr.Markdown(f"## 📚 References\n{data.get('references', 'No references found')}"))
    components.append(gr.Markdown("---"))
    components.append(gr.Markdown("# 🎯 Model Answers & Evaluations"))
    
    # Create evaluation UI for each model
    if "models" in data:
        for model_name, model_data in data["models"].items():
            eval_components = create_model_evaluation_ui(model_name, model_data, data)
            model_evals.append(eval_components)
            components.append(gr.Markdown("---"))
    
    return components, data, model_evals

def save_evaluations(filepath, original_data, *eval_values):
    """Save human evaluations back to the JSON file"""
    # Reconstruct evaluations from flat list of values
    # eval_values comes in groups of 7: acc, use, comp, conc, clar, total, comments
    models = original_data.get("models", {})
    model_names = list(models.keys())
    
    idx = 0
    for model_name in model_names:
        if idx + 6 < len(eval_values):
            models[model_name]["human_eval"] = {
                "accuracy": int(eval_values[idx]),
                "usefulness": int(eval_values[idx + 1]),
                "completeness": int(eval_values[idx + 2]),
                "conciseness": int(eval_values[idx + 3]),
                "clarity": int(eval_values[idx + 4]),
                "total": int(eval_values[idx + 5]),
                "comments": eval_values[idx + 6]
            }
            idx += 7
    
    return save_json_file(filepath, original_data)

def get_json_files(directory):
    """Get list of JSON files in directory"""
    try:
        json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
        return json_files if json_files else ["No JSON files found"]
    except:
        return ["Directory not found"]

# Main Gradio App
with gr.Blocks(title="Medical Q&A Evaluation Tool", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🏥 Medical Q&A Model Evaluation Tool")
    gr.Markdown("Load a JSON file to evaluate model responses to medical questions.")
    
    # File selection
    with gr.Row():
        directory = gr.Textbox(label="Directory Path", value=".", placeholder="Enter directory path")
        refresh_btn = gr.Button("🔄 Refresh Files")
    
    file_dropdown = gr.Dropdown(label="Select JSON File", choices=get_json_files("."))
    load_btn = gr.Button("📂 Load File", variant="primary")
    
    # Placeholder for dynamic content
    current_file = gr.State(None)
    current_data = gr.State({})
    content_column = gr.Column(visible=False)
    
    # Save section
    with gr.Row():
        save_btn = gr.Button("💾 Save Evaluations", variant="primary", visible=False)
        save_status = gr.Markdown("", visible=False)
    
    # Dynamic content container
    dynamic_components = gr.State([])
    model_eval_components = gr.State([])
    
    def load_file_handler(directory, filename):
        if filename == "No JSON files found" or filename == "Directory not found":
            return {
                content_column: gr.Column(visible=False),
                save_btn: gr.Button(visible=False),
                save_status: gr.Markdown(visible=False)
            }
        
        filepath = os.path.join(directory, filename)
        return {
            current_file: filepath,
            content_column: gr.Column(visible=True),
            save_btn: gr.Button(visible=True),
            save_status: gr.Markdown(visible=True)
        }
    
    def refresh_files_handler(directory):
        return gr.Dropdown(choices=get_json_files(directory))
    
    # Event handlers
    refresh_btn.click(
        fn=refresh_files_handler,
        inputs=[directory],
        outputs=[file_dropdown]
    )
    
    load_btn.click(
        fn=load_file_handler,
        inputs=[directory, file_dropdown],
        outputs=[current_file, content_column, save_btn, save_status]
    )
    
    with content_column:
        # This will be populated dynamically
        question_display = gr.Markdown("## Question will appear here")
        references_display = gr.Markdown("## References will appear here")
        models_display = gr.Column()
    
    # Create a complex update function that rebuilds the UI
    def update_display(filepath):
        if not filepath:
            return {}
        
        data = load_json_file(filepath)
        if "error" in data:
            return {
                question_display: gr.Markdown(f"# ❌ Error: {data['error']}"),
                references_display: gr.Markdown(""),
                current_data: data
            }
        
        updates = {
            question_display: gr.Markdown(f"# 📋 Question\n\n**{data.get('question', 'No question')}**"),
            references_display: gr.Markdown(f"## 📚 References\n{data.get('references', 'No references')}"),
            current_data: data
        }
        return updates
    
    current_file.change(
        fn=update_display,
        inputs=[current_file],
        outputs=[question_display, references_display, current_data]
    )

if __name__ == "__main__":
    app.launch()
