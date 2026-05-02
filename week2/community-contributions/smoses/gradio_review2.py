import gradio as gr
import json
import os

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
    if all(s is not None for s in scores):
        return sum(scores)
    return 0

def get_json_files(directory):
    """Get list of JSON files in directory"""
    try:
        json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
        return json_files if json_files else ["No JSON files found"]
    except:
        return ["Directory not found"]

def build_model_display(data):
    """Build markdown display for all models"""
    if "error" in data or "models" not in data:
        return "No models found in data."
    
    output = "# 🎯 Model Answers & Evaluations\n\n"
    
    for model_name, model_data in data["models"].items():
        output += f"---\n\n## 🤖 Model: `{model_name}`\n\n"
        
        # Display model's answer
        output += "### 📝 Answer:\n\n"
        output += model_data.get("answer", "No answer provided") + "\n\n"
        
        # Display LLM evaluations if they exist
        if "evals" in model_data and model_data["evals"]:
            output += "### 🔍 LLM Evaluations:\n\n"
            for eval_model, eval_data in model_data["evals"].items():
                output += f"**Evaluated by: {eval_model}**\n\n"
                output += f"- **Accuracy:** {eval_data.get('accuracy', 'N/A')}\n"
                output += f"- **Usefulness:** {eval_data.get('usefulness', 'N/A')}\n"
                output += f"- **Completeness:** {eval_data.get('completeness', 'N/A')}\n"
                output += f"- **Conciseness:** {eval_data.get('conciseness', 'N/A')}\n"
                output += f"- **Clarity:** {eval_data.get('clarity', 'N/A')}\n"
                output += f"- **Total:** {eval_data.get('total', 'N/A')}\n\n"
                output += f"**Comments:** {eval_data.get('comments', 'No comments')}\n\n"
        
        # Display existing human evaluation if present
        if "human_eval" in model_data:
            output += "### 👤 Your Previous Evaluation:\n\n"
            human_eval = model_data["human_eval"]
            output += f"- **Accuracy:** {human_eval.get('accuracy', 'N/A')}\n"
            output += f"- **Usefulness:** {human_eval.get('usefulness', 'N/A')}\n"
            output += f"- **Completeness:** {human_eval.get('completeness', 'N/A')}\n"
            output += f"- **Conciseness:** {human_eval.get('conciseness', 'N/A')}\n"
            output += f"- **Clarity:** {human_eval.get('clarity', 'N/A')}\n"
            output += f"- **Total:** {human_eval.get('total', 'N/A')}\n\n"
            output += f"**Comments:** {human_eval.get('comments', 'No comments')}\n\n"
    
    return output

def create_evaluation_inputs(data):
    """Create evaluation input components for all models"""
    if "error" in data or "models" not in data:
        return [], []
    
    model_names = list(data["models"].keys())
    return model_names, [data["models"][name].get("human_eval", {}) for name in model_names]

def save_evaluations(filepath, data, model_names, *eval_values):
    """Save human evaluations back to the JSON file"""
    if not data or "models" not in data:
        return "❌ No data to save"
    
    # eval_values comes in groups of 7 per model: acc, use, comp, conc, clar, total, comments
    idx = 0
    for model_name in model_names:
        if idx + 6 < len(eval_values):
            data["models"][model_name]["human_eval"] = {
                "accuracy": int(eval_values[idx]),
                "usefulness": int(eval_values[idx + 1]),
                "completeness": int(eval_values[idx + 2]),
                "conciseness": int(eval_values[idx + 3]),
                "clarity": int(eval_values[idx + 4]),
                "total": int(eval_values[idx + 5]),
                "comments": str(eval_values[idx + 6])
            }
            idx += 7
    
    return save_json_file(filepath, data)

# Main Gradio App
with gr.Blocks(title="Medical Q&A Evaluation Tool", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🏥 Medical Q&A Model Evaluation Tool")
    gr.Markdown("Load a JSON file to evaluate model responses to medical questions.")
    
    # State variables
    current_file = gr.State(None)
    current_data = gr.State({})
    model_names_state = gr.State([])
    
    # File selection
    with gr.Row():
        directory = gr.Textbox(label="Directory Path", value=".", placeholder="Enter directory path")
        refresh_btn = gr.Button("🔄 Refresh Files")
    
    file_dropdown = gr.Dropdown(label="Select JSON File", choices=get_json_files("."))
    load_btn = gr.Button("📂 Load File", variant="primary")
    
    gr.Markdown("---")
    
    # Display sections
    question_display = gr.Markdown("", visible=False)
    references_display = gr.Markdown("", visible=False)
    models_display = gr.Markdown("", visible=False)
    
    gr.Markdown("---")
    gr.Markdown("# ✍️ Enter Your Evaluations")
    
    # Create evaluation inputs for up to 10 models (adjust as needed)
    evaluation_sections = []
    all_eval_inputs = []
    
    for i in range(10):  # Support up to 10 models
        with gr.Group(visible=False) as group:
            model_label = gr.Markdown(f"### Model {i+1}")
            
            with gr.Row():
                acc = gr.Slider(1, 5, step=1, label="Accuracy", value=3)
                use = gr.Slider(1, 5, step=1, label="Usefulness", value=3)
                comp = gr.Slider(1, 5, step=1, label="Completeness", value=3)
            
            with gr.Row():
                conc = gr.Slider(1, 5, step=1, label="Conciseness", value=3)
                clar = gr.Slider(1, 5, step=1, label="Clarity", value=3)
                total = gr.Number(label="Total Score", value=15, interactive=False)
            
            comments = gr.Textbox(label="Comments", lines=3, placeholder="Enter your evaluation comments...")
            
            # Update total when scores change
            for slider in [acc, use, comp, conc, clar]:
                slider.change(
                    fn=calculate_total,
                    inputs=[acc, use, comp, conc, clar],
                    outputs=total
                )
            
            evaluation_sections.append({
                'group': group,
                'label': model_label,
                'inputs': [acc, use, comp, conc, clar, total, comments]
            })
            all_eval_inputs.extend([acc, use, comp, conc, clar, total, comments])
    
    # Save button
    with gr.Row():
        save_btn = gr.Button("💾 Save All Evaluations", variant="primary", visible=False)
        save_status = gr.Markdown("", visible=False)
    
    def load_file_handler(directory, filename):
        """Handle file loading"""
        if filename in ["No JSON files found", "Directory not found", None]:
            return [
                None,  # current_file
                {},    # current_data
                [],    # model_names
                gr.Markdown(visible=False),  # question
                gr.Markdown(visible=False),  # references
                gr.Markdown(visible=False),  # models
                gr.Button(visible=False),    # save_btn
                gr.Markdown(visible=False),  # save_status
            ] + [gr.Group(visible=False) for _ in range(10)]  # Hide all eval sections
        
        filepath = os.path.join(directory, filename)
        data = load_json_file(filepath)
        
        if "error" in data:
            return [
                filepath,
                data,
                [],
                gr.Markdown(f"# ❌ Error: {data['error']}", visible=True),
                gr.Markdown(visible=False),
                gr.Markdown(visible=False),
                gr.Button(visible=False),
                gr.Markdown(visible=False),
            ] + [gr.Group(visible=False) for _ in range(10)]
        
        # Build displays
        question_md = f"# 📋 Question\n\n**{data.get('question', 'No question')}**"
        references_md = f"## 📚 References\n{data.get('references', 'No references')}"
        models_md = build_model_display(data)
        
        # Get model names and prepare evaluation sections
        model_names, human_evals = create_evaluation_inputs(data)
        num_models = len(model_names)
        
        # Prepare updates for evaluation sections
        eval_updates = []
        for i in range(10):
            if i < num_models:
                model_name = model_names[i]
                human_eval = human_evals[i]
                
                # Make group visible and update values
                eval_updates.append(gr.Group(visible=True))
                
            else:
                # Hide unused groups
                eval_updates.append(gr.Group(visible=False))
        
        return [
            filepath,
            data,
            model_names,
            gr.Markdown(question_md, visible=True),
            gr.Markdown(references_md, visible=True),
            gr.Markdown(models_md, visible=True),
            gr.Button(visible=True if num_models > 0 else False),
            gr.Markdown("", visible=True),
        ] + eval_updates
    
    def update_model_labels_and_values(data, model_names):
        """Update the labels and values for evaluation sections"""
        updates = []
        
        for i in range(10):
            if i < len(model_names):
                model_name = model_names[i]
                human_eval = data["models"][model_name].get("human_eval", {})
                
                # Return values for: label, acc, use, comp, conc, clar, total, comments
                updates.extend([
                    gr.Markdown(f"### 🤖 Model: `{model_name}`"),
                    human_eval.get('accuracy', 3),
                    human_eval.get('usefulness', 3),
                    human_eval.get('completeness', 3),
                    human_eval.get('conciseness', 3),
                    human_eval.get('clarity', 3),
                    human_eval.get('total', 15),
                    human_eval.get('comments', ''),
                ])
            else:
                # Empty values for hidden sections
                updates.extend([gr.Markdown(""), 3, 3, 3, 3, 3, 15, ""])
        
        return updates
    
    def refresh_files_handler(directory):
        return gr.Dropdown(choices=get_json_files(directory))
    
    # Event handlers
    refresh_btn.click(
        fn=refresh_files_handler,
        inputs=[directory],
        outputs=[file_dropdown]
    )
    
    # Collect all outputs for load button
    load_outputs = [
        current_file,
        current_data,
        model_names_state,
        question_display,
        references_display,
        models_display,
        save_btn,
        save_status,
    ] + [sec['group'] for sec in evaluation_sections]
    
    load_btn.click(
        fn=load_file_handler,
        inputs=[directory, file_dropdown],
        outputs=load_outputs
    ).then(
        fn=update_model_labels_and_values,
        inputs=[current_data, model_names_state],
        outputs=[item for sec in evaluation_sections for item in [sec['label']] + sec['inputs']]
    )
    
    # Save button handler
    save_btn.click(
        fn=save_evaluations,
        inputs=[current_file, current_data, model_names_state] + all_eval_inputs,
        outputs=save_status
    )

if __name__ == "__main__":
    app.launch()


