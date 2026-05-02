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
    
    gr.Markdown("---")
    
    # Create evaluation inputs for up to 10 models (adjust as needed)
    evaluation_sections = []
    all_eval_inputs = []
    model_display_sections = []
    
    for i in range(10):  # Support up to 10 models
        with gr.Group(visible=False) as group:
            model_header = gr.Markdown(f"### Model {i+1}")
            
            with gr.Row():
                # Left column: Model answer and LLM evaluation
                with gr.Column(scale=1):
                    model_answer_display = gr.Markdown("")
                    llm_eval_display = gr.Markdown("")
                
                # Right column: User evaluation
                with gr.Column(scale=1):
                    gr.Markdown("#### ✍️ Your Evaluation")
                    
                    with gr.Row():
                        acc = gr.Slider(1, 5, step=1, label="Accuracy", value=3)
                        use = gr.Slider(1, 5, step=1, label="Usefulness", value=3)
                    
                    with gr.Row():
                        comp = gr.Slider(1, 5, step=1, label="Completeness", value=3)
                        conc = gr.Slider(1, 5, step=1, label="Conciseness", value=3)
                    
                    with gr.Row():
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
                'header': model_header,
                'inputs': [acc, use, comp, conc, clar, total, comments]
            })
            model_display_sections.append({
                'answer': model_answer_display,
                'llm_eval': llm_eval_display
            })
            all_eval_inputs.extend([acc, use, comp, conc, clar, total, comments])
    
    # Save button
    with gr.Row():
        save_btn = gr.Button("💾 Save All Evaluations", variant="primary", visible=False)
        save_status = gr.Markdown("", visible=False)
    
    def build_single_model_display(model_name, model_data):
        """Build display for a single model's answer and LLM evaluation"""
        answer_md = f"#### 🤖 Model: `{model_name}`\n\n**Answer:**\n\n{model_data.get('answer', 'No answer')}"
        
        llm_eval = model_data.get('llm_eval', {})
        eval_md = "#### 🔍 LLM Evaluation\n\n"
        
        if llm_eval:
            eval_md += f"- **Accuracy:** {llm_eval.get('accuracy', 'N/A')}/5\n"
            eval_md += f"- **Usefulness:** {llm_eval.get('usefulness', 'N/A')}/5\n"
            eval_md += f"- **Completeness:** {llm_eval.get('completeness', 'N/A')}/5\n"
            eval_md += f"- **Conciseness:** {llm_eval.get('conciseness', 'N/A')}/5\n"
            eval_md += f"- **Clarity:** {llm_eval.get('clarity', 'N/A')}/5\n"
            eval_md += f"- **Total:** {llm_eval.get('total', 'N/A')}/25\n\n"
            
            if llm_eval.get('comments'):
                eval_md += f"**Comments:** {llm_eval['comments']}\n"
        else:
            eval_md += "*No LLM evaluation available*"
        
        return answer_md, eval_md
    
    def load_file_handler(directory, filename):
        """Handle file loading"""
        if filename in ["No JSON files found", "Directory not found", None]:
            return [
                None,  # current_file
                {},    # current_data
                [],    # model_names
                gr.Markdown(visible=False),  # question
                gr.Markdown(visible=False),  # references
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
                gr.Button(visible=False),
                gr.Markdown(visible=False),
            ] + [gr.Group(visible=False) for _ in range(10)]
        
        # Build displays
        question_md = f"# 📋 Question\n\n**{data.get('question', 'No question')}**"
        references_md = f"## 📚 References\n{data.get('references', 'No references')}"
        
        # Get model names and prepare evaluation sections
        model_names, human_evals = create_evaluation_inputs(data)
        num_models = len(model_names)
        
        # Prepare updates for evaluation sections
        eval_updates = []
        for i in range(10):
            if i < num_models:
                # Make group visible
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
            gr.Button(visible=True if num_models > 0 else False),
            gr.Markdown("", visible=True),
        ] + eval_updates
    
    def update_model_displays_and_values(data, model_names):
        """Update both model displays and evaluation values"""
        updates = []
        
        for i in range(10):
            if i < len(model_names):
                model_name = model_names[i]
                model_data = data["models"][model_name]
                human_eval = model_data.get("human_eval", {})
                
                # Build model answer and LLM eval displays
                answer_md, eval_md = build_single_model_display(model_name, model_data)
                
                # Return values for: header, answer_display, llm_eval_display, acc, use, comp, conc, clar, total, comments
                updates.extend([
                    gr.Markdown(f"### 🤖 Model {i+1}: `{model_name}`"),
                    gr.Markdown(answer_md),
                    gr.Markdown(eval_md),
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
                updates.extend([gr.Markdown(""), gr.Markdown(""), gr.Markdown(""), 3, 3, 3, 3, 3, 15, ""])
        
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
        save_btn,
        save_status,
    ] + [sec['group'] for sec in evaluation_sections]
    
    # Collect all outputs for the update function
    display_and_value_outputs = []
    for i in range(10):
        display_and_value_outputs.extend([
            evaluation_sections[i]['header'],
            model_display_sections[i]['answer'],
            model_display_sections[i]['llm_eval'],
        ] + evaluation_sections[i]['inputs'])
    
    load_btn.click(
        fn=load_file_handler,
        inputs=[directory, file_dropdown],
        outputs=load_outputs
    ).then(
        fn=update_model_displays_and_values,
        inputs=[current_data, model_names_state],
        outputs=display_and_value_outputs
    )
    
    # Save button handler
    save_btn.click(
        fn=save_evaluations,
        inputs=[current_file, current_data, model_names_state] + all_eval_inputs,
        outputs=save_status
    )
