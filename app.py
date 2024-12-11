import gradio as gr
from model import load_model
from generate import generate_text
from generate import generate_code
from database import create_db, insert_into_db, clear_database
import train

# Supported models_general
models_options_general = ["gpt2", "gpt2-medium", "gpt2-persian"]
models_options_codegen = ['codegen']

train_pass = '68187451'

# Load models_general
models_general = {model_name: {"model": load_model(model_name)[0], "tokenizer": load_model(model_name)[1]} for model_name in models_options_general}
models_codegen = {"codegen": {"model": load_model("codegen")[0], "tokenizer": load_model("codegen")[1]}}

# AI-Powered Story World Builder Functions
world_data = {}

def define_world(world_name, locations, characters):
    """
    Define a new story world with locations and characters.
    """
    world_data["world_name"] = world_name
    world_data["locations"] = locations.split(", ")
    world_data["characters"] = characters.split(", ")
    return f"World '{world_name}' created with locations: {locations} and characters: {characters}"

def generate_story(world_name, event, max_length, seed):
    """
    Generate a story based on the defined world and an event.
    """
    if not world_name or not world_data.get("world_name"):
        return "Error: Please define a world first."

    if world_name != world_data["world_name"]:
        return f"Error: World '{world_name}' not found. Define it first."

    prompt = f"In the world of {world_name}, {event}. Locations: {', '.join(world_data['locations'])}. Characters: {', '.join(world_data['characters'])}."
    generated_story = generate_text(models_general, "gpt2", prompt, max_length, seed)
    return generated_story


# Story Mode
story = []

# Main Function For Story Generating
def interactive_story(input_text, selected_model, max_length, seed):
    global story
    if input_text.strip():
        story.append(input_text)  # افزودن ورودی کاربر به داستان
    current_text = " ".join(story)  # ساخت داستان تجمعی
    
    generated_text = generate_text(models_general, selected_model, current_text, max_length, seed)
    story.append(generated_text)  # افزودن متن تولید شده توسط مدل
    
    return current_text + "\n\n" + generated_text

def reset_story():
    global story
    story = []  # بازنشانی داستان
    return ""

# Function to generate multiple parallel worlds from a single input
def generate_multiverse(input_text, selected_model, max_new_tokens, seed, num_worlds=3):
    worlds = []
    for i in range(num_worlds):
        world_intro = f"World {i + 1}: "
        
        # Custom logic for different parallel worlds
        if i == 0:
            world_intro += f"{input_text} This world leads to a parallel universe!"
        elif i == 1:
            world_intro += f"{input_text} In this world, time splits into different periods!"
        elif i == 2:
            world_intro += f"{input_text} This world faces a strange physical anomaly that changes everything!"
        
        # Generate the continuation of the story
        generated_text = generate_text(
            models_general, 
            selected_model, 
            world_intro, 
            max_new_tokens, 
            seed, 
            #repetition_penalty=1.2  # Avoid repetition
        )
        worlds.append(generated_text)
    
    return "\n\n".join(worlds)

# Main function for generating text
def generate(input_text, selected_model, max_new_token, seed):
    insert_into_db(input_text, selected_model, seed)
    return generate_text(models_general, selected_model, input_text, max_new_token, seed)

# Function to verify password, train the model, and clear the database
def verify_and_train(selected_model, epochs, batch_size, password, custom_text):
    if password != train_pass:
        return "Error: Incorrect password. Training not started."
    
    if custom_text.strip():  # Check if custom text is provided
        train.train_model_with_text(selected_model, custom_text, epochs, batch_size)
        return f"Training completed for model: {selected_model} using custom text."
    else:
        train.train_model(selected_model, epochs, batch_size)
        clear_database()
        return f"Training completed for model: {selected_model}. Database cleared."

# Create database
create_db()

# Interface
with gr.Blocks() as interface:
    gr.Markdown(
        "# **GPT Tools**\n\n"
        "Generate something using GPT models. Select the model and adjust the parameters for optimal results."
    )
    with gr.Tabs():
        with gr.Tab("Text Generator"):
            with gr.Row():
                with gr.Column(scale=1, min_width=350):
                    input_text = gr.Textbox(label="Input Text", placeholder="Enter your text here...", lines=4, max_lines=6)
                    selected_model = gr.Radio(choices=models_options_general, value="gpt2", label="Select Model", type="value")
                    with gr.Row():
                        max_tokens = gr.Slider(10, 100, value=50, step=1, label="Max New Tokens", interactive=True)
                        seed = gr.Slider(10, 50, value=42, step=1, label="Seed", interactive=True)

                with gr.Column(scale=1, min_width=350):
                    output_text = gr.Textbox(label="Generated Text", interactive=False, lines=8, max_lines=12)
                    generate_button = gr.Button("Generate Text", variant="primary")

            generate_button.click(
                generate,
                inputs=[input_text, selected_model, max_tokens, seed],
                outputs=output_text,
            )

        # Multiverse Story Generator Tab
        with gr.Tab("Multiverse Story Generator"):
            with gr.Row():
                with gr.Column(scale=1, min_width=350):
                    input_text = gr.Textbox(label="Enter your story idea", placeholder="e.g. A scientist discovers a parallel universe...", lines=4, max_lines=6)
                    selected_model = gr.Radio(choices=models_options_general, value="gpt2", label="Select Model for Story Generation", type="value")
                    max_length = gr.Slider(50, 300, value=150, step=1, label="Max Length", interactive=True)
                    seed = gr.Slider(1, 100, value=42, step=1, label="Seed", interactive=True)

                with gr.Column(scale=1, min_width=350):
                    output_text = gr.Textbox(label="Generated Worlds", interactive=False, lines=12, max_lines=20)
                    generate_button = gr.Button("Generate Parallel Worlds", variant="primary")

            generate_button.click(
                generate_multiverse,
                inputs=[input_text, selected_model, max_length, seed],
                outputs=output_text,
            )

        # Interactive Story Writing Tab
        with gr.Tab("Interactive Story Writing"):
            with gr.Row():
                with gr.Column(scale=1, min_width=350):
                    story_input = gr.Textbox(label="Add to Story", placeholder="Enter your part of the story...", lines=4, max_lines=6)
                    story_model = gr.Radio(choices=models_options_general, value="gpt2", label="Select Model", type="value")
                    story_max_length = gr.Slider(50, 300, value=50, step=1, label="Max Length", interactive=True)
                    story_seed = gr.Slider(10, 50, value=42, step=1, label="Seed", interactive=True)

                with gr.Column(scale=1, min_width=350):
                    story_text = gr.Textbox(label="Story So Far", interactive=False, lines=12, max_lines=20)
                    story_button = gr.Button("Generate Next Part", variant="primary")
                    reset_button = gr.Button("Reset Story", variant="secondary")

            # Connecting buttons to respective functions
            story_button.click(
                interactive_story,
                inputs=[
                    story_input,
                    story_model,
                    story_max_length,
                    story_seed
                ],
                outputs=story_text,
            )
            reset_button.click(
                reset_story,
                inputs=[],
                outputs=story_text,
            )

        # Training Tab
        with gr.Tab("Training"):
            gr.Markdown("# **Train Model**\n\n")
            with gr.Column(scale=1, min_width=250):
                train_model_selector = gr.Radio(choices=models_options_general, value="gpt2", label="Select Model for Training", type="value")
                epochs = gr.Slider(1, 100, value=50, step=1, label="Epochs", interactive=True)
                batch_size = gr.Slider(1, 100, value=16, step=1, label="Batch Size", interactive=True)
                password = gr.Textbox(label="Enter Training Password", placeholder="Enter password", type="password")
                custom_text = gr.Textbox(label="Custom Text (optional)", placeholder="Enter custom text for training...")
                train_button = gr.Button("Train Model", variant="primary")
                train_status = gr.Textbox(label="Training Status", interactive=False)
                train_button.click(
                    verify_and_train,
                    inputs=[train_model_selector, epochs, batch_size, password, custom_text],
                    outputs=train_status,
                )
       
        # Code Generator Tab
        with gr.Tab("Code Generator"):
            gr.Markdown("### Generate Code from Descriptions")
            with gr.Row():
                with gr.Column(scale=1, min_width=350):
                    code_prompt = gr.Textbox(label="Code Prompt", placeholder="Describe your coding task, e.g., 'Write a Python function to calculate Fibonacci numbers.'")
                    code_max_tokens = gr.Slider(10, 500, value=150, step=10, label="Max Tokens")
                    code_seed = gr.Slider(0, 100, value=42, step=1, label="Seed")
                with gr.Column(scale=1, min_width=350):
                    generated_code = gr.Textbox(label="Generated Code", interactive=False, lines=10, max_lines=20)
                    generate_code_button = gr.Button("Generate Code")

        generate_code_button.click(
            lambda prompt, max_new_tokens, seed: generate_code(models_codegen, "codegen", prompt, max_new_tokens, seed),
            inputs=[code_prompt, code_max_tokens, code_seed],
            outputs=generated_code,
        )
        # Add AI-Powered Story World Builder Tab
        with gr.Tab("AI-Powered Story World Builder"):
            with gr.Row():
                with gr.Column(scale=1, min_width=350):
                    world_name = gr.Textbox(label="World Name", placeholder="Enter your world name...")
                    locations = gr.Textbox(label="Locations", placeholder="Enter locations separated by commas...")
                    characters = gr.Textbox(label="Characters", placeholder="Enter characters separated by commas...")
                    create_button = gr.Button("Create World", variant='primary')
                    generate_story_button = gr.Button("Generate Story")
                with gr.Column(scale=1, min_width=350):
                    world_status = gr.Textbox(label="World Status", interactive=False)
                    generated_story = gr.Textbox(label="Generated Story", interactive=False, lines=12, max_lines=20)


            create_button.click(
                define_world,
                inputs=[world_name, locations, characters],
                outputs=world_status,
            )

            gr.Markdown("### Generate a Story in Your World")
            with gr.Row():
                with gr.Column(scale=1, min_width=350):
                    story_world = gr.Textbox(label="Enter World Name", placeholder="World name...")
                    event = gr.Textbox(label="Event", placeholder="Describe an event in the world...")
                    max_length = gr.Slider(50, 300, value=150, step=1, label="Max Length")
                    seed = gr.Slider(1, 100, value=42, step=1, label="Seed")

    generate_story_button.click(
        generate_story,
        inputs=[story_world, event, max_length, seed],
        outputs=generated_story,
    )

    gr.Markdown(
        "___\n"
        "Made by **AliMc2021** with ❤️"
    )

# Launch the interface
interface.queue().launch(
    server_port=7860, 
    show_error=True, 
    inline=False,
)
