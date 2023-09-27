import gradio as gr
import utility
from tifffile import imread, imsave
from model.fmirmodel import FmirModel


# according to your vision task
TASKS = ['SR_CCPs','Isotropic_Liver','Denoising_Planaria']
MODELS = ['FBuncertainty_f_uncer', 'FBuncertainty_f_twostage']


def run_model(img_input, model_type, task_type, progress=gr.Progress()):
    # debug info
    print(f'Running model {model_type} on task {task_type}...')

    # preprocessing
    if img_input is None:
        gr.Error("Model not loaded!")
        return [None, None]

    print(f'Opening {img_input.name}...')
    if not img_input.name.endswith('.tif'):
        gr.Error("Image must be a tiff file!")
        return None

    # read input
    image = imread(img_input.name)

    if 'SR' in task_type:
        if image.ndim != 2:
            gr.Error("SR Image must be 2 dimensional!")
            return [None, None]

    elif 'Isotropic' or 'Denoising' in task_type:
        if image.ndim != 3:
            gr.Error("Isotropic/Denoising Image must be 3 dimensional!")
            return [None, None]            
            
    else:
        gr.Error("This task is not supported yet!") 
        return [None, None]
    
    # run model
    model = FmirModel(image, task_type, model_type)
    sr, sr_norm = model.run_model()
    imsave('output.tif', sr)

    del model

    return ['output.tif', sr_norm]


def visualize(img_input, progress=gr.Progress()):
    print(f'Opening {img_input.name}...')
    if not img_input.name.endswith('.tif'):
        gr.Error("Image must be a tiff file!")
        return None
    
    image = imread(img_input.name)
    print(f'Image shape: {image.shape}')

    if len(image.shape) == 2:
        image = utility.savecolorim(None, image, norm=True)
        return [[image], f'2D image loaded with shape {image.shape}']
    elif len(image.shape) == 3:
        clips = []
        for i in range(image.shape[0]):
            clips.append(utility.savecolorim(None, image[i], norm=True))
        return [clips, f'3D image loaded with shape {image.shape}']
    else:
        gr.Error("Image must be 2 or 3 dimensional!")
        return None


"""
- the frontend of the demo
"""
with gr.Blocks() as demo:

    gr.Markdown("# Distribution-informed Learning Enables High-quality and High-robustness Restoration of Fluorescence Microscopy")
    gr.Markdown(
    "This demo allows you to run the models on your own images or the examples  from the paper. Please refer to the paper for more details.")

    gr.Markdown("## Instructions")
    gr.Markdown("1. Upload your tiff image or use the examples below. This online project accepts 2 (xy) dimensional images for SR and 3 (zxy) dimensional images for Denoising and Isotropic.")
    gr.Markdown("2. Click 'Check Input' to inspect your input image. This may take a while to display the image.")
    gr.Markdown("3. Select the model you want to run. This online project provides models for different tasks and datasets, including SR (CCPs), Denoising (Planaria),Isotropic (Liver).")
    gr.Markdown("4. Click 'Restore Image' to run the model on the input image. Some tasks like denoising will take several minutes to run.")
    gr.Markdown("5. The output image will be saved as 'output.tif' for download.")

    with gr.Row():
        with gr.Column():
            gr.Markdown("## Upload Image or Use Examples")
                
            with gr.Column():
                img_input = gr.File(label="Input File", interactive=True)
                img_visual = gr.Gallery(label="Input Viusalization", interactive=False)

            with gr.Row():
                load_image = gr.Textbox(label="Image Information", value="Image not loaded")
                check_input = gr.Button("Check Input") 

            with gr.Row():
                with gr.Column():
                    gr.Examples(
                        label='Super Resolution Examples',
                        examples=[
                            ["exampledata/BioSR/CCPs/im1_LR.tif",'SR_CCPs'],
                        ],
                        inputs=[img_input, load_image],
                    )

                    gr.Examples(
                        label='Isotropic Examples',
                        examples=[
                            ["exampledata/Isotropic/input_subsample_8.tif",'Isotropic_Liver'],
                        ],
                        inputs=[img_input, load_image],
                    )

                    gr.Examples(
                        label='Denoising Examples',
                        examples=[
                            ["exampledata/Denoising_Planaria/EXP278_Smed_fixed_RedDot1_sub_5_N7_m0003.tif","Denoising_Planaria"],
                        ],
                        inputs=[img_input, load_image],
                    )

        with gr.Column():
            gr.Markdown("## Load and Run Model")
            output_file = gr.File(label="Output File", interactive=False)
            img_output = gr.Gallery(label="Output Visualiztion")

            with gr.Row():
                # device = gr.Dropdown(label="Device", choices=DEVICES, value="CUDA")
                # quantization = gr.Dropdown(label="Quantization", choices=QUANT, value="float16")
                # chop = gr.Dropdown(label="Chop", choices=['Yes','No'], value="Yes")
                model_type = gr.Dropdown(label="Model Type", choices=MODELS, value="FBuncertainty_f_uncer",
                                     interactive=True)

            with gr.Row():
                task_type = gr.Dropdown(label="Task Type", choices=TASKS, value="SR_CCPs", interactive=True)
                # load_btn = gr.Button("Load Model")
                
            with gr.Row():
                # load_progress = gr.Textbox(label="Model Information", value="Model not loaded")
                run_btn = gr.Button("Restore Image")

    gr.Markdown("Internet Content Provider ID: [沪ICP备2023024810号-1](https://beian.miit.gov.cn/)", rtl=True)

    check_input.click(visualize, inputs=img_input, outputs=[img_visual,load_image], queue=True)
    # load_btn.click(load_model,inputs=[type, device, chop, quantization],outputs=load_progress, queue=True)
    run_btn.click(run_model, inputs=[img_input, model_type, task_type], outputs=[output_file, img_output], queue=True)

demo.queue().launch(server_name='0.0.0.0', server_port=7867)

