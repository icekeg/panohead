import gradio as gr
from PIL import Image
from pathlib import Path
import os

import torch
import imageio

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from skimage.transform import resize
import warnings

#prepare 
cropping_folder = "/data/home/alfredchen/PanoHead/3DDFA_V2_cropping"
dataset_folder = "/data/home/alfredchen/PanoHead/dataset"

#finetuen 
source_Model = "easy-khair-180-gpc0.8-trans10-025000.pkl"
finetune_path = "pti_out"

##########################
#  interface and layout  #
##########################
with gr.Blocks() as demo:
    with gr.Group():
        with gr.Row():
            img_rgb = gr.Image(type="filepath", label="Source imgage", height=384, width=384)
            img_seg = gr.Image(type="filepath", label="Mask image", height=384, width=384)
        with gr.Row():
            folderName = gr.Textbox(label="Project name",value="temp")
            project_label = gr.Label(label="Current Project name is:",value="temp", color='#48D1CC')
        with gr.Row():
            prepareData_btn = gr.Button("prepare data for new image")
            submit_seg_btn = gr.Button("submit improved image")

    with gr.Group():
        with gr.Row():
            compare_video = gr.Video(label="Compare video", height=384, width=384)
            output_video = gr.Video(label="Post video", height=384, width=384)  

        source_model_path = gr.Textbox(label="Fine tune source model", value="/data/home/alfredchen/PanoHead/models/easy-khair-180-gpc0.8-trans10-025000.pkl")
        with gr.Row(elem_id="create finetune model"):
            finetune_steps = gr.Slider(label="Finetune steps",minimum=100,maximum=5000,step=100,value=500)
            finetune_steps_pti = gr.Slider(label="Finetune steps for pti",minimum=100,maximum=5000,step=100,value=500)
            createMesh = gr.Checkbox(False,label="Create geomtry?")
            mesh_type = gr.Dropdown(label="Mesh type", choices=['.ply','.mrc'], value='.ply')     
        finetune_btn = gr.Button("create finetune model")

    with gr.Group():
        finetune_model_path = gr.Textbox(label="Fine tune model", value="")
        finetune_latent_path = gr.Textbox(label="Fine tune latent", value="")
        createplyMesh = gr.Checkbox(False,label="Create ply geomtry with video?") 
        with gr.Row():
            video_up_angles = gr.Textbox(label="Video up angles", value="-1.4 -1.2 -1.0 -0.8 -0.7 -0.65 -0.6 -0.4 -0.2 0.0 0.2 0.4 0.6 0.8 1.0 1.2 1.4")
            mul_video_btn = gr.Button("create multiply view videos")
        with gr.Row():
            image_v_angles = gr.Textbox(label="Image vertical angles", value="-1.2 -1.0 -0.8 -0.7 -0.65 -0.6 -0.4 -0.2 0.0 0.2 0.4 0.6 0.8 1.0 1.2")
            image_h_angles = gr.Textbox(label="Image horizontal angles", value="-0.5 -0.4 -0.3 -0.25 -0.15 -0.1 0 0.1 0.15 0.25 0.3 0.4 0.5 0.57")
            img_name_type = gr.Dropdown(label="Name type", choices=['index', 'parameter'], value='index') 
            mul_img_btn = gr.Button("create multiply view images")

        gen_result = gr.Textbox(label="Generate result")
    

    #####################
    ### prepare data ####
    #####################
    def prepareData(img_path,floder_name):
        image_path = Path(img_path)
        image_raw = Image.open(image_path)

        temp_folder_name = floder_name


        source_img_folder = cropping_folder + "/img/" + temp_folder_name

        if not os.path.exists(source_img_folder):
            os.makedirs(source_img_folder)

        target_img_folder = dataset_folder + "/" + temp_folder_name
        if not os.path.exists(target_img_folder):
            os.makedirs(target_img_folder)
        if not os.path.exists(target_img_folder + "/img"):
            os.makedirs(target_img_folder + "/img")
        if not os.path.exists(target_img_folder + "/seg"):
            os.makedirs(target_img_folder + "/seg")

        source_img_path = source_img_folder + "/" + temp_folder_name + ".jpg"
        img_final = image_raw.save(source_img_path)
        print(os.system("sudo chmod 777 " + source_img_path))
        #assert os.path.exists(source_img_path)


        prepare_command ="python " + cropping_folder + "/dlib_kps_new.py " + "-i " + cropping_folder + " -sname " + temp_folder_name
        print(os.system(prepare_command))

        prepare_command ="python " + cropping_folder + "/recrop_images.py -i " + cropping_folder + "/data.pkl -o " + cropping_folder + "/quads.pkl --out_dir " + target_img_folder + "/img --config " + cropping_folder + "/configs/mb1_120x120_new.yml"
        #print(prepare_command)
        print(os.system(prepare_command))

        prepare_command = "rembg i -om " + target_img_folder + "/img/" + temp_folder_name +".jpg " + target_img_folder + "/seg/" + temp_folder_name +".png"
        print(os.system(prepare_command))

        return target_img_folder + "/seg/" + temp_folder_name +".png", floder_name
    
    #########################
    ### submit seg image ####
    #########################
    def replace_seg(seg_path, projectlabel):
        image_path = Path(seg_path)
        image_raw = Image.open(image_path)

        floder_name = projectlabel['label']
        seg_img_folder = dataset_folder + "/" + floder_name + "/seg/" + floder_name + ".png"

        img_final = image_raw.save(seg_img_folder)

        return "replaced " + seg_img_folder
    
    ##############################
    #   create finetune modle    #
    ##############################
    def gen_finetunemodel(projectlabel,tune_steps, tune_steps_pti,needMesh,meshtype,source_path):
        
        floder_name = projectlabel['label']
        #finetune_command = "python projector_withseg_new.py --outdir=" + finetune_path + " --target_img=dataset/" + floder_name + "/img --network models/" + source_Model + " --idx 0 --num-steps " + str(tune_steps) + " --idx 0 --num-steps-pti " + str(tune_steps_pti) + " --shapes=" + str(needMesh) + " --shape-format=" + meshtype + " --dir_name=" + floder_name
        finetune_command = "python projector_withseg_new.py --outdir=" + finetune_path + " --target_img=dataset/" + floder_name + "/img --network " + source_path + " --idx 0 --num-steps " + str(tune_steps) + " --idx 0 --num-steps-pti " + str(tune_steps_pti) + " --shapes=" + str(needMesh) + " --shape-format=" + meshtype + " --dir_name=" + floder_name
        print(os.system(finetune_command))

        #out_path = finetune_path + "/" + source_Model + "/" + floder_name
        out_path = finetune_path + "/" + floder_name

        if not os.path.exists(out_path + "/PTI_render"):
            os.makedirs(out_path + "/PTI_render")

        finetune_command = "python gen_videos_proj_withseg.py --output=" + out_path + "/PTI_render/post.mp4 --latent=" + out_path + "/projected_w.npz --trunc 0.7 --network " + out_path + "/fintuned_generator.pkl --cfg Head --shapes=" + str(needMesh) + " --shape-format=" + meshtype
        print(os.system(finetune_command))

        temp_path = "/data/home/alfredchen/PanoHead/" + out_path
        return temp_path + "/proj.mp4", temp_path + "/PTI_render/post.mp4", temp_path + "/fintuned_generator.pkl", temp_path + "/projected_w.npz"
    
    ########################################
    #   generate multiple angles videos    #
    ########################################
    def gen_videos(up_angles, modelpath, latentpath, create_plyMesh):
        videopath = modelpath.replace("fintuned_generator.pkl","") + "videos"
        if not os.path.exists(videopath):
            os.makedirs(videopath)

        angles = up_angles.split(" ")

        for i in angles:
            if i != angles[0]:
                needCreateMesh = False
            else:
                needCreateMesh = create_plyMesh
            
            gen_command = "python gen_videos_proj_withseg.py --output=" + videopath + "/" + i + ".mp4 --latent=" + latentpath + " --trunc 0.7 --network " + modelpath + " --cfg Head --shapes=" + str(needCreateMesh) + " --shape-format='.ply' --camera-up=" + i 
            print(os.system(gen_command))

        return videopath 
    
    ##########################################
    #   generate multiple angles pictures    #
    ##########################################
    def gen_images(v_angles, h_angles, modelpath, latentpath, nametype):
        imgpath = modelpath.replace("fintuned_generator.pkl","") + "images"

        if not os.path.exists(imgpath):
            os.makedirs(imgpath)

        v_angles = "'" + v_angles + "'"
        h_angles = "'" + h_angles + "'"
        gen_command = "python gen_samples_latent.py --outdir=" + imgpath + " --trunc=0.7 --shapes=False --network " + modelpath + " --shape-format='.ply' --shape-res=512 --camera-up=-0.9 --latent=" + latentpath + " --vangles=" + v_angles + " --hangles=" + h_angles + " --name-type=" + nametype
        print(gen_command)
        print(os.system(gen_command))

        return imgpath


    prepareData_btn.click(prepareData,[img_rgb,folderName],[img_seg, project_label])    
    submit_seg_btn.click(replace_seg, [img_seg, project_label], gen_result)

    finetune_btn.click(gen_finetunemodel,[project_label, finetune_steps, finetune_steps_pti, createMesh, mesh_type, source_model_path],[compare_video, output_video, finetune_model_path, finetune_latent_path])

    mul_video_btn.click(gen_videos, [video_up_angles, finetune_model_path, finetune_latent_path, createplyMesh], gen_result)
    mul_img_btn.click(gen_images, [image_v_angles, image_h_angles, finetune_model_path, finetune_latent_path, img_name_type], gen_result)

demo.launch()

  