#!/usr/bin/env bash

models=("easy-khair-180-gpc0.8-trans10-025000.pkl")

in="models"
out="pti_out"
img_dir="my_img"
seg_dir="my_seg"

for model in ${models[@]}

do

    for i in 0

    do 
        # perform the pti and save w
        python projector_withseg.py --outdir=${out} --target_img=dataset/${img_dir} --target_seg=dataset/${seg_dir} --network ${in}/${model} --idx ${i} --num-steps=1000 --num-steps-pti 1000 --shapes=True --shape-format='.ply'
        # generate .mp4 before finetune
        # python gen_videos_proj_withseg.py --output=${out}/${model}/${i}/PTI_render/pre.mp4 --latent=${out}/${model}/${i}/projected_w.npz --trunc 0.7 --network ${in}/${model} --cfg Head
        # generate .mp4 after finetune
        python gen_videos_proj_withseg.py --output=${out}/${model}/${i}/PTI_render/post.mp4 --latent=${out}/${model}/${i}/projected_w.npz --trunc 0.7 --network ${out}/${model}/${i}/fintuned_generator.pkl --cfg Head --shapes=True --shape-format='.ply'


    done

done
