#!/usr/bin/env bash

model="pti_out/easy-khair-180-gpc0.8-trans10-025000.pkl/1/fintuned_generator.pkl"
latent_path="pti_out/easy-khair-180-gpc0.8-trans10-025000.pkl/1/projected_w.npz"


head_angles=(-1.4 -1.2 -1.0 -0.8 -0.7 -0.65 -0.6 -0.4 -0.2 0.0 0.2 0.4 0.6 0.8 1.0 1.2 1.4)

floder_name="temp"
out="out"


for angle in ${head_angles[@]}

do

    video_name=${floder_name}_"$angle".mp4
    python gen_videos_proj_withseg.py --output=${out}/${floder_name}/${video_name} --latent=${latent_path} --trunc 0.7 --network ${model} --cfg Head --shapes=False --shape-format='.ply'
    #echo ${out}/${floder_name}/${video_name}
done

#sleep 10000
