#!/usr/bin/env bash

cd "./outputs/connectome_constrained_vision/closed_loop_control/"
for terrain_type in "flat" "blocks"
do
    for stabilization_on in "True" "False"
    do
        echo $terrain_type $stabilization_on
        glob_pattern="${terrain_type}terrain_stabilization${stabilization_on}_x*y*.mp4"
        videolist_file="videolist_${terrain_type}_stab${stabilization_on}.txt"
        ls $glob_pattern > $videolist_file
        sed -i \
            -e "s/${terrain_type}/file '${terrain_type}/g" \
            -e "s/\.mp4/.mp4'/g" \
            $videolist_file
        output_file="concatvideo_${terrain_type}_stab${stabilization_on}.mp4"
        ffmpeg -f concat -i $videolist_file -c copy $output_file
        rm $videolist_file
    done
done
