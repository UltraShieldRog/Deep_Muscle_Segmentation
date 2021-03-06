#!/bin/sh

# Stage 1 Thresholding => Stage 1 Morphological Processing and Downsampling => Manual Cleaning => Stage 2 Upsampling and Calculation
# --no-crop --no-n3 --no-norm --no-thr --no-mor --no-inter --no-downsam --no-upsam --no-clean --no-cal


s=$'-s 14_Jack'
tp=$'-t fu'
leg=$'-l'
cut=$'-bottom 60 -top 270'
thr=$'-upthr 0.4 -lowthr 0.1'
cal_cut=$'-calb 2 -calt 24'


# Just cropping
# python3 pipeline.py $s $tp $leg $cut $thr --no-snip --no-thr --no-norm --no-n3 --no-mor --no-inter --no-downsam --no-upsam --no-clean --no-cal

# Registration
python3 pipeline.py $s $tp $leg $cut $thr --no-snip --no-crop --no-thr --no-norm --no-n3 --no-mor --no-inter --no-downsam --no-upsam --no-clean --no-cal

# Stage 1 Preprocessing 
python3 pipeline.py $s $tp $leg $cut $thr --no-thr --no-mor --no-inter --no-downsam --no-upsam --no-clean --no-cal

# Stage 1 Testing thresholding
python3 pipeline.py $s $tp $leg $thr --no-snip --no-crop --no-n3 --no-norm --no-mor --no-inter --no-downsam --no-upsam --no-clean --no-cal

# Stage 1 Morphological processing
python3 pipeline.py $s $tp $leg $thr --no-snip --no-crop --no-n3 --no-norm --no-thr -eron 2 -diln 10 --no-upsam --no-clean --no-cal

# ***  Stage 1 ALL !!  ***
# python3 pipeline.py $s $tp $leg $thr --no-upsam --no-cal

# Manual Cleaning downsampled mask...
# modify [Timepoint]_[left/right]_mask_downsmapled_before_manual.nii.gz
# delete _before_manual after finished

# Stage 2 Upsampling and cleaning
# python3 pipeline.py $s $tp $leg $thr --no-snip --no-crop --no-n3 --no-norm --no-thr --no-mor --no-inter --no-downsam --no-cal

# Stage 2 Calculation (change back the method when back)
# python3 pipeline.py $s $tp $leg $thr $cal_cut --no-crop --no-snip --no-n3 --no-norm --no-thr --no-mor --no-inter --no-downsam --no-upsam --no-clean

# Batch Process
# s=$'-s 07_Jack'
# list=$list$"bl fu fu2 fu3 fu4"
# leg=$'-l'
# for i in $list; do
# 	echo $i
# 	# python3 pipeline.py $s $'-t'$i $leg $cut $thr --no-thr --no-mor --no-inter --no-downsam --no-upsam --no-clean --no-cal
# 	# python3 pipeline.py $s '-t'$i $leg $thr $cal_cut --no-crop --no-snip --no-n3 --no-norm --no-thr --no-mor --no-inter --no-downsam --no-upsam --no-clean
# 	# python3 pipeline.py $s $'-t'$i $leg $thr $cal_cut --no-crop --no-snip --no-n3 --no-norm --no-thr --no-mor --no-inter --no-downsam --no-upsam --no-clean
# 	python3 pipeline.py $s $'-t'$i $leg $thr $cal_cut --no-crop --no-snip --no-n3 --no-norm --no-thr --no-mor --no-inter --no-downsam --no-upsam --no-clean
# done

# leg=$'-l'
# for i in $list; do
# 	# echo $i
# 	python3 pipeline.py $s $'-t'$i $leg $cut $thr --no-n3 --no-thr --no-mor --no-inter --no-downsam --no-upsam --no-clean --no-cal
# done                   