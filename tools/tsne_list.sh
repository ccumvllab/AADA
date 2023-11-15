# gta2cs_MIC_baseline_AllClass_paratest
CUDA_VISIBLE_DEVICES=3 python -m tools.test_tSNE \
/home/rayeh/master/MIC-TA/seg/work_dirs/gtaHR2csHR_mic_hrda_650a8/gtaHR2csHR_mic_hrda_650a8.py \
/home/rayeh/master/MIC-TA/seg/work_dirs/gtaHR2csHR_mic_hrda_650a8/iter_40000_relevant.pth \
--sample 500 --show-dir gta2cs_MIC_baseline_AllClass_s500 --SelClass all

# gta2cs_MIC_baseline_EasyClass
CUDA_VISIBLE_DEVICES=3 python -m tools.test_tSNE \
/home/rayeh/master/MIC-TA/seg/work_dirs/gtaHR2csHR_mic_hrda_650a8/gtaHR2csHR_mic_hrda_650a8.py \
/home/rayeh/master/MIC-TA/seg/work_dirs/gtaHR2csHR_mic_hrda_650a8/iter_40000_relevant.pth \
--sample 500 --show-dir gta2cs_MIC_baseline_EasyClass_s500 --SelClass easy

# gta2cs_MIC_baseline_HardClass
CUDA_VISIBLE_DEVICES=3 python -m tools.test_tSNE \
/home/rayeh/master/MIC-TA/seg/work_dirs/gtaHR2csHR_mic_hrda_650a8/gtaHR2csHR_mic_hrda_650a8.py \
/home/rayeh/master/MIC-TA/seg/work_dirs/gtaHR2csHR_mic_hrda_650a8/iter_40000_relevant.pth \
--sample 500 --show-dir gta2cs_MIC_baseline_HardClass_s500 --SelClass hard

# gta2cs_MIC_baseline_MixClass
CUDA_VISIBLE_DEVICES=3 python -m tools.test_tSNE \
/home/rayeh/master/MIC-TA/seg/work_dirs/gtaHR2csHR_mic_hrda_650a8/gtaHR2csHR_mic_hrda_650a8.py \
/home/rayeh/master/MIC-TA/seg/work_dirs/gtaHR2csHR_mic_hrda_650a8/iter_40000_relevant.pth \
--sample 500 --show-dir gta2cs_MIC_baseline_MixClass_s500 --SelClass mix

# gta2cs_GT_N_AllClass
CUDA_VISIBLE_DEVICES=3 python -m tools.test_tSNE /home/rayeh/master/MIC-TA/seg/work_dirs/local-basic/230615_0325_gta2cs_MIC_GT_Normal_AugUp1_Auglr-04/230615_0325_gtaHR2csHR_mic_hrda_s2_1ef12.py /home/rayeh/master/MIC-TA/seg/work_dirs/local-basic/230615_0325_gta2cs_MIC_GT_Normal_AugUp1_Auglr-04/best_mIoU_iter_39000.pth --sample 100 --show-dir gta2cs_GT_N_AllClass_val_s500 --SelClass all

# gta2cs_GT_N_EasyClass
CUDA_VISIBLE_DEVICES=3 python -m tools.test_tSNE \
/home/rayeh/master/MIC-TA/seg/work_dirs/local-basic/230615_0325_gta2cs_MIC_GT_Normal_AugUp1_Auglr-04/230615_0325_gtaHR2csHR_mic_hrda_s2_1ef12.py \
/home/rayeh/master/MIC-TA/seg/work_dirs/local-basic/230615_0325_gta2cs_MIC_GT_Normal_AugUp1_Auglr-04/best_mIoU_iter_39000.pth \
--sample 500 --show-dir gta2cs_GT_N_EasyClass_s500 --SelClass easy

# gta2cs_GT_N_HardClass
CUDA_VISIBLE_DEVICES=3 python -m tools.test_tSNE \
/home/rayeh/master/MIC-TA/seg/work_dirs/local-basic/230615_0325_gta2cs_MIC_GT_Normal_AugUp1_Auglr-04/230615_0325_gtaHR2csHR_mic_hrda_s2_1ef12.py \
/home/rayeh/master/MIC-TA/seg/work_dirs/local-basic/230615_0325_gta2cs_MIC_GT_Normal_AugUp1_Auglr-04/best_mIoU_iter_39000.pth \
--sample 500 --show-dir gta2cs_GT_N_HardClass_s500 --SelClass hard

# gta2cs_GT_N_MixClass
CUDA_VISIBLE_DEVICES=3 python -m tools.test_tSNE \
/home/rayeh/master/MIC-TA/seg/work_dirs/local-basic/230615_0325_gta2cs_MIC_GT_Normal_AugUp1_Auglr-04/230615_0325_gtaHR2csHR_mic_hrda_s2_1ef12.py \
/home/rayeh/master/MIC-TA/seg/work_dirs/local-basic/230615_0325_gta2cs_MIC_GT_Normal_AugUp1_Auglr-04/best_mIoU_iter_39000.pth \
--sample 500 --show-dir gta2cs_GT_N_MixClass_s500 --SelClass mix