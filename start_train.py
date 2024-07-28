from tools.train import main as train
from tools.test import main as test
from tools.analysis_tools.benchmark import main as analysis

# test@100.77.203.61
'/home/datassd_2T/lhy_temperary/First_Ablation_Experiment/checkpoints/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco_20230117_180101-5aa5f0f1.pth'
# configs/roi_trans/roi-trans-le90_yolov8_yolopafpn_1x_dota.py --work-dir /mnt/mydisk/code/total_work_result/train_work_result/
# configs/roi_trans/roi-trans-le90_yolov8_yolopafpn_3head_1x_dota.py --work-dir /mnt/mydisk/code/total_work_result/train_work_result/
# configs/roi_trans/roi-trans-le90_yolov8_yolopafpn_l_1x_dota.py --work-dir /mnt/mydisk/code/total_work_result/train_work_result/
# configs/roi_trans/roi-trans-le90_r50_yolov8fpn_multihead_1x_dota.py --work-dir /mnt/mydisk/code/total_work_result/train_work_result/
# configs/roi_trans/roi-trans-le90_r50_fpn_1x_dota.py --work-dir /mnt/mydisk/code/total_work_result/roi_yolo_work_result/
# configs/roi_trans/roi-trans-le90_yolov8_pafpn_1x_dota.py --work-dir /mnt/mydisk/code/total_work_result/train_work_result/
# configs/roi_trans/roi-trans-le90_yolov8_dota.py --work-dir /mnt/mydisk/code/total_work_result/train_work_result/
# configs/roi_trans/roi-trans-le90_wtconv_fpn_1x_dota.py --work-dir /mnt/mydisk/code/total_work_result/train_work_result/
# configs/roi_trans/roi-trans-le90_r18_fpn_1x_dota.py --work-dir /mnt/mydisk/code/total_work_result/train_work_result/

# configs/s2anet/s2anet-le90_yolo_1x_dota.py --work-dir /mnt/mydisk/code/total_work_result/train_work_result/
# configs/s2anet/s2anet-le90_yolo_l_1x_dota.py --work-dir /mnt/mydisk/code/total_work_result/train_work_result/
# configs/s2anet/s2anet-le90_yolo_ex_1x_dota.py --work-dir /mnt/mydisk/code/total_work_result/train_work_result/
# configs/s2anet/s2anet-le90_yolo_5head_1x_dota.py --work-dir /mnt/mydisk/code/total_work_result/train_work_result/
# configs/s2anet/s2anet-le90_r50_fpn_1x_dota.py --work-dir /mnt/mydisk/code/total_work_result/train_work_result/
# configs/s2anet/s2anet-le90_r18_fpn_1x_dota.py --work-dir /mnt/mydisk/code/total_work_result/train_work_result/

# configs/roi_trans/roi-trans-le90_yolov8_dota.py --checkpoint ../checkpoints/yolo/epoch_4.pth --task inference
# configs/roi_trans/roi-trans-le90_yolov8_dota.py --checkpoint ../checkpoints/yolov8/yolov8_s_syncbn_fast_8xb16-500e_coco_20230117_180101-5aa5f0f1.pth --task inference
# configs/roi_trans/roi-trans-le90_r50_fpn_1x_dota.py --checkpoint ../checkpoints/roi_trans_res_result/epoch_12.pth --task inference
# configs/roi_trans/roi-trans-le90_r50_fpn_1x_dota.py --checkpoint ../checkpoints/yolo/epoch_4.pth --task inference
# configs/roi_trans/roi-trans-le90_r50_pafpn_1x_dota.py --checkpoint ../checkpoints/yolo/epoch_4.pth --task inference
# configs/roi_trans/roi-trans-le90_swin-tiny_fpn_1x_dota.py --checkpoint ../checkpoints/roi_trans_swin/roi_trans_swin_tiny_fpn_1x_dota_le90-ddeee9ae.pth --task inference
# configs/roi_trans/roi-trans-le90_vit_fpn_1x_dota.py --checkpoint ../checkpoints/roi_trans_swin/roi_trans_swin_tiny_fpn_1x_dota_le90-ddeee9ae.pth --task inference


# configs/roi_trans/roi-trans-le90_r50_fpn_1x_dota.py ../../total_work_result/roi_result_12/epoch_12.pth --work-dir ../../total_work_result/roi_result_12/test_result
# configs/roi_trans/roi-trans-le90_r50_fpn_1x_dota.py --checkpoint ../../total_work_result/roi_result_12/epoch_12.pth --task inference

# configs/s2anet/s2anet-le90_r50_fpn_1x_dota.py ../../total_work_result/s2anet_re50_result/epoch_12.pth --work-dir ../../total_work_result/s2anet_re50_result/test_result
# configs/s2anet/s2anet-le90_r50_fpn_1x_dota.py --checkpoint ../../total_work_result/s2anet_re50_result/epoch_12.pth --task inference

# configs/roi_trans/roi-trans-le90_yolov8_yolopafpn_1x_dota.py ../../total_work_result/roi_yolo_result/epoch_12.pth --work-dir ../../total_work_result/roi_yolo_result/test_result
# configs/roi_trans/roi-trans-le90_yolov8_yolopafpn_1x_dota.py --checkpoint ../../total_work_result/roi_yolo_result/epoch_12.pth --task inference

# configs/s2anet/s2anet-le90_yolo_1x_dota.py ../../total_work_result/s2_result/epoch_12.pth --work-dir ../../total_work_result/s2_result/test_result
# configs/s2anet/s2anet-le90_yolo_1x_dota.py --checkpoint ../../total_work_result/s2_result/epoch_12.pth --task inference

# configs/s2anet/s2anet-le90_yolo_ex_1x_dota.py ../../total_work_result/s2_yolo_4head_result2/epoch_12.pth --work-dir ../../total_work_result/s2_yolo_4head_result2/test_result/eval/
# configs/s2anet/s2anet-le90_yolo_ex_1x_dota.py --checkpoint ../../total_work_result/s2_yolo_4head_result2/epoch_12.pth --task inference

# configs/s2anet/s2anet-le90_yolo_5head_1x_dota.py ../../total_work_result/s2_yolo_5head/epoch_12.pth --work-dir ../../total_work_result/s2_yolo_5head/test_result/eval/
# configs/s2anet/s2anet-le90_yolo_5head_1x_dota.py --checkpoint ../../total_work_result/s2_yolo_5head/epoch_12.pth --task inference

# configs/s2anet/s2anet-le90_r18_fpn_1x_dota.py ../../total_work_result/s2_r18_result/epoch_12.pth --work-dir ../../total_work_result/s2_r18_result/test_result/eval/
# configs/s2anet/s2anet-le90_r18_fpn_1x_dota.py --checkpoint ../../total_work_result/s2_r18_result/epoch_12.pth --task inference

# configs/roi_trans/roi-trans-le90_r18_fpn_1x_dota.py ../../total_work_result/roi_re18_result/epoch_12.pth --work-dir ../../total_work_result/roi_re18_result/test_result/eval/
# configs/roi_trans/roi-trans-le90_r18_fpn_1x_dota.py --checkpoint ../../total_work_result/roi_re18_result/epoch_12.pth --task inference

# python ./tools/train.py  configs/roi_trans/roi-trans-le90_yolov8_yolopafpn_1x_dota.py --work-dir /mnt/mydisk/code/total_work_result/roi_yolo_result/ --resume

'''exp_module_train'''
# configs/s2anet/s2anet-le90_r50_fpn_1x_dota.py --work-dir /mnt/mydisk/code/total_work_result/train_work_result/
# configs/s2anet/s2anet-le90_yolo_1x_dota.py --work-dir /mnt/mydisk/code/total_work_result/train_work_result/
# configs/s2anet/exp_module/s2anet-le90_yolo_exneck_4head_1x_dota.py --work-dir /mnt/mydisk/code/total_work_result/train_work_result/
# configs/s2anet/exp_module/s2anet-le90_yolo_exlr_4head_1x_dota.py --work-dir /mnt/mydisk/code/total_work_result/train_work_result/
# configs/s2anet/exp_module/s2anet-le90_yolo_exbackbone_1x_dota.py --work-dir /mnt/mydisk/code/total_work_result/train_work_result/
# configs/s2anet/exp_module/s2anet-le90_yolo_nbup_1x_dota.py --work-dir /mnt/mydisk/code/total_work_result/train_work_result/
# configs/s2anet/exp_module/s2anet-le90_yolo_extracsp_1x_dota.py --work-dir /mnt/mydisk/code/total_work_result/train_work_result/
# configs/s2anet/exp_module/s2anet-le90_yolo_splithead_1x_dota.py --work-dir /mnt/mydisk/code/total_work_result/train_work_result/
# configs/s2anet/exp_module/s2anet-le90_yolo_simple_1x_dota.py --work-dir /mnt/mydisk/code/total_work_result/train_work_result/
# configs/s2anet/exp_module/s2anet-le90_yolo_extrafpn_aff_1x_dota.py --work-dir /mnt/mydisk/code/total_work_result/train_work_result/
# configs/s2anet/exp_module/s2anet-le90_yolo_exbackbone_extrafpn_1x_dota.py --work-dir /mnt/mydisk/code/total_work_result/train_work_result/
# configs/s2anet/exp_module/s2anet-le90_yolo_mscam_1x_dota.py --work-dir /mnt/mydisk/code/total_work_result/train_work_result/
# configs/s2anet/exp_module_2/s2anet-le90_yolo_simple_1x_dota.py --work-dir /mnt/mydisk/code/total_work_result/train_work_result/
# configs/s2anet/exp_module_2/s2anet-le90_yolo_simple_aff_1x_dota.py --work-dir /mnt/mydisk/code/total_work_result/train_work_result/
# configs/s2anet/exp_module_2/s2anet-le90_yolo_simple03_1x_dota.py --work-dir /mnt/mydisk/code/total_work_result/train_work_result/
# configs/s2anet/exp_module_2/s2anet-le90_yolo_simple14_1x_dota.py --work-dir /mnt/mydisk/code/total_work_result/train_work_result/
# configs/s2anet/exp_module_2/s2anet-le90_yolo_simple_WT_1x_dota.py --work-dir /mnt/mydisk/code/total_work_result/train_work_result/


if __name__ == '__main__':
    n = input("输入:")
    if n == 'xl':
        train()
    if n == 'fx':
        analysis()
    if n == 'cs':
        test()
