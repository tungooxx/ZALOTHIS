python guided_track_pt_multi.py \
  --weights /path/to/best.pt \
  --source  /path/to/DATA_ROOT \
  --imgsz 640 --conf 0.20 --sim_thr 0.28 --topk 48 \
  --clip_batch 24 \
  --save_video --bench


/chucky/PycharmProjects/ZALOTHIS/public_test/samples/BLackBox_0/object_images  --context 0.05  --save_video --imgsz 640 --conf 0.2 --iou 0.45   --topk 120 --sim_thr 0.39 --seg_exemplars  --seg_mode off --seg_weight yolo11n-seg.pt   --clip_batch 32  --exemplar_mode per --infer track

python debug.py \
  --weights best_yolo11_960.pt \
  --root /home/chucky/PycharmProjects/ZALOTHIS/public_test/samples \
  --choose 1 \
  --save_video --save_context \
  --imgsz 640 --conf 0.2 --sim_thr 0.42 --topk 100 \
  --context 0.03 \
  --seg_mode grabcut --seg_inset 0.12 --seg_grabcut_iter 2 --exemplar_seg_inset 0.06 \
  --save_mask --mask_overlay --mask_stride 1 \
  --color_gate off



python guided_track_pt.py   --weights best_yolo11_960.pt   --source /home/chucky/PycharmProjects/ZALOTHIS/public_test/samples/BlackBox_0/drone_video.mp4   --object_images /home \

python guided_track_pt_true.py     --root /home/chucky/PycharmProjects/ZALOTHIS/public_test     --weights best_yolo11_960.pt     --seg_weights yolo11n-seg.pt     --context_pad 0.05     --save_video     --imgsz 640     --conf 0.2     --iou 0.45     --topk 120     --sim_thr 0.39     --seg_exemplars     --seg_mode off     --clip_batch 32     --exemplar_mode per     --infer track \

python eval_video_SM.py   --checkpoint checkpoints_SM3/siamese_mbv3_epoch26.pth   --support_dir /home/chucky/PycharmProjects/ZALOTHIS/public_test/samples/CardboardBox_1/object_images   --video_in /home/chucky/PycharmProjects/ZALOTHIS/public_test/samples/CardboardBox_0/drone_video.mp4   --video_out debug_out.mp4   --score_thresholds 0.1,0.5,0.85   --device cuda   --resize_short_side 800