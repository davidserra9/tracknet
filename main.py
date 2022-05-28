from AICity import AICity

if __name__== "__main__":
    aic = AICity(data_path="../../data/AICity_data/train/",
                 train_val_split=0,
                 train_seq=["S03", "S04"],
                 test_seq=["S01"],
                 tracking='deep_sort')

    # Train Faster R-CNN
    # aic.train_detectron2(model_yaml="COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml",
    #                      epochs=5000,
    #                      batch_size=2,
    #                      resume=False)

    # Create the object detection predictions
    # aic.infere_detectron2()

    # Track in a single camera from the Faster R-CNN predictions
    aic.sc_tracking()

    # Train the reid model (both finetuning the backbone and the triplet)
    #aic.train_reid(backbone='resnet50', backbone_epochs=5, triplet_epochs=25, batch_size=16, lr=0.001, finetune=True)

    # Re-ID the sequences from the tracking predictions
    #aic.multi_camera_reid(model_name='resnet50_finetune')