#import os
## Configuration - set the parameters in your framework
image_size = 512 # the image size to the network (image_size, image_size, 3)
batchsize = 2  # batch size
pad_total = 0.384
infer_mode = 'deploy' # 'train', 'test' 
shuffle_key =  False
num_workers_key =  0 # 0 for single-process loading methods, n for multi-process  
best_PCmodel_path  = "trained_models/ShoePoseClassificationResNet/PC_ResNet34_0.9812.pdparams" # pose classification model path 0.5*0.9812 + 0.5*0.9801
#best_PCmodel_path2 = ''  # set '' or None if only use a single model for classification
best_PCmodel_path2 = "trained_models/ShoePoseClassificationResNet/PC_ResNet34_0.9801.pdparams" # 0.5*0.9812 + 0.5*0.9801, 2 models for model ensemble
#best_KPmodel_path = "trained_models/8_3Heatmap_GResNet-Deepest_C32_no_TestSet_finetune_220315_1219/best_model_0.016625/model.pdparams" # keypoint model path
best_KPmodel_path = "trained_models/2DShoeKeypointDetection/2DKeypointNet_0.016625.pdparams"
### input imgs
#testset_root2 = "TestingSetSmall"
#test_filelists2 = [os.path.join(testset_root2,i) for i in os.listdir(testset_root2)]
test_filelists2 = ['TestingSet/IMG_20210302_151345.jpg', 'TestingSet/IMG_20211207_114000.jpg']
#test_filelists2 = ['TestingSet/IMG_20211207_141723.jpg', 'TestingSet/IMG_20211207_141952.jpg']
#test_filelists2 = ['TestingSet/IMG_20211207_142159.jpg', 'TestingSet/IMG_20211207_113758.jpg']
#test_filelists2 = ['TestingSet/IMG_20211207_143408.jpg', 'TestingSet/IMG_20211207_141637.jpg']
#test_filelists2 = ['TestingSet/IMG_20211207_144827_1.jpg', 'TestingSet/IMG_20210302_155456.jpg']