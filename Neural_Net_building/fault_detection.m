trainingSetup = load("D:\SRP_project\simple_examples" + ...
    "\params_2022_03_06__10_51_30.mat");

imdsTrain = imageDatastore("D:\SRP_project\simple_examples\dataset",...
    "IncludeSubfolders",true,"LabelSource","foldernames");

imageAugmenter = imageDataAugmenter(...
    "RandRotation",[0 20],...
    "RandScale",[1 10],...
    "RandXReflection",true,...
    "RandYReflection",true);

augimdsTrain = augmentedImageDatastore([840 594 3],imdsTrain,...
    "DataAugmentation",imageAugmenter);

opts = trainingOptions("sgdm",...
    "ExecutionEnvironment","auto",...
    "InitialLearnRate",0.01,...
    "Shuffle","every-epoch",...
    "Plots","training-progress",...
    "MiniBatchSize", 30);

layers = [
    imageInputLayer([840 594 3],"Name","imageinput")

    convolution2dLayer([30 30],32,"Name","conv_1","Padding","same","Stride",[15 15])
    batchNormalizationLayer("Name","batchnorm_1")
    reluLayer("Name","relu_1")
    maxPooling2dLayer([5 5],"Name","maxpool_1","Padding","same")

    convolution2dLayer([30 30],32,"Name","conv_3","Padding","same","Stride",[15,15])
    batchNormalizationLayer("Name","batchnorm_3")
    reluLayer("Name","relu_3")

    fullyConnectedLayer(5,"Name","fc")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")
    ];

[net, traininfo] = trainNetwork(augimdsTrain,layers,opts);


