clc; clear all;
%% initail variables
classNum = 68;
dimension = 48*40;

choice = 1;
if choice == 1
    load('..\Data\pose.mat'); %each image: 48*40, pose: 48x40x13x68
    totalSampleperClass = 13;
    trainingSampleperClass = 11;
    Data = reshape(pose, [dimension, totalSampleperClass, classNum]);
else
    load('..\Data\illumination.mat'); %each image: 48*40, pose: 48x40x21x68
    totalSampleperClass = 21;
    trainingSampleperClass = 16;
    Data = reshape(illum, [dimension, totalSampleperClass, classNum]);
end

testingSampleperClass = totalSampleperClass - trainingSampleperClass;
numofTrainingSample = trainingSampleperClass*classNum;
numofTestingSample = testingSampleperClass*classNum;

%% First loop: Parse data
trainData = zeros(dimension, trainingSampleperClass, classNum);
testData = zeros(dimension, testingSampleperClass, classNum);
for i = 1 : classNum
    for j = 1 : trainingSampleperClass
        trainData(:, j, i) = Data(:, j, i);
    end
    for j = 1 + trainingSampleperClass : totalSampleperClass
        testData(:, j-trainingSampleperClass, i) = Data(:, j, i);
    end
end
trainData = reshape(trainData, [dimension, numofTrainingSample]);
testData = reshape(testData, [dimension, numofTestingSample]);

%% Process KNN
distances = zeros(numofTrainingSample, classNum);
results = zeros(classNum,testingSampleperClass, 1);

for i = 1 : numofTestingSample
    for j = 1 : numofTrainingSample
        distances(j, i) = sqrt(... 
        sum((testData(:, i) - trainData(:, j)).^2));
    end
    
    [sortDistance, sortPos] = sort(distances(:, i));
    
    %implement K = 1, 3, 5, 7, 9
    K = 1;
    nearestKth = zeros(K,1);
    for j = 1 : K
        nearestKth(j, 1) = ceil(sortPos(j)/trainingSampleperClass);
    end
    
    table = tabulate(nearestKth);
    [F,I] = max(table(:,2));
    I = find(table(:,2)==F);
    result = table(I,1);
    % result might be several elements
    if(size(result,1) > 1)
        x = result(1);
        for j = 2 : size(result)
            if( x < find(sortPos==result(j)))
                x = result(j);
            end
        end
        result = x;
    end
    if (mod(i, testingSampleperClass)==0)
        tmp = testingSampleperClass;
    else
        tmp = mod(i, testingSampleperClass);
    end
    results(ceil(i/testingSampleperClass), tmp, 1) = result;
end

correct = 0;
for i = 1 : classNum
    if(i == results(i,1))
        correct = correct + 1;
    end
end
disp(correct/classNum);
