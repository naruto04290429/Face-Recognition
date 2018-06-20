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

lowerDimensionto = classNum - 1;

fixtoInverseEigen = 250; %250~400
fixtoInverse = 0.75;
U = zeros(lowerDimensionto, classNum);
C = zeros(lowerDimensionto, lowerDimensionto, classNum);
Wi = zeros(lowerDimensionto, lowerDimensionto, classNum);
wi = zeros(lowerDimensionto, classNum);
wio = zeros(1,classNum);

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

%% Process LDA
% Get mi, m 

mis = zeros(dimension, classNum);
for i = 1 : classNum
    temp = 0;
    for j = 1 : trainingSampleperClass
        temp = temp + trainData(:, trainingSampleperClass*(i-1)+j);
    end
    mis(:,i) = temp/trainingSampleperClass;
end

m = sum(trainData,2)/numofTrainingSample;

% Get SB
SB = 0;
for i = 1 : classNum
    Si = 2 * (mis(:, i) - m) * transpose(mis(:, i) - m);
    SB = SB + Si;
end

% Get SW
SW = 0;
for i = 1 : classNum
    Si = (trainData(:, 2*i-1) - mis(:, i)) * transpose(trainData(:, 2*i-1) - mis(:, i)) +...
         (trainData(:, 2*i) - mis(:, i)) * transpose(trainData(:, 2*i) - mis(:, i));
    SW = SW + Si;
end

% Find the corresponding eigenvector
%SB = SB + fixtoInverseEigen*eye(size(SB));
SW = SW + fixtoInverseEigen*eye(size(SW));
[W, LAMBDA] = eig(SB,SW);
lambda = diag(LAMBDA);
[lambda, SortOrder] = sort(lambda,'descend');
W = W(:,SortOrder);
eigenvectors = zeros(dimension, lowerDimensionto);
for j = 1: lowerDimensionto
        eigenvectors(:, j) = W(:, j);
end

% Get the data after transforamtion
trainData = transpose(eigenvectors) * trainData; 
testData = transpose(eigenvectors) * testData;

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