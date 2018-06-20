clc; clear all;
load('..\Data\data.mat')

%% initail variables
classNum = 200;
dimension = 504;
lowerDimensionto = 200; 
numofTrainingSample = 400;
numofTestingSample = 200;

%% First loop: Process PCA
trainData = zeros(dimension, numofTrainingSample);
testData = zeros(dimension, numofTestingSample);
% Get mu of the data, and center the data
for i = 1 : classNum
    trainData(:, 2*i-1) = reshape(face(: , : , (3*i-2)),[dimension,1]);
    trainData(:, 2*i) = reshape(face(: , : , (3*i-1)),[dimension,1]);
    testData(:, i) = reshape(face(: , : , (3*i)),[dimension,1]);
end

UData = sum(trainData,2)/numofTrainingSample;

for i = 1 : numofTrainingSample
    trainData(:, i) = trainData(:, i) - UData;
end

for i = 1 : numofTestingSample
    testData(:, i) = testData(:, i) - UData;
end

% Form C head, dimension * dimension matrix
C_h =  trainData * transpose(trainData) /numofTrainingSample;

% D: eigenvalue, V: eigenvector
[V,D] = eig(C_h);
D = eig(C_h);
[D_sort D_index] = sort(D,'descend');
V_sort=V(:, D_index);

% Get the corresponding eigenvector
eigenvectors = zeros(dimension, lowerDimensionto);
for j = 1 : lowerDimensionto
        eigenvectors(:, j) = V_sort(:, j);
end

% Get the traing data after transformation
trainData = transpose(eigenvectors) * trainData; 
testData = transpose(eigenvectors) * testData; 

%% Process KNN
results = zeros(classNum,1);
distances = zeros(numofTrainingSample, classNum);
for i = 1 : classNum
    for j = 1 : numofTrainingSample
        distances(j, i) = sqrt(... 
        sum((testData(:, i) - trainData(:, j)).^2));
    end
    
    [sortDistance, sortPos] = sort(distances(:, i));
    
    %implement K = 1, 3, 5, 7, 9
    K = 1;
    nearestKth = zeros(K,1);
    for j = 1 : K
        if ( mod(sortPos(j), 2) == 1 )
            nearestKth(j, 1) = (sortPos(j)+1)/2;
        else
            nearestKth(j, 1) = sortPos(j)/2;
        end
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
    results(i, 1) = result;
end

correct = 0;
for i = 1 : classNum
    if(i == results(i,1))
        correct = correct + 1;
    end
end
disp(correct/classNum);
