clc; clear all;
%% Initailize variables
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

fixtoInverse = 0.75;
U = zeros(dimension, classNum);
C = zeros(dimension, dimension, classNum);
Wi = zeros(dimension, dimension, classNum);
wi = zeros(dimension, classNum);
wio = zeros(1,classNum);

%% First loop: Parse data

trainData = zeros(dimension, trainingSampleperClass, classNum);
for i = 1 : classNum
    for j = 1 : trainingSampleperClass
        trainData(:, j, i) = Data(:, j, i);
    end
end

%% Second loop: Calaulate ui, convi using MLE
for i = 1 : classNum
    % Accorind to Maximun Likelihood Estimation
    
    U_i = sum(trainData(:, :, i),2)/trainingSampleperClass;
    C_i = 0;
    for j = 1 : trainingSampleperClass
        C_i = C_i + (trainData(:, j, i) - U_i) * transpose(trainData(:, j, i) - U_i);
    end
    C_i = C_i/trainingSampleperClass;
    C_i = C_i + fixtoInverse*eye(size(C_i));
    U(:, i) = U_i;
    C(:, :, i) = C_i;
    
    % Quardratic Machine: Wi, wi, wip
    Wi(:, :, i) = (-1/2)*inv(C_i);
    wi(:, i) = inv(C_i)*U_i;
    wio(:, i) =  (-1/2)*( transpose(U_i)*inv(C_i)*U_i + log(det(C_i)) )+log(1/classNum); 
end
x = 0;
%% Third Loop: Run the gi(x) to classify

results = zeros(classNum,testingSampleperClass, 1);

for i = 1 : classNum
    max = intmin;
    for j = 1 + trainingSampleperClass : totalSampleperClass
        if choice == 1 
            X = reshape(pose(:, :, j, i) , [dimension,1]);
        else
            X = reshape(illum(:, j, i) , [dimension,1]);
        end
        for k = 1 : classNum
            G_k = transpose(X)*Wi(:, :, k)*X + transpose(wi(:, k))*X + wio(:, k);
            if G_k > max
                max = G_k;
                results(i, j -trainingSampleperClass , 1) = k;
            end
        end
    end
end

correct = 0;
for i = 1 : classNum
    for j = 1 : testingSampleperClass
        if(i == results(i,j,1))
            correct = correct + 1;
        end
    end
end
disp(correct/(classNum*testingSampleperClass));

