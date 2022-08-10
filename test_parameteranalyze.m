% multivariable time series prediction

clc
clear all
close all

%% Initialize Deep Network Parameters

% sparsityParam = 0.1;   % desired average activation of the hidden units.  % corresponding to paper parameters q   

inputZeroMaskedFraction   = 0.0;  % denoising ratio
dropoutFraction  = 0.0;          % dropout ratio

%% Load data from the MNIST database
% load('temperature_zscore.csv');
% InputIndex = [2:17, 21, 22, 23];
% data = temperature_zscore(:,InputIndex);

% load('traffic.csv');
% InputIndex = [2:7];                       %  [2:7];
% data = traffic(:,InputIndex);

load('weather.csv');
data = weather(:, 2:10);

% load('electricity_zscore.csv');
% InputIndex = [1:7];
% data = electricity_zscore(:,InputIndex);

% load('sz_speed.csv')
% data1 = sz_speed;
% data = data1(2:2977, 1:6);



inputSize = size(data,2);
% minmaxnorm
mindata = min(data, [], 1);
maxdata = max(data, [] ,1);
data = ((data-repmat(mindata, size(data, 1), 1))./(repmat(maxdata, size(data, 1), 1)-repmat(mindata, size(data, 1), 1)));

%% STEP 2: Train sparse autoencoder
lambda1V = [10,1,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10];
lambdaV = [1,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8];
betaV = [0.01,0.1,0.5,1:15];
NumSAEset = [2]; % [1,2,3, 4];
NumHFCMset = [2]; %[1,2,3, 4];
% NumHFCM = 2;
% NumSAE = 2;
orderV = 2:1:3;

HiddensizeV = [10:5:100];

RMSE = [];
for m = 1:1 %length(orderV) % L
for i = 3:3 %length(HiddensizeV) % k5
for j = 2:2 % tao7
for l = 13:13 % alpha
for g = 2:2%length(lambdaV) % beta
for k = 8:8 %lambda
for r1 = 1: 1
for r2 = 1: 1
    
% C = (1e-25)*(10^(k+4));
C = (10^(-k-2));          % corresponding to paper parameters lamda
CC(k) = C;
% C = (1e-8)*(10^k);
% % the parameter need be optimized
hiddenSize = HiddensizeV(i);      % corresponding to paper parameters k
lambda = lambdaV(g);            % corresponding to paper parameters beta
beta = betaV(l);               % corresponding to paper parameters alpha
lambda1 = lambda1V(j);        % corresponding to paper parameters simga
order = orderV(m);   % corresponding to paper parameters L
sparsityParam = 0.1;   % desired average activation of the hidden units.  % corresponding to paper parameters q
% Randomly initialize the parameters

% the ratio for training
datasize = size(data);
numratio = floor(0.7 * datasize(1));
% ceil(ratio*numsample)

trainData = data(1:numratio,:)';
testData = data(numratio-order:end,:)';

% train and test dimension for temperature outputs
trainLabels = trainData(:, order+1:end)';
testLabels = testData(:, order+1:end)';

NumSAE = NumSAEset(r1);
NumHFCM = NumHFCMset(r2); % 

SAEFeaturesTotal = [];
SAEFeaturesTestTotal = [];
thetaW2 = [];
for seed = 1 : NumSAE
    % seed = 1;
    theta = initializeParameters_nonneg(hiddenSize, inputSize, seed);

    %  Use minFunc to minimize the function
    addpath minFunc/
    options.Method = 'lbfgs';
    options.maxIter = 400;
    options.display = 'on';

    [opttheta, cost] = minFunc( @(p) sparseAutoencoderCost_nonneg(p, ...
                                       inputSize, hiddenSize, ...
                                       lambda, inputZeroMaskedFraction,...
                                       dropoutFraction, sparsityParam, ...
                                       beta, trainData), ...
                                       theta, options);
    W1 = reshape(opttheta(1:hiddenSize*inputSize), hiddenSize, inputSize);
    b1 = opttheta(2*hiddenSize*inputSize+1:2*hiddenSize*inputSize+hiddenSize);
    thetaW2 = [thetaW2; W1(:); b1(:)];
    [saeFeatures] = feedForwardAutoencoder(opttheta, hiddenSize, ...
                                            inputSize, dropoutFraction, trainData);
    [saeFeaturesTest] = feedForwardAutoencoder(opttheta, hiddenSize, ...
                                        inputSize, dropoutFraction, testData);
    SAEFeaturesTotal = [SAEFeaturesTotal; saeFeatures];
    SAEFeaturesTestTotal = [SAEFeaturesTestTotal; saeFeaturesTest];
end

[a2,a2t] = DataforHFCM(SAEFeaturesTotal,order); % output of training data and target
%% Construct high-order FCMs
atemp = [];
atemp = [atemp, a2'];
% % Randomly initialize the parameters;
allWFCM = {};
for seed = NumSAE + 1: NumSAE + NumHFCM
    [a2,a2t] = DataforHFCM(SAEFeaturesTotal,order);
    rand('state',seed)
    WFCM = rand(hiddenSize*NumSAE*order,hiddenSize*NumSAE)*0.1;
    WFCM = [WFCM;zeros(1,hiddenSize*NumSAE)];
    allWFCM{1, seed - NumSAE} = WFCM;

    %% Establish the model of output

    % the output of HFCM
    W2 = WFCM(1:hiddenSize*NumSAE,:);
    b2 = WFCM(end,:);
    Wx = WFCM(hiddenSize*NumSAE+1:end-1,:);

   
    a3 = sigmoid(W2'*a2 + repmat(b2',1,size(a2,2)) + Wx'*a2t); % output of HFCM
    atemp = [atemp, a3'];
end
% the weight matrix of output
W3 = pinv(atemp'*atemp + C)*atemp'*trainLabels;
%% Fine-tuning AE,HFCM

for index = 1 : NumHFCM % defaults NumHFCM
    WFCM = [];
    WFCM = allWFCM{1, index};
    W2 = WFCM(1:hiddenSize*NumSAE,:);
    b2 = WFCM(end,:);
    Wx = WFCM(hiddenSize*NumSAE+1:end-1,:);
    thetaW2 = [thetaW2; W2(:) ; Wx(:) ; b2(:)];
end
thetaW2 = [ thetaW2; W3(:)];
    % W2
options.Method = 'lbfgs';
options.maxIter = 400; % defaults 400	  
options.display = 'on';

dbstop if error
% [OptThetaW2,cost1] = minFunc( @(p) HFCMCostW2x(p, NumSAE, NumHFCM, SAEFeaturesTotal, hiddenSize, ...
%                                          lambda1, trainLabels, order, ...
%                                  W3, allWFCM),thetaW2, options);
% NumallWFCM = size(allWFCM, 2);
% UnitLength = size(OptThetaW2, 1)/NumallWFCM;
% for index = 1 : NumHFCM % defaults NumHFCM
%     InnerTheta = OptThetaW2(1 + (index-1)* UnitLength: index* UnitLength);
%     WFCM(1:hiddenSize*NumSAE,:) = reshape(InnerTheta(1:hiddenSize*NumSAE*hiddenSize*NumSAE),...
%                                                   hiddenSize*NumSAE, hiddenSize*NumSAE);
% 
%     WFCM(end,:) = InnerTheta(hiddenSize*NumSAE*hiddenSize*NumSAE*order+1:end)';
% 
%     WFCM(hiddenSize*NumSAE+1:end-1,:) = reshape(InnerTheta(hiddenSize*NumSAE*hiddenSize*NumSAE+1:hiddenSize*NumSAE*hiddenSize*NumSAE*order),...
%         hiddenSize*NumSAE*(order-1), hiddenSize*NumSAE);
%     allWFCM{1, index} = WFCM;

% [OptThetaW2,cost1, exitflag,outputstructure] = minFunc( @(p) HFCMCostW123X(p, NumSAE, NumHFCM, trainData, hiddenSize, ...
%                                          lambda1, trainLabels, order),thetaW2, options);
OptThetaW2 = thetaW2;                               
NumSamples = size(trainData, 2);
Inputlength = size(trainData, 1);
W1b1length = (Inputlength + 1)* hiddenSize;

allSAEvector = [];
for index = 1: NumSAE
    W1b1 = OptThetaW2( 1 + (index - 1) * W1b1length: index * W1b1length);
    allSAEvector{1, index} = W1b1;
end
Zlength = size(SAEFeaturesTotal, 1);
W3vectorlength = Zlength* (1 + NumHFCM)* size(trainLabels, 2);
UnitLength = (size(OptThetaW2, 1)-W3vectorlength - W1b1length * NumSAE)/NumHFCM;
for index = 1 : NumHFCM 
    allWFCMvector{1, index} = OptThetaW2(NumSAE * W1b1length +  1 + (index-1)* UnitLength:NumSAE * W1b1length + index* UnitLength);
end
W3vector = OptThetaW2(W1b1length * NumSAE + NumHFCM * UnitLength + 1: end);
W3 = reshape(W3vector, Zlength * (1 + NumHFCM), size(trainLabels, 2));


%% the optimized W3
% % the output of HFCM
% W2 = WFCM(1:hiddenSize,:);
% b2 = WFCM(end,:);
% Wx = WFCM(hiddenSize+1:end-1,:);
% 
% % [a2,a2t] = DataforHFCM(saeFeatures,order); % output of training data and target
% a3 = sigmf(W2'*a2 + repmat(b2',1,size(a2,2)) + Wx'*a2t, [1 0]); % output of HFCM
% atemp = [a3',a2'];

%% Test

% [R(1,k)] = AEHFCM_predict(opttheta, hiddenSize, inputSize, ...
%             dropoutFraction, trainData, WFCM, trainLabels, W3, order, [mindata,maxdata]);

% [R(1,j)] = AEHFCM_predict(SAEFeaturesTestTotal, NumSAE, NumHFCM, hiddenSize,...
%             allWFCM, testLabels, W3, order, [mindata, maxdata]);
        
R{r1, r2}= AEHFCM_predict(testData, NumSAE, NumHFCM, hiddenSize,...
            allSAEvector, allWFCMvector, testLabels, W3, order, [mindata; maxdata]);

% [R(1,k)] = AEHFCM_predict(opttheta, hiddenSize, inputSize, ...
%             dropoutFraction, trainData, WFCM, trainLabels, W3, order, [mindata,maxdata]);

% pp = pp+1;
% end
% plot(outfinal,'r-*');
% hold on
% plot(testLabels,'b-.');
% legend('predicted data','real data');
% bar(R)
% xlabel('\delta');
% ylabel('RMSE');
% set(gca, 'XTicklabel',10:5:55);
% fprintf('After Finetuning Test Accuracy: %0.3f%%\n', acc_after(seed) * 100);
end
% RMSE = [RMSE ; R];
end
end
end
end
end
end
end


function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end