% function [cost,grad] = HFCMCostW2x(theta, visibleSize, hiddenSize, ...
%                                              lambda, data, order, thetaW1, W3)
function [cost,grad] = HFCMCostW23X(theta, NumSAE, NumHFCM, SAEFeaturesTotal, hiddenSize, ...
                                             lambda, Labels, order)

NumSamples = size(SAEFeaturesTotal, 2);
Zlength = size(SAEFeaturesTotal, 1);
W3vectorlength = Zlength* (1 + NumHFCM)* size(Labels, 2);

UnitLength = (size(theta, 1)-W3vectorlength)/NumHFCM;
for index = 1 : NumHFCM % defaults NumHFCM
    allWFCMvector{1, index} = theta(1 + (index-1)* UnitLength: index* UnitLength);
end
W3vector = theta(NumHFCM * UnitLength + 1: end);
W3 = reshape(W3vector, Zlength * (1 + NumHFCM), size(Labels, 2));
cost = 0;
a1 = SAEFeaturesTotal;
Y = Labels';
[a2,a2t] = DataforHFCM(a1,order); % output of training data and target
atemp = [];
atemp = [atemp; a2];
for index = 1 : NumHFCM 
    thetaInner = allWFCMvector{1, index};
    W2 = reshape(thetaInner(1:hiddenSize*NumSAE*hiddenSize*NumSAE), hiddenSize*NumSAE, hiddenSize*NumSAE);
    b2 = thetaInner(hiddenSize*NumSAE*hiddenSize*NumSAE*order+1:end);
    Wx = reshape(thetaInner(hiddenSize*NumSAE*hiddenSize*NumSAE+1:hiddenSize*NumSAE*hiddenSize*NumSAE*order), hiddenSize*NumSAE*(order-1), hiddenSize*NumSAE);
    a3inner = sigmoid(W2'*a2 + repmat(b2,1,size(a2,2)) + Wx'*a2t); % output of HFCM
    atemp = [atemp; a3inner];
    
end
a4 = W3'*atemp; % output of fianl layer
cost = sum(sum((a4 - Y).^2))/length(Y) + (lambda/2)*(sum(sum(W2.^2)) + sum(sum(Wx.^2)));
fprintf("%f\n", cost);

% Backward propagation
grad = [];
for index = 1 : NumHFCM 
    thetaInner = allWFCMvector{1, index};
    W2 = reshape(thetaInner(1:hiddenSize*NumSAE*hiddenSize*NumSAE), hiddenSize*NumSAE, hiddenSize*NumSAE);
    b2 = thetaInner(hiddenSize*NumSAE*hiddenSize*NumSAE*order+1:end);
    Wx = reshape(thetaInner(hiddenSize*NumSAE*hiddenSize*NumSAE+1:hiddenSize*NumSAE*hiddenSize*NumSAE*order), hiddenSize*NumSAE*(order-1), hiddenSize*NumSAE);
%     W2grad = zeros(size(W2)); 
%     b2grad = zeros(size(b2));
%     Wxgrad = zeros(size(Wx));
    a3inner = sigmoid(W2'*a2 + repmat(b2,1,size(a2,2)) + Wx'*a2t); % output of HFCM
    atemp = [atemp; a3inner];
    W3Hindex = W3(hiddenSize*NumSAE*index + 1:hiddenSize*NumSAE*(index + 1), :);
    
    del2 = -2*(Y - a4).*sum(a3inner.*(1-a3inner))/length(Y);
    del1 = [];
    for i = 1: size(del2, 1)
        del1 = [del1, del2(i, :)];
    end
    d2 = [];
    for i = 1: size(W3Hindex, 2)
        d2 = [d2, repmat(W3Hindex(:, i), hiddenSize*NumSAE, size(a3inner,2))];
    end
    d1 = [];
    for i = 1:hiddenSize*NumSAE        
        d1 = [d1; repmat(a2(i,:),hiddenSize*NumSAE, size(Y, 1))];
    end

    temp = repmat(del1,hiddenSize*NumSAE*hiddenSize*NumSAE,1).*(d1.*d2);
    temp1 = reshape(sum(temp,2),hiddenSize*NumSAE,hiddenSize*NumSAE);
    W2grad =  temp1'+ (lambda*W2);
    b2grad = sum(W3Hindex*del2,2);

    Wxgrad = [];
    for i = 1:order-1
        at = a2t((i-1)*hiddenSize*NumSAE+1:i*hiddenSize*NumSAE,:);
        d2 = [];
        for j = 1: size(W3Hindex, 2)
            d2 = [d2, repmat(W3Hindex(:, j), hiddenSize*NumSAE, size(a3inner,2))];
        end
        d1 = [];
        for j = 1:hiddenSize*NumSAE
            d1 = [d1; repmat(at(j,:),hiddenSize*NumSAE, size(Y, 1))];
        end
        temp = repmat(del1,hiddenSize*NumSAE*hiddenSize*NumSAE,1).*(d1.*d2);
        temp1 = reshape(sum(temp,2),hiddenSize*NumSAE,hiddenSize*NumSAE);
        Wxgrad = [Wxgrad; temp1'];
    end
    Wxgrad = Wxgrad + (lambda*Wx);
    

    grad = [grad; W2grad(:) ; Wxgrad(:) ; b2grad(:)];
end
W3grad = zeros(size(W3));
for u = 1: size(W3grad, 1)
    for v = 1 : size(W3grad, 2)
        W3grad(u, v) = (Y(v, :)- a4(v, :))* atemp(u, :)';
    end
end
grad = [grad; W3grad(:)];

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
%% ---------- YOUR CODE HERE --------------------------------------

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%recode the BP of the WHFCMs%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% for p = 1 : (hiddenSize*NumSAE)
%     for q = 1 : hiddenSize*NumSAE
%         innerV = 0;
%         innerB = 0;
%         for i = 1 : size(Y, 1)
%             for j = 1 : size(Y, 2)
%                 for m = 1 : hiddenSize*NumSAE
%                     innerV = innerV + 2 * (a4(i, j)-Y(i, j)) * W3Hindex(m, i) * a3(m, j) * (1 - a3(m, j)) * a2(q, j);
%                     innerB = innerB + (a4(i, j)-Y(i, j)) * W3Hindex(m, i) * a3(m, j) * (1 - a3(m, j));
%                 end
%             end
%         end
%         W2grad(p, q) = innerV;
%         b2grad(p) = innerB;
%     end
% end
% W2grad =  W2grad + (lambda*W2);
% for p = 1 : hiddenSize*NumSAE
%     for q = 1 : hiddenSize*NumSAE * (order - 1)
%         innerV2 = 0;
%         for i = 1 : size(Y, 1)
%             for j = 1 : size(Y, 2)
%                 for m = 1 : hiddenSize*NumSAE
%                     innerV2 = innerV2 + 2 * (a4(i, j)-Y(i, j)) * W3Hindex(m, i) * a3(m, j) * (1 - a3(m, j)) * a2t(q, j);
%                 end
%             end
%         end
%         Wxgrad(q, p) = innerV;
%     end
% end
% Wxgrad = Wxgrad + (lambda*Wx);
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Wxgrad = Wxgrad;
end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end

