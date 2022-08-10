function [cost,grad] = HFCMCostW2(theta, visibleSize, hiddenSize, ...
                                             lambda, data, order, thetaW1, WFCM, W3)

% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
% W3 K*1
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

W1 = reshape(thetaW1(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
b1 = thetaW1(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
W2 = reshape(theta(1:hiddenSize*hiddenSize), hiddenSize, hiddenSize);
b2 = theta(hiddenSize*hiddenSize+1:end);

Wx = WFCM(hiddenSize+1:end-1,:);

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
cost = 0;
W2grad = zeros(size(W2)); 
b2grad = zeros(size(b2));


%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term 
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2 
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
% 
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 
% 

% Forward propagation
a0 = data;
a1 = sigmoid(W1*a0 + repmat(b1,1,size(a0,2))); % output of SAE
Y = data(:,order+1:end);
[a2,a2t] = DataforHFCM(a1,order); % output of training data and target
a3 = sigmoid(W2'*a2 + repmat(b2,1,size(a2,2)) + Wx'*a2t); % output of HFCM
atemp = [a3;a2];
a4 = W3'*atemp; % output of fianl layer
% Cost
cost = sum(sum((a4 - Y).^2)) + (lambda/2)*(sum(sum(W2.^2)) + sum(sum(Wx.^2)));

% yuan
W3H = W3(1:hiddenSize);
W3Z = W3(hiddenSize+1:end);
% yuan

% Backward propagation
del1 = -2*(Y - a4).*sum(a3.*(1-a3));
del2 = del1.*sum(a2.*(1-a2));
del3 = W2*W3H*(del2.*data(:,order:end-1));

d1 = [];
d2 = repmat(W3, hiddenSize, size(a3,2));
for i = 1:hiddenSize
    d1 = [d1; repmat(a2(i,:),hiddenSize,1)];
end
temp = repmat(del1,hiddenSize*hiddenSize,1).*(d1.*d2);
temp1 = reshape(sum(temp,2),hiddenSize,hiddenSize);
W2grad =  temp1'+ (lambda*W2);
b2grad = sum(W3*del1,2);


%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W2grad(:) ; b2grad(:)];

% % Visualize weights
% W1 = reshape(grad(1:hiddenSize * 784), hiddenSize, 784);
% display_network(W1');


end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end

