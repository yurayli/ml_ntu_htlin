% Initialization
clear all; close all; clc

% Load Data
data = load('hw4_nnet_train.dat');
testset = load('hw4_nnet_test.dat');
X = data(:, [1, 2]); y = data(:, 3);
Xtest = testset(:, [1, 2]); ytest = testset(:, 3);

% Setup the data matrix appropriately, and the training paramters
[m, n] = size(X);

% Add intercept term to X and X_test
X = [ones(m, 1) X];
Xtest = [ones(size(Xtest,1), 1) Xtest];
M = [n 8 3 1];  % n-8-3-1 ANN architecture, Eout avg = 0.0366
l = length(M);
r = .1;  		% weights randomly initialized in (-r,r)
eta = .01;  	% learning rate

% Neural network initialization
model = {};
for i = 1:l
	if i == 1
		model{i} = layer(m, r, i-1, M(i), 0);
	else
		model{i} = layer(m, r, i-1, M(i), M(i-1));
	end
end

% Backprop algorithm training and evaluation
Eout_arr = [];
tic
for exp = 1:50
	
	for iter = 1:5e4
		model = forward(X, model);
		model = backprop(y, eta, model);
	end
	% evaluation
	eval = forward(Xtest, model);
	H = eval{l}.output(:, 2:end);
	ypred = sign(H);
	ypred(ypred == 0) = 1;
	Eout = mean(double(ypred ~= ytest));
	Eout_arr = [Eout_arr Eout];

end
toc  % 318.177 sec in MATLAB, 2172.75 sec in Octave
fprintf('Averaged Eout: %d \n\n', mean(Eout_arr))  % avg Eout = 0.04

% To check convergence, calculate below in each iter
% H = model{length(M)}.output(:,2:end);
% Esq = mean((y - H).^2);


