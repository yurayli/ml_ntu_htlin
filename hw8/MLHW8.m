%%% Implement and experiment the ARTIFICIAL NEURAL NETWORK with MATLAB


% Initialization
clear all; close all; clc

part = 3;

%% Part 1: check convergence of ANN
if part == 1
	
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
M = 6;  	% d-M-1 ANN architecture
r = .1;  	% weights randomly initialized in (-r,r)
eta = .1;   % learning rate

% Backprop algorithm
w1 = -r + 2*r*rand(n+1, M);
w2 = -r + 2*r*rand(M+1, 1);
Esq_arr = [];
tic
for iter = 1:5e4
	%
	hidden = tanh(X * w1);
	hidden = [ones(m,1) hidden];
	H = tanh(hidden * w2);
	Esq = mean((y - H).^2);
	Esq_arr = [Esq_arr Esq];
	%
	delH = -2 * (y-H) .* (1-H.^2);
	grad2 = hidden' * delH;
	delHidden = (delH * w2') .* (1 - hidden.^2);
	grad1 = X' * delHidden(:, 2:end);
	%
	w1 = w1 - eta * grad1;
	w2 = w2 - eta * grad2;
end
toc
% Check convergence
figure
plot(1:5e4, Esq_arr)
title('Check Convergence')
xlabel('iteration')
ylabel('E_{sq}')
% Evaluation
hidden = tanh(X * w1);
hidden = [ones(m,1) hidden];
H = tanh(hidden * w2);
ypred = sign(H);
ypred(ypred == 0) = 1;
accu = mean(double(ypred == y)) % accuracy

end



% --------------------------------------------------------------------
%% Part 2: calculate avg Eout with training parameters
if part == 2

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
M = 3;
% M = [1, 6, 11, 16, 21]
% Eout avg = 0.2722 for M = 1, r = .1, eta = .1
% Eout avg = 0.1279 for M = 6, r = .1, eta = .1
% Eout avg = 0.1732 for M = 11, r = .1, eta = .1
% Eout avg = 0.1787 for M = 16, r = .1, eta = .1
% Eout avg = 0.2080 for M = 21, r = .1, eta = .1
r = .1;
% r = [0, .1, 10, 100, 1000]
% Eout avg = 0.0788 for M = 3, r = .1, eta = .1
% Eout avg = 0.3045 for M = 3, r = 10, eta = .1
% Eout avg = 0.4931 for M = 3, r = 100, eta = .1
eta = .01;
% eta = [.001 .01 .1 1 10]
% Eout avg = 0.0360 for M = 3, r = .1, eta = .001
% Eout avg = 0.0365 for M = 3, r = .1, eta = .01
% Eout avg = 0.0788 for M = 3, r = .1, eta = .1
% Eout avg = 0.4999 for M = 3, r = .1, eta = 1
% Eout avg = 0.4724 for M = 3, r = .1, eta = 10

% Backprop algorithm
Eout_arr = [];
tic
for exp = 1:500
	w1 = -r + 2*r*rand(n+1, M);
	w2 = -r + 2*r*rand(M+1, 1);
	for iter = 1:50000
		%
		hidden = tanh(X * w1);
		hidden = [ones(m,1) hidden];
		H = tanh(hidden * w2);
		%
		delH = -2 * (y-H) .* (1-H.^2);
		grad2 = hidden' * delH;
		delHidden = (delH * w2') .* (1 - hidden.^2);
		grad1 = X' * delHidden(:, 2:end);
		%
		w1 = w1 - eta * grad1;
		w2 = w2 - eta * grad2;
	end
	
	% calculate Eout from the test set
	hidden = tanh(Xtest * w1);
	hidden = [ones(size(Xtest,1),1) hidden];
	H = tanh(hidden * w2);
	ypred = sign(H);
	ypred(ypred == 0) = 1;
	Eout = mean(double(ypred ~= ytest));
	Eout_arr = [Eout_arr Eout];
end
toc

end



% --------------------------------------------------------------------
%% Part 3: deepen the ANN architecture
if part == 3

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
M1 = 8;  	% d-M1-M2-1 ANN architecture
M2 = 3;
r = .1;  	% weights randomly initialized in (-r,r)
eta = .01;  % learning rate

% Backprop algorithm
Eout_arr = [];
tic
for exp = 1:50
	w1 = -r + 2*r*rand(n+1, M1);
	w2 = -r + 2*r*rand(M1+1, M2);
	w3 = -r + 2*r*rand(M2+1, 1);
	for iter = 1:50000
		%
		hidden1 = tanh(X * w1);
		hidden1 = [ones(m,1) hidden1];
		hidden2 = tanh(hidden1 * w2);
		hidden2 = [ones(m,1) hidden2];
		H = tanh(hidden2 * w3);
		%
		delH = -2 * (y-H) .* (1-H.^2);
		delHidden2 = (delH * w3') .* (1 - hidden2.^2);
		delHidden1 = (delHidden2(:, 2:end) * w2') .* (1 - hidden1.^2);
		grad3 = hidden2' * delH;
		grad2 = hidden1' * delHidden2(:, 2:end);
		grad1 = X' * delHidden1(:, 2:end);
		%
		w1 = w1 - eta * grad1;
		w2 = w2 - eta * grad2;
		w3 = w3 - eta * grad3;
	end
	
	% calculate Eout from the test set
	hidden1 = tanh(Xtest * w1);
	hidden1 = [ones(size(Xtest,1),1) hidden1];
	hidden2 = tanh(hidden1 * w2);
	hidden2 = [ones(size(Xtest,1),1) hidden2];
	H = tanh(hidden2 * w3);
	ypred = sign(H);
	ypred(ypred == 0) = 1;
	Eout = mean(double(ypred ~= ytest));
	Eout_arr = [Eout_arr Eout];
end
toc  % 92.237 sec in MATLAB R2016a, 452.447 sec in Octave
fprintf('Averaged Eout: %d \n\n', mean(Eout_arr))  % avg Eout = 0.03632

end
