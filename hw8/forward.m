% Feeding data forward to get evaluation

function layers = forward(X, layers)
	
	l = length(layers);
	layers{1}.output = X;
	m = size(X, 1);
	for i = 2:l
		layers{i}.input = layers{i-1}.output * layers{i}.weight;
		layers{i}.output = tanh(layers{i}.input);
		layers{i}.output = [ones(m,1) layers{i}.output];
	end
	
end