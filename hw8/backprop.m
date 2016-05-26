% Backprop for training the ann with gradient descent of each weight of layer

function layers = backprop(y, eta, layers)
	
	l = length(layers);
	H = layers{l}.output(:,2:end);
	layers{l}.error = -2 * (y - H) .* (1-H.^2);
	grad = layers{l-1}.output' * layers{l}.error;
	layers{l}.weight = layers{l}.weight - eta * grad;
	for i = (l-1):-1:2
		
		if i == (l-1)
			layers{i}.error = (layers{i+1}.error * layers{i+1}.weight') .* (1 - layers{i}.output.^2);
		else
			layers{i}.error = (layers{i+1}.error(:, 2:end) * layers{i+1}.weight') .* (1 - layers{i}.output.^2);
		end
		grad = layers{i-1}.output' * layers{i}.error(:, 2:end);
		layers{i}.weight = layers{i}.weight - eta * grad;
		
	end
	
end