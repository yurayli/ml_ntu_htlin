% Setting structure of layer for artificial neural network

function l_struc = layer(m, r, id, n_neurons, prev_neurons)
	
	f1 = 'id'; v1 = id;
	f2 = 'n_neurons'; v2 = n_neurons;
	% initialization
	f3 = 'input'; v3 = zeros(m, n_neurons);
	f4 = 'output'; v4 = zeros(m, n_neurons);
	f5 = 'error'; v5 = zeros(m, n_neurons);
	f6 = 'weight'; v6 = -r + 2*r*rand(prev_neurons+1, n_neurons);
	%
	l_struc = struct(f1,v1,f2,v2,f3,v3,f4,v4,f5,v5,f6,v6);
	
end