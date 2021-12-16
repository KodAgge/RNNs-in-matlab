% === Read data ===
[book_data, char_to_ind, ind_to_char, K] = getContainers();


% === Initialize network ===
m = 100;
eta = 0.05;
seq_length = 25;
seed = 1336;
sig = 0.01;
n_epochs = 10;
[RNN, Dim] = initializeNetwork(m, eta, seq_length, K, sig, seed);


% === Test of synthesizing
n = 25;
h0 = zeros(Dim.m, 1);
x0 = zeros(Dim.K, 1);
x0(1) = 1;
Y = synthesizeText(RNN, Dim, h0, x0, n);

oneHotToString(Y, ind_to_char)


% === Debugging ===
X_chars = book_data(1:seq_length);
Y_chars = book_data(2:seq_length+1);

X_chars = StringToOneHot(X_chars, char_to_ind, Dim.K);
Y_chars = StringToOneHot(Y_chars, char_to_ind, Dim.K);

[loss, A, H, O, P] = ComputeLoss(X_chars, Y_chars, RNN, Dim, h0);

grad = ComputeGradients(X_chars, Y_chars, RNN, Dim, A, H, P);

num_grad = ComputeGradsNum(X_chars, Y_chars, RNN, Dim, 1e-4);

tol = 1e-6;

for f = fieldnames(RNN)'
    f{1}
    [accuracy, max_error] = DifferenceGradients(grad.(f{1}), num_grad.(f{1}), tol)
end


% === Running gradient descent ===
RNN_trained = MiniBatchGD(book_data, RNN, Dim, char_to_ind, ind_to_char, seq_length, n_epochs)


function RNN = MiniBatchGD(book_data, RNN, Dim, char_to_ind, ind_to_char, seq_length, n_epochs)
    % Compute helpful variables
    e_values = 1:seq_length:(length(book_data) - seq_length - 1);
    n_updates = length(e_values) * n_epochs
    j = 1;
    smooth_loss = zeros([n_updates, 1]);

    % Initialize the adagrad object
    adagrad = GradientObject;
    adagrad.b = zeros(size(RNN.b));
    adagrad.c = zeros(size(RNN.c));
    adagrad.U = zeros(size(RNN.U));
    adagrad.W = zeros(size(RNN.W));
    adagrad.V = zeros(size(RNN.V));
    
    % Main loop
    for i = 1:n_epochs
        epoch = i
        h0 = zeros([Dim.m, 1]);
        h_prev = h0;
        
        for e = e_values
            % Extract data
            X_batch_chars = book_data(e:(e + seq_length - 1));
            Y_batch_chars = book_data((e + 1):(e + seq_length));

            % Transform to one hot vectors
            X_batch = StringToOneHot(X_batch_chars, char_to_ind, Dim.K);
            Y_batch = StringToOneHot(Y_batch_chars, char_to_ind, Dim.K);
            
            % Compute loss
            [loss, A, H, ~, P] = ComputeLoss(X_batch, Y_batch, RNN, Dim, h_prev);
            
            % Compute gradients
            grad = ComputeGradients(X_batch, Y_batch, RNN, Dim, A, H, P);
            
            % Clip gradients, compute adagrads & update variables
            for f = fieldnames(RNN)'
                
                % Clip the gradients
                grad.(f{1}) = max(min(grad.(f{1}), 5), -5);
                
                % Adagrad calculation
                adagrad.(f{1}) = adagrad.(f{1}) + grad.(f{1}).^2;   % Eq 6
                
                % Update variable
                RNN.(f{1}) = RNN.(f{1}) - (Dim.eta * grad.(f{1})) ./ ...
                    ((adagrad.(f{1}) + eps).^0.5);                  % Eq 7
            end
            
            % Calculate smooth loss
            if j == 1
                smooth_loss(j) = loss;
            else
                smooth_loss(j) = 0.999 * smooth_loss(j - 1) + 0.001 * loss;
            end
            
            % Printing
            if j == 1 || mod(j, 5000) == 0
                % Print helpful values
                j
                smooth_loss(j)
                
                % Synthesize text
                if j == 1 || mod(j, 10000) == 0
                    synthesizedY = synthesizeText(RNN, Dim, h_prev, X_batch(:, 1), 200);
                    oneHotToString(synthesizedY, ind_to_char)
                end
            end
            
            % For next iteration
            j = j + 1;
            h_prev = H(:, end);
        end
    end
    
    % Plotting the smooth loss
    plot(smooth_loss, 'color', [0, 0.5, 0])
    hold all
    xlabel('update step')
    ylabel('smooth loss')
    title('Smooth loss for 3 epochs of training')
end

function [RNN, Dim] = initializeNetwork(m, eta, seq_length, K, sig, seed)
    % Set hyperparameters
    Dim = DimensionObject;
    Dim.m = m;
    Dim.K = K;
    Dim.eta = eta;
    Dim.seq_length = seq_length;
    
    % Initialize bias vectors
    rng(seed)
    RNN = RNNObject;
    RNN.b = zeros([m, 1]);
    RNN.c = zeros([K, 1]);
    
    % Initialize weight matrices
    RNN.U = randn(m, K)*sig;
    RNN.W = randn(m, m)*sig;
    RNN.V = randn(K, m)*sig;
end

function [book_data, char_to_ind, ind_to_char, K] = getContainers()
    % Read book
    book_fname = 'data/goblet_book.txt';
    fid = fopen(book_fname,'r');
    book_data = fscanf(fid,'%c');
    fclose(fid);

    % Find dimensions and unique characters
    book_chars = unique(book_data);
    K = numel(book_chars);

    % Create containers
    char_to_ind = containers.Map('KeyType','char','ValueType','int32');
    ind_to_char = containers.Map('KeyType','int32','ValueType','char');
    for i = 1:K
       char_to_ind(book_chars(i)) = i;
       ind_to_char(i) = book_chars(i);
    end
end

function Y = synthesizeText(RNN, Dim, h0, x0, n)
    % Initialize
    Y = zeros([Dim.K, n]);
    h_t = h0;
    x_t = x0;

    % Implements equations 1-4
    for t = 1:n
        a_t = RNN.W * h_t + RNN.U * x_t + RNN.b;    % Eq 1
        h_t = tanh(a_t);                            % Eq 2
        o_t = RNN.V * h_t + RNN.c;                  % Eq 3
        p_t = softMax(o_t);                         % Eq 4

        % Sample character
        x_next = sampleCharacter(p_t, Dim.K);
        
        % Save in matrix
        Y(:,t) = x_next;
        
        % Update x_t
        x_t = x_next;
    end
end

function character = sampleCharacter(p, K)
    % Samples a character using suggested code in instructions
    cp = cumsum(p);
    r = rand;
    ixs = find(cp - r >0);
    ii = ixs(1);
    character = oneHotEncoder(ii, K); % One hot encodes the character
end

function Y = oneHotEncoder(y, C)
    % One hot encodes a vector/matrix given the number of classes
    Y = (y==(1:C))';
end

function p = softMax(s)
    % Softmax function
    p = exp(s) ./ sum(exp(s));
end

function str = oneHotToString(Y, ind_to_char)
    str = [];
    [~, indices] = max(Y); % Finds indices of the 1's
    
    % Appends the indices into a list
    for i = 1:length(indices)
       str = [str ind_to_char(indices(i))];
    end
end

function oneHot = StringToOneHot(Y, char_to_ind, K)
    % Pre allocate memory
    y = zeros(length(Y), 1);
    
    % Translates the character to its index
    for i = 1:length(Y)
        y(i) = char_to_ind(Y(i));
    end
    
    % One hot encodes the vector
    oneHot = oneHotEncoder(y, K);
end

function [loss, A, H, O, P] = ComputeLoss(X, Y, RNN, Dim, h0)
    n = size(X, 2);
    h_t = h0;
    
    % Pre allocate memory
    A = zeros([Dim.m, n]);
    H = zeros([Dim.m, n]);
    O = zeros([Dim.K, n]);
    P = zeros([Dim.K, n]);
    
    % Calculation loop
    % Filling in the matrices on every row but using vectors in ...
    % ... calculations since they need to be computed sequentially
    for t = 1:n
        A(:, t) = RNN.W * h_t + RNN.U * X(:, t) + RNN.b;    % Eq 1
        H(:, t) = tanh(A(:, t));                            % Eq 2
        h_t = H(:, t);  % Needed for next iteration
        O(:, t) = RNN.V * h_t + RNN.c;                      % Eq 3
        P(:, t) = softMax(O(:, t));                         % Eq 4
    end
    
    H = [h0 H]; % Saves first value
    
    % Computes the cross entropy loss
    loss = -sum(log(sum(P .* Y)));
end

function grad = ComputeGradients(X, Y, RNN, Dim, A, H, P)
    grad = GradientObject;
    n = size(X, 2);
    
    % Pre allocate memory
    grad.b = zeros(size(RNN.b));
    grad.c = zeros(size(RNN.c));
    grad.U = zeros(size(RNN.U));
    grad.W = zeros(size(RNN.W));
    grad.V = zeros(size(RNN.V));
    
    % dL_dO for all t
    G = - (Y - P);
    
    % Second bias vector (sum of dL_dO's over t)
    grad.c = G * ones([n, 1]);
    
    % Gradient of V
    grad.V = G * H(:, 2:end)'; % Skip the dummy value h0
    
    % Gradients of L w.r.t. A and H
    grad_A = zeros(n, Dim.m); % Pre allocate memory
    grad_h_end = G(:, end)' * RNN.V; % Gradient when t = tau
    grad_A_end = grad_h_end * diag(1 - tanh(A(:, end)).^2); % Gradient when t = tau
    grad_A(end, :) = grad_A_end;
    
    % Compute gradients w.r.t. a_t and h_t going backwards from t = tau - 1
    for j = n-1:-1:1
        grad_h_t = G(:, j)' * RNN.V + grad_A(j + 1, :) * RNN.W;
        grad_A(j, :) = grad_h_t * diag(1 - tanh(A(:, j)).^2);
    end
    
    % Gradients w.r.t. weight matrices and first bias vector
    grad.W = grad_A' * H(:, 1:end-1)';
    grad.U = grad_A' * X';
    grad.b = grad_A' * ones([n, 1]); % (sum of dL_dA's over t)
end

function num_grads = ComputeGradsNum(X, Y, RNN, Dim, h)
    % Taken from Canvas. Added Dim
    for f = fieldnames(RNN)'
        disp('Computing numerical gradient for')
        disp(['Field name: ' f{1} ]);
        num_grads.(f{1}) = ComputeGradNumSlow(X, Y, f{1}, RNN, Dim, h);
    end

end

function grad = ComputeGradNumSlow(X, Y, f, RNN, Dim, h)
    % Taken from Canvas. Added Dim
    n = numel(RNN.(f));
    grad = zeros(size(RNN.(f)));
    hprev = zeros(size(RNN.W, 1), 1);
    for i=1:n
        RNN_try = RNN;
        RNN_try.(f)(i) = RNN.(f)(i) - h;
        l1 = ComputeLoss(X, Y, RNN_try, Dim, hprev);
        RNN_try.(f)(i) = RNN.(f)(i) + h;
        l2 = ComputeLoss(X, Y, RNN_try, Dim, hprev);
        grad(i) = (l2-l1)/(2*h);
    end

end

function [accuracy, max_error] = DifferenceGradients(analyticalG, numericalG, tol)
    % Compute accuracy (less than threshold) and maximum relative error
    relative_error = ComputeRelativeError(analyticalG, numericalG);
    errors = relative_error(relative_error > tol);
    max_error = max(relative_error(:));
    accuracy = 100 * (1 - numel(errors) / numel(relative_error));
end

function [relative_error] = ComputeRelativeError(analyticalX, numericalX)
    % Compute the relative errors
    relative_error = abs(analyticalX - numericalX) ./ ...
        max(eps, abs(analyticalX) + abs(numericalX));
end