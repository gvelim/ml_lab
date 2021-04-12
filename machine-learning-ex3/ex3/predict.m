function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% add layer 1 bias to 1 and calc activation functions for L2
% 5000x401 * 401x25 = 5000 x 25 * 25 * 10 = 5000 * 10
% L1 = 400 + bias
% 5000x401 * 401x25 = 5000 x 25

L2 = sigmoid( [ones(size(X,1),1) X] * Theta1' );

% L2 = 25 + bias
% 5000 x 25 * 25 * 10 = 5000 * 10

L3 = sigmoid( [ones(size(L2,1),1) L2] * Theta2' );

% get idx for max value per row and put it in "p"
[val,p] = max( L3, [], 2)


% =========================================================================


end
