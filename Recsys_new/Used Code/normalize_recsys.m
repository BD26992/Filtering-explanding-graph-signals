function normalized_matrix = normalize_recsys(input_matrix,mean_val,std_val)
% Create a sample sparse matrix (replace this with your own data)
% Example: A sparse matrix of size 4x4

A=full(input_matrix);
% Find the non-zero elements
non_zero_elements = A(A ~= 0);

% Normalize the non-zero elements
normalized_non_zero_elements = (non_zero_elements - mean_val) / std_val;

% Replace the non-zero elements in the original matrix with normalized values
normalized_matrix = A;
normalized_matrix(A ~= 0) = normalized_non_zero_elements;

% Display the normalized matrix
disp('Normalized Matrix:');
disp(full(normalized_matrix));
end