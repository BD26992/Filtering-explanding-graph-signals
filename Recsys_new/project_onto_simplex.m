function projected_vector = project_onto_simplex(input_vector)
    % Ensure that the input vector is non-negative
    % if any(input_vector < 0)
    %     error('Input vector must be non-negative.');
    % end

    input_vector(input_vector < 0)=0;

    % Sort the input vector in descending order
    sorted_vector = sort(input_vector, 'descend');

    % Calculate the cumulative sum of the sorted vector
    cumulative_sum = cumsum(sorted_vector);

    % Find the largest index (k) such that (1/k) * (cumulative_sum(k)) >= 1
    k = find((1:length(sorted_vector)) .* cumulative_sum >= 1, 1);

    % Check if k was found, otherwise, set k to the length of the input_vector
    if isempty(k)
        k = length(input_vector);
    end

    % Calculate the threshold value (tau) based on the largest index found
    tau = (cumulative_sum(min(k, length(sorted_vector))) - 1) / k;

    % Calculate the projected vector
    projected_vector = max(input_vector - tau, 0);

    % Ensure that the projected vector sums to 1 (within numerical tolerance)
    projected_vector = projected_vector / sum(projected_vector);
end