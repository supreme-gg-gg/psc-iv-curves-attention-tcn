%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Preprocessing of data from an optoelectronic model in COMSOL
%
% Authors: Lee Lehan, Erik Birgersson
% Data modified: 3 October 2024
%
% Preprocessing
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Load Input and Output Data
% Load current density (A/m^2) and input parameters from text files

% current_density_raw = load('Data\Data_1k_rng1\iV_m.txt'); % Output data (current density)
% input_params_raw = load('Data\Data_1k_rng1\LHS_parameters_m.txt'); % 31 input parameters
% SavedFileName = "Data_1k_rng1";

% current_density_raw = load('Data\Data_1k_rng2\iV_m.txt'); % Output data (current density)
% input_params_raw = load('Data\Data_1k_rng2\LHS_parameters_m.txt'); % 31 input parameters
% SavedFileName = "Data_1k_rng2";

% current_density_raw = load('Data\Data_1k_rng3\iV_m.txt'); % Output data (current density)
% input_params_raw = load('Data\Data_1k_rng3\LHS_parameters_m.txt'); % 31 input parameters
% SavedFileName = "Data_1k_rng3";

% current_density_raw = load('Data\Data_10k_rng1\iV_m.txt'); % Output data (current density)
% input_params_raw = load('Data\Data_10k_rng1\LHS_parameters_m.txt'); % 31 input parameters
% SavedFileName = "Data_10k_rng1";

% current_density_raw = load('Data\Data_10k_rng2\iV_m.txt'); % Output data (current density)
% input_params_raw = load('Data\Data_10k_rng2\LHS_parameters_m.txt'); % 31 input parameters
% SavedFileName = "Data_10k_rng2";
% 
% current_density_raw = load('Data\Data_10k_rng3\iV_m.txt'); % Output data (current density)
% input_params_raw = load('Data\Data_10k_rng3\LHS_parameters_m.txt'); % 31 input parameters
% SavedFileName = "Data_10k_rng3";

current_density_raw = load('Data\Data_100k\iV_m.txt'); % Output data (current density)
input_params_raw = load('Data\Data_100k\LHS_parameters_m.txt'); % 31 input parameters
SavedFileName = "Data_100k";

%% Applied Voltage Range
% See MATLAB file for COMSOL sweep that defines the applied voltage range
voltage_raw = [0:0.1:0.4, 0.425:0.025:1.4]; % Applied voltage (V)

%% Interpolation and IV-Curve Reduction
% Initialize parameters for interpolation
num_points_pre_mpp = 3;
num_points_post_mpp = 4;
num_inputs = size(input_params_raw, 1); % Number of input data rows

% Initialize matrices for reduced current density and voltage
current_density_reduced = zeros(num_inputs, num_points_pre_mpp + num_points_post_mpp - 1);
voltage_reduced = zeros(num_inputs, num_points_pre_mpp + num_points_post_mpp - 1);

% Loop over each input row to process IV curves
for idx = 1:num_inputs
    interp_voltage_range = linspace(0, 1.4, 1e4); % High-resolution voltage range
    % Interpolate current density using 'pchip' interpolation
    current_density_interp = interp1(voltage_raw, current_density_raw(idx,:), interp_voltage_range, 'pchip');

    % Find index where current density first becomes negative
    neg_index = find(current_density_interp < 0, 1);

    if ~isempty(neg_index)
        % Calculate open circuit voltage (Voc)
        Voc = interp1(current_density_interp(neg_index-2:end), interp_voltage_range(neg_index-2:end), 0, 'linear');
        % If no negative current density, extrapolate beyond the existing voltage range
    else
        disp(['Extrapolation: ', num2str(idx)]);
        interp_voltage_range = linspace(0, 2, 1e4); % Extend voltage range
        current_density_interp = interp1(voltage_raw, current_density_raw(idx,:), interp_voltage_range, 'linear', 'extrap');
        neg_index = find(current_density_interp < 0, 1);
        % Calculate open circuit voltage (Voc)
        Voc = interp1(current_density_interp(neg_index-2:end), interp_voltage_range(neg_index-2:end), 0, 'linear');
    end


    % Find max power point (MPP)
    power_interp = interp_voltage_range .* current_density_interp;
    [P_max, mpp_index] = max(power_interp);
    V_mpp = interp_voltage_range(mpp_index); % Voltage at MPP
    I_mpp = current_density_interp(mpp_index); % Current at MPP

    % Partition the voltage range before and after MPP
    voltage_pre_mpp = linspace(0, V_mpp, num_points_pre_mpp);
    voltage_post_mpp = linspace(V_mpp, Voc, num_points_post_mpp);

    % Interpolate current density at the partitioned voltage points
    current_pre_mpp = interp1(interp_voltage_range, current_density_interp, voltage_pre_mpp);
    current_post_mpp = interp1(interp_voltage_range, current_density_interp, voltage_post_mpp);

    % Store reduced voltage and current density values
    voltage_reduced(idx,:) = [voltage_pre_mpp, voltage_post_mpp(2:end)];
    current_density_reduced(idx,:) = [current_pre_mpp, current_post_mpp(2:end)];
end

% Rename variables for clarity
voltage_reduced_final = voltage_reduced;
current_density_reduced_final = current_density_reduced;

%% Outlier Removal
% Initialize threshold and empty list for rows to delete
current_density_threshold = 1000; % Max allowed current density (A/m^2)
rows_to_remove = [];

% Loop through each row and check for outliers
for idx = size(current_density_reduced_final, 1):-1:1
    if max(current_density_reduced_final(idx,:)) > current_density_threshold
        rows_to_remove = [rows_to_remove; idx];
        disp(['Current density too large: ', num2str(idx)]);
    elseif min(current_density_reduced_final(idx,:)) < -1
        rows_to_remove = [rows_to_remove; idx];
        disp(['Current density too negative: ', num2str(idx)]);
    end
end

% Remove outlier rows from raw and reduced data
current_density_raw_clean = current_density_raw;
input_params_clean = input_params_raw;
current_density_reduced_clean = current_density_reduced_final;
voltage_reduced_clean = voltage_reduced_final;

current_density_raw_clean(rows_to_remove, :) = [];
input_params_clean(rows_to_remove, :) = [];
current_density_reduced_clean(rows_to_remove, :) = [];
voltage_reduced_clean(rows_to_remove, :) = [];

%% Set all current densities at Voc to be exactly zero
current_density_reduced_clean(:,num_points_pre_mpp + num_points_post_mpp - 1) = 0;


%% Reduce the number of digits in input and output
n_round = 4; % Calculate number of significant digits and round to 4 significant digits
% input_params_clean 
% current_density_reduced_clean 
% voltage_reduced_clean

% Loop through each element of the matrix to round it to 4 significant digits
A = input_params_clean;
B = zeros(size(A));
for i = 1:numel(A)
    if A(i) ~= 0
        % Calculate the number of digits to shift for rounding to 4 significant digits
        scaleFactor = 10^(4 - ceil(log10(abs(A(i)))));
        B(i) = round(A(i) * scaleFactor) / scaleFactor;
    else
        B(i) = 0;  % Handle the case where the element is 0
    end
end
input_params_clean = B;


% Loop through each element of the matrix to round it to 4 significant digits
A = current_density_reduced_clean ;
B = zeros(size(A));
for i = 1:numel(A)
    if A(i) ~= 0
        % Calculate the number of digits to shift for rounding to 4 significant digits
        scaleFactor = 10^(4 - ceil(log10(abs(A(i)))));
        B(i) = round(A(i) * scaleFactor) / scaleFactor;
    else
        B(i) = 0;  % Handle the case where the element is 0
    end
end
current_density_reduced_clean = B;

% Loop through each element of the matrix to round it to 4 significant digits
A = voltage_reduced_clean;
B = zeros(size(A));
for i = 1:numel(A)
    if A(i) ~= 0
        % Calculate the number of digits to shift for rounding to 4 significant digits
        scaleFactor = 10^(4 - ceil(log10(abs(A(i)))));
        B(i) = round(A(i) * scaleFactor) / scaleFactor;
    else
        B(i) = 0;  % Handle the case where the element is 0
    end
end
voltage_reduced_clean = B;


%% Repeat Input Rows for Normalization
num_repeat_per_row = numel(voltage_reduced_clean(1,:)); % Number of times to repeat each row

% Repeat each row of input and append voltage as the last column
input_params_mod = [];
for idx = 1:size(input_params_clean, 1)
    repeated_row = repmat(input_params_clean(idx, :), num_repeat_per_row, 1);
    repeated_row(:, end+1) = voltage_reduced_clean(idx,:)';
    input_params_mod = [input_params_mod; repeated_row];
end

% Reshape current density to a column vector
current_density_mod = reshape(current_density_reduced_clean', [], 1);

%% Normalize Input and Output
% Apply log10 normalization to input parameters
input_params_mod_norm = input_params_mod;
input_params_mod_norm(:,1:31) = log10(input_params_mod(:,1:31));

% No normalization applied to output (current density) in this case
current_density_mod_norm = current_density_mod;

% Change so that the rows are the nFeatures and the columns are the
% nSamples
Y = current_density_mod_norm;
X = input_params_mod_norm;

save(SavedFileName,'X','Y','current_density_raw','input_params_raw','voltage_raw');
return
%% Plot IV Curves and Reduced Curves with Normalized Data in multiple figures
% Define sample points and colors for plotting
N_samples = 180;
sample_indices = 1:ceil(size(current_density_reduced_clean, 1)/N_samples):size(current_density_reduced_clean, 1);
plot_colors = lines(length(sample_indices));

% Maximum rows and columns per figure
max_columns = 10;
max_rows = 6;
max_plots_per_figure = max_columns * max_rows;

% Total number of figures required
num_figures = ceil(length(sample_indices) / max_plots_per_figure);

for fig = 1:num_figures
    % Create a new figure for each batch of plots
    figure(fig);

    % Determine the indices for the current figure
    start_idx = (fig - 1) * max_plots_per_figure + 1;
    end_idx = min(fig * max_plots_per_figure, length(sample_indices));

    % Number of plots in the current figure
    num_plots = end_idx - start_idx + 1;

    % Calculate the rows and columns for the current figure
    num_rows = ceil(num_plots / max_columns);

    % Create tiled layout for the current figure
    t = tiledlayout(num_rows, max_columns, 'TileSpacing', 'compact', 'Padding', 'compact');

    for ii = start_idx:end_idx
        plot_idx = ii - start_idx + 1;  % Adjust index for the current figure

        % Create the next tile for IV curves and normalized data
        nexttile(plot_idx);
        hold on;

        % --- Plot raw and reduced IV curves ---
        plot(voltage_raw, current_density_raw_clean(sample_indices(ii), :), 'Color', plot_colors(ii, :), 'LineWidth', 1.0);
        plot(voltage_reduced_clean(sample_indices(ii), :), current_density_reduced_clean(sample_indices(ii), :), 'o', 'Color', plot_colors(ii, :), 'MarkerSize', 6);

        % Set axis limits for the IV curves
        axis tight;
        ylim([-100, max(current_density_raw_clean(sample_indices(ii), :))]);
        grid on;

        % Add dynamic title for IV curves
        title(['Row ', num2str(sample_indices(ii))]);

        % --- Overlay the Verification Plot for Normalized Data ---
        idx = sample_indices(ii);
        plot_start = num_repeat_per_row * (idx - 1) + 1;

        % Plot normalized data
        plot(input_params_mod_norm(plot_start:plot_start+num_repeat_per_row-1, 32), current_density_mod_norm(plot_start:plot_start+num_repeat_per_row-1), '.', 'Color', plot_colors(ii, :), 'MarkerSize', 6);

        % Keep the axis limits consistent
        ylim([-100, max(current_density_raw_clean(sample_indices(ii), :))]);

        hold off;
    end
end


%% Boxplot Visualization
figure;

% Plot boxplots for raw input parameters
subplot(4, 2, 1);
boxplot(input_params_raw);
set(gca, 'YScale', 'log');
title('Input Parameters (Raw)');

% Plot boxplots for raw current density
subplot(4, 2, 2);
boxplot(current_density_raw);
title('Current Density (Raw)');

% Plot boxplots for reduced current density
subplot(4, 2, 4);
boxplot(current_density_reduced_final);
title('Current Density (Reduced)');

% Plot boxplots for modified input parameters (after repeat and append)
subplot(4, 2, 5);
boxplot(input_params_mod);
set(gca, 'YScale', 'log');
title('Input Parameters (Modified)');

% Plot boxplots for reshaped current density
subplot(4, 2, 6);
boxplot(current_density_mod);
title('Current Density (Reshaped)');

% Plot boxplots for normalized input parameters
subplot(4, 2, 7);
boxplot(input_params_mod_norm);
title('Input Parameters (Normalized)');

% Plot boxplots for normalized current density
subplot(4, 2, 8);
boxplot(current_density_mod_norm);
title('Current Density (Normalized)');
