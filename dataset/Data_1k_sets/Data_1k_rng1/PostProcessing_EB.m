%%------------------------------------------------------------------------
% Code to load and postprocess raw data from COMSOL simulations of a
% single-junction perovskite solar cell. See Zhao Xinhai's PhD thesis for
% more details on the generation of the data.

%% PREPROCESSING
clear
format long
close all
clc

%% LOAD DATA

% Load input and output data from the text files
Output = load('iV_m.txt'); % output in the form of current density, A/m^2
Input = load('LHS_parameters_m.txt'); %31 input parameters

% see MATLAB file for COMSOL sweep that defines the applied voltage range
Va = [0:0.1:0.4,0.425:0.025:1.4]; % applied voltage, V;

N = length(Output); % number of cases

%% POSTPROCESSING
% We visualise the raw data

% iV-curve for the first case
figure(1)
plot(Va,Output(1,:),'.')
ylim([0 400])

% iV-curves for all cases
figure(2)
for i=1:N
    plot(Va,Output(i,:),'.')
    hold on
end
ylim([-2000 400])

% Output current density
figure(3)
boxplot(Output)
ylim([-2500 400])
grid on

% Input data
figure(4)
boxplot(Input)
set(gca, 'YScale', 'log');

