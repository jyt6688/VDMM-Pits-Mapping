clear; clc; close all;
filename = 'data\Morphology_data.csv';
data = readtable(filename, 'VariableNamingRule', 'preserve');
input_rh = {'FC1,2', 'FC2,1', 'FC2,2', 'FCsum'};
output_rh = {'r', 'h'};
X = table2array(data(:, input_rh));
Y = table2array(data(:, output_rh));
rng(42);
train_ratio = 0.8;
num_train = round(train_ratio * size(X, 1));
indices = randperm(size(X, 1));
train_indices = indices(1:num_train);
test_indices = indices(num_train+1:end);
X_train_raw = X(train_indices, :);
Y_train_raw = Y(train_indices, :);
X_test_raw = X(test_indices, :);
Y_test_raw = Y(test_indices, :);
[X_train, X_ps] = mapminmax(X_train_raw', -1, 1);
X_train = X_train';
X_test = mapminmax('apply', X_test_raw', X_ps);
X_test = X_test';
[Y_train, Y_ps] = mapminmax(Y_train_raw', -1, 1);
Y_train = Y_train';
Y_test = mapminmax('apply', Y_test_raw', Y_ps);
Y_test = Y_test';
hidden_layer_size = 10;
train_epochs = 2000;
temp_net = fitnet(hidden_layer_size, 'trainlm');
temp_net = configure(temp_net, X_train', Y_train');
num_weights_biases = numel(temp_net.IW{1,1}) + numel(temp_net.b{1}) ...
    + numel(temp_net.LW{2,1}) + numel(temp_net.b{2});
ga_pop_size = 50;
ga_max_gen = 30;
ga_pc = 0.7;
ga_pm = 0.1;
weight_range = [-1, 1];
ga_train_epochs = 50;
bound = [repmat(weight_range(1), num_weights_biases, 1), ...
         repmat(weight_range(2), num_weights_biases, 1)];
population = zeros(ga_pop_size, num_weights_biases);
for i = 1:ga_pop_size
    population(i, :) = encodeChromosome(num_weights_biases, bound);
end
global_best_mse = inf;
best_chromosome = [];
fprintf('Starting genetic algorithm optimization...\n');
for gen = 1:ga_max_gen
    mse_cost = zeros(ga_pop_size, 1);
    for i = 1:ga_pop_size
        try
            net_temp = fitnet(hidden_layer_size, 'trainlm');
            net_temp = configure(net_temp, X_train', Y_train');
            net_temp.trainParam.showWindow = false;
            net_temp.trainParam.showCommandLine = false;
            net_temp.trainParam.epochs = ga_train_epochs;
            net_temp.divideFcn = 'divideind';
            net_temp.divideParam.trainInd = 1:size(X_train, 1);
            net_temp.divideParam.valInd = [];
            net_temp.divideParam.testInd = [];
            net_temp = decodeWeightsBiases(population(i, :), net_temp);
            [net_temp, ~] = train(net_temp, X_train', Y_train');
            Y_pred_temp = net_temp(X_train');
            mse_cost(i) = mean(mean((Y_train' - Y_pred_temp).^2));
        catch
            mse_cost(i) = 1e6;
        end
    end
    [best_mse, best_idx] = min(mse_cost);
    if best_mse < global_best_mse
        global_best_mse = best_mse;
        best_chromosome = population(best_idx, :);
    end
    if mod(gen, 10) == 0 || gen == 1 || gen == ga_max_gen
        fprintf('GA generation %d/%d, best fitness = %.6f\n', gen, ga_max_gen, 1 / (1 + best_mse));
    end
    population = gaSelect(population, mse_cost);
    population = gaCross(ga_pc, num_weights_biases, population, ga_pop_size, bound);
    population = gaMutation(ga_pm, num_weights_biases, population, ga_pop_size, gen, ga_max_gen, bound);
    population(1, :) = best_chromosome;
end
net_rh = fitnet(hidden_layer_size, 'trainlm');
net_rh = configure(net_rh, X_train', Y_train');
net_rh = decodeWeightsBiases(best_chromosome, net_rh);
net_rh.trainParam.epochs = train_epochs;
net_rh.trainParam.showWindow = true;
net_rh.trainParam.showCommandLine = false;
net_rh.trainParam.goal = 1e-8;
net_rh.divideFcn = 'divideind';
net_rh.divideParam.trainInd = 1:size(X_train, 1);
net_rh.divideParam.valInd = [];
net_rh.divideParam.testInd = [];
fprintf('Starting neural network training...\n');
net_rh = train(net_rh, X_train', Y_train');
Y_train_pred = mapminmax('reverse', net_rh(X_train'), Y_ps)';
Y_test_pred = mapminmax('reverse', net_rh(X_test'), Y_ps)';
Y_train_actual = Y_train_raw;
Y_test_actual = Y_test_raw;
train_rmse_r = sqrt(mean((Y_train_actual(:,1) - Y_train_pred(:,1)).^2));
train_rmse_h = sqrt(mean((Y_train_actual(:,2) - Y_train_pred(:,2)).^2));
test_rmse_r = sqrt(mean((Y_test_actual(:,1) - Y_test_pred(:,1)).^2));
test_rmse_h = sqrt(mean((Y_test_actual(:,2) - Y_test_pred(:,2)).^2));
train_r2_r = 1 - sum((Y_train_actual(:,1) - Y_train_pred(:,1)).^2) / ...
    sum((Y_train_actual(:,1) - mean(Y_train_actual(:,1))).^2);
train_r2_h = 1 - sum((Y_train_actual(:,2) - Y_train_pred(:,2)).^2) / ...
    sum((Y_train_actual(:,2) - mean(Y_train_actual(:,2))).^2);
test_r2_r = 1 - sum((Y_test_actual(:,1) - Y_test_pred(:,1)).^2) / ...
    sum((Y_test_actual(:,1) - mean(Y_test_actual(:,1))).^2);
test_r2_h = 1 - sum((Y_test_actual(:,2) - Y_test_pred(:,2)).^2) / ...
    sum((Y_test_actual(:,2) - mean(Y_test_actual(:,2))).^2);
fprintf('Training set - r: RMSE=%.4f, R²=%.4f | h: RMSE=%.4f, R²=%.4f\n', ...
    train_rmse_r, train_r2_r, train_rmse_h, train_r2_h);
fprintf('Test set - r: RMSE=%.4f, R²=%.4f | h: RMSE=%.4f, R²=%.4f\n', ...
    test_rmse_r, test_r2_r, test_rmse_h, test_r2_h);
fig = figure('Name', 'Pit morphology prediction vs actual (test set)', 'Position', [100, 100, 1100, 480]);
subplot(1, 2, 1);
hold on;
plot(Y_test_actual(:,1), Y_test_pred(:,1), 'bo', 'MarkerFaceColor', [0.3 0.5 1], ...
    'MarkerSize', 6, 'DisplayName', sprintf('Test set (n=%d)', size(Y_test_actual, 1)));
r_lim = [min(Y_test_actual(:,1)), max(Y_test_actual(:,1))];
plot(r_lim, r_lim, 'k-', 'LineWidth', 1.5, 'DisplayName', 'y=x');
xlabel('Actual radius r'); ylabel('Predicted radius r');
title('pits r');
text(0.05, 0.95, sprintf('R^2 = %.3g\nRMSE = %.3f', test_r2_r, test_rmse_r), ...
    'Units', 'normalized', 'VerticalAlignment', 'top', 'HorizontalAlignment', 'left', ...
    'FontSize', 10, 'BackgroundColor', 'w', 'EdgeColor', 'k', 'Margin', 4);
legend('Location', 'southeast'); grid on; axis equal tight;
subplot(1, 2, 2);
hold on;
plot(Y_test_actual(:,2), Y_test_pred(:,2), 'bo', 'MarkerFaceColor', [0.3 0.5 1], ...
    'MarkerSize', 6, 'DisplayName', sprintf('Test set (n=%d)', size(Y_test_actual, 1)));
h_lim = [min(Y_test_actual(:,2)), max(Y_test_actual(:,2))];
plot(h_lim, h_lim, 'k-', 'LineWidth', 1.5, 'DisplayName', 'y=x');
xlabel('Actual depth h'); ylabel('Predicted depth h');
title('pits h');
text(0.05, 0.95, sprintf('R^2 = %.3g\nRMSE = %.3f', test_r2_h, test_rmse_h), ...
    'Units', 'normalized', 'VerticalAlignment', 'top', 'HorizontalAlignment', 'left', ...
    'FontSize', 10, 'BackgroundColor', 'w', 'EdgeColor', 'k', 'Margin', 4);
legend('Location', 'southeast'); grid on; axis equal tight;
saveas(fig, 'actual_vs_predicted_morphology.png');
fprintf('Figure saved: actual_vs_predicted_morphology.png\n');
save('net_rh.mat', 'net_rh', 'X_ps', 'Y_ps', 'input_rh', 'output_rh', ...
    'train_rmse_r', 'train_rmse_h', 'test_rmse_r', 'test_rmse_h', ...
    'train_r2_r', 'train_r2_h', 'test_r2_r', 'test_r2_h');
fprintf('Model saved: net_rh.mat\n');

function ret = encodeChromosome(lenchrom, bound)
    flag = 0;
    while flag == 0
        pick = rand(1, lenchrom);
        ret = bound(:, 1)' + (bound(:, 2) - bound(:, 1))' .* pick;
        flag = testChromosome(ret, bound);
    end
end

function flag = testChromosome(chrom, bound)
    flag = all(chrom >= bound(:, 1)') & all(chrom <= bound(:, 2)');
end

function ret = gaSelect(population, cost)
    sizepop = size(population, 1);
    fitness1 = 10 ./ cost;
    fitness1(~isfinite(fitness1)) = max(fitness1(isfinite(fitness1)));
    sumf = fitness1 / sum(fitness1);
    index = zeros(1, sizepop);
    for i = 1:sizepop
        pick = rand;
        while pick == 0
            pick = rand;
        end
        for j = 1:sizepop
            pick = pick - sumf(j);
            if pick < 0
                index(i) = j;
                break;
            end
        end
    end
    ret = population(index, :);
end

function ret = gaCross(pcross, lenchrom, chrom, sizepop, bound)
    for i = 1:sizepop
        pick = rand(1, 2);
        while prod(pick) == 0
            pick = rand(1, 2);
        end
        index = ceil(pick .* sizepop);
        pick = rand;
        while pick == 0
            pick = rand;
        end
        if pick > pcross
            continue;
        end
        flag = 0;
        while flag == 0
            pick = rand;
            while pick == 0
                pick = rand;
            end
            pos = ceil(pick * lenchrom);
            pick = rand;
            v1 = chrom(index(1), pos);
            v2 = chrom(index(2), pos);
            chrom(index(1), pos) = pick * v2 + (1 - pick) * v1;
            chrom(index(2), pos) = pick * v1 + (1 - pick) * v2;
            flag = testChromosome(chrom(index(1), :), bound) && ...
                   testChromosome(chrom(index(2), :), bound);
        end
    end
    ret = chrom;
end

function ret = gaMutation(pmutation, lenchrom, chrom, sizepop, num, maxgen, bound)
    for i = 1:sizepop
        pick = rand;
        while pick == 0
            pick = rand;
        end
        index = ceil(pick * sizepop);
        pick = rand;
        if pick > pmutation
            continue;
        end
        flag = 0;
        while flag == 0
            pick = rand;
            while pick == 0
                pick = rand;
            end
            pos = ceil(pick * lenchrom);
            pick = rand;
            fg = (rand * (1 - num / maxgen))^2;
            if pick > 0.5
                chrom(index, pos) = chrom(index, pos) + ...
                    (bound(pos, 2) - chrom(index, pos)) * fg;
            else
                chrom(index, pos) = chrom(index, pos) - ...
                    (chrom(index, pos) - bound(pos, 1)) * fg;
            end
            flag = testChromosome(chrom(index, :), bound);
        end
    end
    ret = chrom;
end

function net = decodeWeightsBiases(chromosome, net)
    idx = 1;

    if ~isempty(net.IW{1,1})
        [r, c] = size(net.IW{1,1});
        net.IW{1,1} = reshape(chromosome(idx:idx+r*c-1), r, c);
        idx = idx + r*c;
    end

    if ~isempty(net.b{1})
        [r, ~] = size(net.b{1});
        net.b{1} = reshape(chromosome(idx:idx+r-1), r, 1);
        idx = idx + r;
    end

    if length(net.LW) >= 2 && ~isempty(net.LW{2,1})
        [r, c] = size(net.LW{2,1});
        net.LW{2,1} = reshape(chromosome(idx:idx+r*c-1), r, c);
        idx = idx + r*c;
    end

    if length(net.b) >= 2 && ~isempty(net.b{2})
        [r, ~] = size(net.b{2});
        net.b{2} = reshape(chromosome(idx:idx+r-1), r, 1);
    end
end
