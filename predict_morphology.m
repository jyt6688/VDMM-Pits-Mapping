clear; clc;
predict_filename = 'data\Experimental_data.csv';
model_filename   = 'net_rh.mat';
output_filename  = 'prediction_morphology_results.csv';

S = load(model_filename);
net_rh = S.net_rh;
X_ps   = S.X_ps;
Y_ps   = S.Y_ps;

data = readtable(predict_filename, 'VariableNamingRule', 'preserve');
if ismember('No.', data.Properties.VariableNames)
    pit_ids = data.('No.');
else
    pit_ids = data.(data.Properties.VariableNames{1});
end

FC2_1 = data.('FC2,1');
FC2_2 = data.('FC2,2');
FC2_3 = data.('FC2,3');
k1 = 0.23;
k3 = -0.32;
k4 = -0.32;
FCsum = FC2_1 + FC2_2 + FC2_3;
FC1_2C = k1 * FCsum / (1 + k3 + k4);
FC2_1C = k3 * FCsum / (1 + k3 + k4);
FC2_2C = FCsum / (1 + k3 + k4);
X = [FC1_2C, FC2_1C, FC2_2C, FCsum];
X_norm = mapminmax('apply', X', X_ps);
Y_pred = mapminmax('reverse', net_rh(X_norm), Y_ps)';

Result = table(pit_ids, Y_pred(:,1), Y_pred(:,2), ...
    'VariableNames', {'No.','r','h'});
writetable(Result, output_filename);
fprintf('Prediction results saved to: %s\n', output_filename);
disp(Result);
