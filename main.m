clc; clear; close all;
% benchmark data loading
benchmark_data  = load ('benchmarks.mat');

% seperating 13 dataset from benchmark data
field_names = fieldnames(benchmark_data);
dataset_num = [1 2 3 4 5 6 7 8 9 10 11 12 13];

%% dataset:
% 1. banana                   % 8. ringnorm
% 2. breast_cancer            % 9. splice
% 3. diabetis                 % 10. thyroid
% 4. flare_solar              % 11. titanic
% 5. german                   % 12. twonorm
% 6. heart                    % 13. waveform
% 7. image

% define the range from start and end dataset
start_dataset = 1;
end_of_dataset = 1;

% definig the number of file for cv
num_of_file = 5;

dataset = struct('x',{},'t',{},'test',{},'train',{}); %preallocation
%% cross validation
for d=start_dataset:end_of_dataset
    dataset(dataset_num(d)) = benchmark_data.(field_names{dataset_num(d)});
    
    for a=1:num_of_file
        [accuracy, SVs_per_cv,max_acc,c_param,sigma_param]=...
            fn_cvMain(dataset,d,a);
        
        per_file_cv_accuracy_for_per_dataset_in_grid {a,d}=accuracy;
        per_file_cv_num_of_SVs_for_per_dataset_in_grid {a,d}=SVs_per_cv;
        
        per_file_cv_accuracy_for_per_dataset {a,d}=max_acc;
        
        per_file_c_value_for_per_dataset{a,d} = c_param;
        per_file_S_value_for_per_dataset{a,d} = sigma_param;
    end
    
    best_C(d) = mean([per_file_c_value_for_per_dataset{:,d}]);
    best_S(d) = mean([per_file_S_value_for_per_dataset{:,d}]);
    
end

%% final training and testing
for d=start_dataset:end_of_dataset
    dataset(dataset_num(d)) = benchmark_data.(field_names{dataset_num(d)});
    
    for a=1:size(dataset(d).train, 1)
        [train_matrix,train_label_vector,test_matrix,test_label_vector] =...
            fn_trainTestSplit (dataset,d,a);
        
        c_pls = best_C(d);
        c_mns = best_C(d);
        s = best_S(d);
        
        [SvmOut,bias,trainingTime,lambda] = fn_train(c_pls,...
            c_mns,s,train_matrix,train_label_vector);
        
        training_Time {a,d} = trainingTime;
        
        train_set = train_matrix;
        actual_train_labels = train_label_vector;
        
%       per_file_SvmOut_for_per_data{a,d} = SvmOut;
        per_file_num_of_SVs_for_per_data{a,d} = SvmOut.number;
        per_file_bias_for_per_data{a,d} = bias;
        per_file_lambda_for_per_data{a,d} = lambda.eqlin;
        
        [FP_rate_test,FN_rate_test,TotEr_rate_test,TruePred_rate_test,...
            tot_test_pattern,testTime] = fn_test...
            (test_matrix,test_label_vector,SvmOut,s,bias);
        prediction_test_time {a,d}=testTime;
        
        [FP_rate_train,FN_rate_train,TotEr_rate_train,TruePred_rate_train,...
            tot_train_pattern,trainTime] = fn_test...
            (train_matrix,train_label_vector,SvmOut,s,bias);
        prediction_train_time {a,d}=trainTime;
        
        test_acc{a,d} = TruePred_rate_test;
        train_acc{a,d} = TruePred_rate_train;
    end
    per_dataset_test_acc(d) = mean([test_acc{:,d}]);
    per_dataset_train_acc(d) = mean([train_acc{:,d}]);
    per_dataset_test_std(d) = std([test_acc{:,d}]);
    per_dataset_train_std(d) = std([train_acc{:,d}]);
    per_dataset_mean_SVs(d) = mean([per_file_num_of_SVs_for_per_data{:,d}]);
    per_dataset_training_time(d) = mean([training_Time{:,d}]);
    per_dataset_prediction_test_time(d) = mean([prediction_test_time{:,d}]);
    per_dataset_prediction_train_time(d) = mean([prediction_train_time{:,d}]);
end
