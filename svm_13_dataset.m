clc; clear; close all;
% benchmark data loading
benchmark_data  = load ('benchmarks.mat');

% seperating 13 dataset from benchmark data
field_names = fieldnames(benchmark_data);
dataset_num = [1 2 3 4 5 6 7 8 9 10 11 12 13];

% define the range from start and end dataset 
start_dataset = 1;
end_of_dataset = 2;

% definig the number of file for cv
num_of_file = 1;

%% dataset:
% 1. banana                   % 8. ringnorm
% 2. breast_cancer            % 9. splice
% 3. diabetis                 % 10. thyroid
% 4. flare_solar              % 11. titanic
% 5. german                   % 12. twonorm
% 6. heart                    % 13. waveform
% 7. image

%% cross validation
for d=start_dataset:end_of_dataset
    dataset(dataset_num(d)) = benchmark_data.(field_names{dataset_num(d)});
    
    for a=1:num_of_file
        train_inst_matrix {a,d} = dataset(d).x(dataset(d).train(a,:),:);
        train_inst_label_vector {a,d} = dataset(d).t(dataset(d).train(a,:),:);
        test_inst_matrix {a,d} = dataset(d).x(dataset(d).test(a,:),:);
        test_inst_label_vector {a,d} = dataset(d).t(dataset(d).test(a,:),:);
        
        
        train_matrix= train_inst_matrix{a,d};
        train_label_vector= train_inst_label_vector{a,d};
        test_matrix=test_inst_matrix{a,d};
        test_label_vector=test_inst_label_vector{a,d};
        
        %% preparing train data for Cross Validation
        tr_data_set_length = size(train_matrix,1);
        fold = 5;
        cv_each_set_length = floor(tr_data_set_length/fold); %% dividing train data into 5 fold
        
        %% creating C and Sigma Array
        CP=2.^(0:2:20); %%C_param
        SP = 2.^(-2:2:8);%%Sigma_param
        
        accuracy=zeros(length(CP),length(SP));% Initializing the accuracy matrix
        alpha_per_cv= zeros (length(CP),length(SP));
        MAxIter = 10000;
        
        %% CV Looping Starts
        for c=1:length(CP)
            for s=1:length(SP)
                C_param=CP(c);
                Sigma_param=SP(s);
                ind = 1:size(train_matrix,1);
                
                for i = 1:fold
                    test_ind{i}  = (i-1)*cv_each_set_length+1:i*cv_each_set_length;
                    train_ind{i} = setdiff(ind, test_ind{i}); 
                    
                    cv_train_matrix = train_matrix(train_ind{i},:);
                    cv_train_label = train_label_vector(train_ind{i});
                    cv_test_matrix = train_matrix(test_ind{i},:);
                    cv_test_label = train_label_vector(test_ind{i});
                    
                    % skipping the machine from biasing behaviour and to keep it as synchronous
                    C_plus = C_param;
                    C_minus = C_param;
                    
                    clear SvmOut bias lambda SVs_coeff SVs_labels support_vect_pattern_set
                    
                    %% training function
                    [SvmOut,bias,alpha,lambda] = Train_Fun_SVM(C_plus,...
                        C_minus,Sigma_param,cv_train_matrix,...
                        cv_train_label,MAxIter);
                    
                    test_set = cv_test_matrix;
                    actual_test_labels = cv_test_label;
                    support_vect_pattern_set = SvmOut.SVs_pattern_set;
                    SVs_coeff = SvmOut.SVs_coeff;
                    SVs_labels = SvmOut.labels;
                    gamma = 1/(2*Sigma_param^2);
                    
                    %% testing function
                    [FP_rate,FN_rate,TotEr_rate,TruePred_rate,tot_test_pattern]...
                        = Test_Fun_SVM(test_set,actual_test_labels,...
                        support_vect_pattern_set,SVs_coeff,SVs_labels,gamma,bias);
                    
                    acc(i) = TruePred_rate;
                end
                accuracy(c,s) = mean(acc);
                SVs_per_cv(c,s) = SvmOut.number;
            end %%End of CV s loop
        end %%End of CV c loop
        per_file_cv_accuracy_for_per_dataset_in_grid {a,d}=accuracy;
        per_file_cv_num_of_SVs_for_per_dataset_in_grid {a,d}=SVs_per_cv;
        %% best params:
        [temp,c]=max(accuracy);
        [temp,s]=max(temp);
        per_file_cv_accuracy_for_per_dataset {a,d}=temp;
        c=c(s);
        c_param =CP(c);
        Sigma_param=SP(s);
        per_file_c_value_for_per_dataset{a,d} = c_param;
        per_file_S_value_for_per_dataset{a,d} = Sigma_param;
        
    end
    best_C(d) = mean([per_file_c_value_for_per_dataset{:,d}]);
    best_S(d) = mean([per_file_S_value_for_per_dataset{:,d}]);
end

for d=start_dataset:end_of_dataset
    dataset(dataset_num(d)) = benchmark_data.(field_names{dataset_num(d)});
    
    for a=1:size(dataset(d).train, 1)
        train_inst_matrix_13 {a,d} = dataset(d).x(dataset(d).train(a,:),:);
        train_inst_label_vector_13 {a,d} = dataset(d).t(dataset(d).train(a,:),:);
        test_inst_matrix_13 {a,d} = dataset(d).x(dataset(d).test(a,:),:);
        test_inst_label_vector_13 {a,d} = dataset(d).t(dataset(d).test(a,:),:);
        
        train_matrix= train_inst_matrix_13{a,d};
        train_label_vector= train_inst_label_vector_13{a,d};
        test_matrix=test_inst_matrix_13{a,d};
        test_label_vector=test_inst_label_vector_13{a,d};
        
        c = best_C(d);
        s = best_S(d);
        MAxIter = 10000;
        [SvmOut,bias,alpha,lambda] = Train_Fun_SVM(c,...
            c,s,train_matrix,train_label_vector,MAxIter);
        
        test_set = test_matrix;
        actual_test_labels = test_label_vector;
        
        train_set = train_matrix;
        actual_train_labels = train_label_vector;
        
        support_vect_pattern_set = SvmOut.SVs_pattern_set;
        SVs_coeff = SvmOut.SVs_coeff;
        SVs_labels = SvmOut.labels;
        gamma = 1/(2*s^2);
        per_file_SvmOut_for_per_data{a,d} = SvmOut;
        per_file_num_of_SVs_for_per_data{a,d} = SvmOut.number;
        per_file_bias_for_per_data{a,d} = bias;
        per_file_lambda_for_per_data{a,d} = lambda.eqlin;
        
        [FP_rate_test,FN_rate_test,TotEr_rate_test,TruePred_rate_test,...
            tot_test_pattern_test] = Test_Fun_SVM(test_set,actual_test_labels,...
            support_vect_pattern_set,SVs_coeff,SVs_labels,gamma,bias);
        
        [FP_rate_train,FN_rate_train,TotEr_rate_train,TruePred_rate_train,...
            tot_test_pattern_train] = Test_Fun_SVM(train_set,actual_train_labels,...
            support_vect_pattern_set,SVs_coeff,SVs_labels,gamma,bias);
        
        test_acc{a,d} = TruePred_rate_test;
        train_acc{a,d} = TruePred_rate_train;
    end
    per_dataset_test_acc(d) = mean([test_acc{:,d}]);
    per_dataset_train_acc(d) = mean([train_acc{:,d}]);
    per_dataset_test_std(d) = std([test_acc{:,d}]);
    per_dataset_train_std(d) = std([train_acc{:,d}]);
    per_dataset_mean_SVs(d) = mean([per_file_num_of_SVs_for_per_data{:,d}]);
end
