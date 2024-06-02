function [accuracy, SVs_per_cv,max_acc,c_param,sigma_param] = fn_cvMain...
    (dataset_name, num_of_dataset, num_of_file)
%% Splitting Train and Test matrix
[train_matrix,train_label_vector] = fn_trainTestSplit ...
    (dataset_name, num_of_dataset, num_of_file);

%% preparing train data for Cross Validation
tr_data_set_length = size(train_matrix,1);
fold = 5;
cv_each_set_length = floor(tr_data_set_length/fold);

%% creating C and Sigma Array
CP=2.^(0:2:20); %%C_param 20
SP = 2.^(-2:2:8);%%Sigma_param

%% initialize accuracy matrix and MAxIter
accuracy=zeros(length(CP),length(SP));
SVs_per_cv=zeros(length(CP),length(SP));

%% CV Looping Starts
for c=1:length(CP)
    for s=1:length(SP)
        C_param=CP(c);
        Sigma_param=SP(s);
        ind = 1:size(train_matrix,1);
        
        [acc,SvsPerCv] = fn_crossValidation...
            (C_param,Sigma_param,ind,train_matrix,...
            train_label_vector,cv_each_set_length,...
            fold);
        accuracy(c,s) = mean(acc);
        SVs_per_cv(c,s) = SvsPerCv;
    end %%End of CV s loop
end %%End of CV c loop

%% best params:
[max_acc,c]=max(accuracy);
[max_acc,s]=max(max_acc);
c=c(s);
c_param =CP(c);
sigma_param=SP(s);
