function[FP_rate,FN_rate,TotEr_rate,TruePred_rate,tot_test_pattern,testTime] = fn_test...
    (test_set,actual_labels,SvmOut,Sigma_param,bias)

%%
support_vect_pattern_set = SvmOut.SVs_pattern_set;
SVs_coeff = SvmOut.SVs_coeff;
SVs_labels = SvmOut.labels;
tot_test_pattern = size(test_set,1);

tic
%% decision rule
for q=1:tot_test_pattern
    
    train_each_term=[];
    for p= 1:size(support_vect_pattern_set,1)
        [krnl] = fn_rbf_kernel(support_vect_pattern_set,p,test_set,q,Sigma_param);
        train_each_term(p) = (SVs_coeff(p))*(SVs_labels(p))*krnl;
        %it is positive or negative if the corresponding SVs are from positive or negative classes respectively
    end
    
    decisn_func_test_data(q) = sum(train_each_term) + bias;
    
    % decision function value for each of the test data.This value leads to
    % a prediction about the corresponding test data to be in the positive
    % or negative class if its(this function's) value is positive or negative respectively.
    
    clear train_each_term
end

predict_labels =(sign(decisn_func_test_data))';
testTime=toc;

%% performance measure
dif_predic_actual_labels = predict_labels - actual_labels;
num_flase_pos = length(find(dif_predic_actual_labels == 2));
num_flase_neg = length(find(dif_predic_actual_labels == -2));
num_wrong_prediction = num_flase_pos + num_flase_neg;
num_correct_prediction = length(find(dif_predic_actual_labels == 0));

FP_rate = (num_flase_pos*100)/length(find(actual_labels==-1));
FN_rate = (num_flase_neg*100)/length(find(actual_labels==1));
TotEr_rate = (num_wrong_prediction*100)/tot_test_pattern;
TruePred_rate = (num_correct_prediction*100)/tot_test_pattern;
end