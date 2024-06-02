function [acc,SvsPerCv] = fn_crossValidation...
            (C_param,Sigma_param,ind,train_matrix, train_label_vector,...
            cv_each_set_length,fold)

for i = 1:fold
    test_ind{i}  = (i-1)*cv_each_set_length+1:i*cv_each_set_length;
    train_ind{i} = setdiff(ind, test_ind{i});
    
    cv_train_matrix = train_matrix(train_ind{i},:);
    cv_train_label = train_label_vector(train_ind{i});
    cv_test_matrix = train_matrix(test_ind{i},:);
    cv_test_label = train_label_vector(test_ind{i});
    
    C_plus = C_param;
    C_minus = C_param;
    
    clear SvmOut bias lambda SVs_coeff SVs_labels support_vect_pattern_set
    
    %% training function
    [SvmOut,bias] = fn_train(C_plus,C_minus,...
        Sigma_param,cv_train_matrix,...
        cv_train_label);

    SvsPerCv = SvmOut.number;
    
    %% testing function/prediction function
    [FP_rate,FN_rate,TotEr_rate,TruePred_rate,tot_test_pattern,testTime(i)] = fn_test...
    (cv_test_matrix,cv_test_label,SvmOut,Sigma_param,bias);

    acc(i) = TruePred_rate;
end