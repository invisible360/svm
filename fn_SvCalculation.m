function [SvmOut] = fn_SvCalculation...
    (alpha,tol,train_matrix,train_label_vector)

SVs_index = find(alpha>tol);
SVs_coeff = alpha(SVs_index);
SVs_set = train_matrix(SVs_index,:);
SVs_labels = train_label_vector( SVs_index);

total_SVs = size(SVs_set,1);

SVs.index = SVs_index;
SVs.SVs_coeff = SVs_coeff;
SVs.SVs_pattern_set = SVs_set;
SVs.labels = SVs_labels;
SVs.number = total_SVs;
SvmOut = SVs;