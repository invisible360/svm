function [SvmOut,bias,trainingTime,lambda] = fn_train...
    (C_plus,C_minus,...
    Sigma_param,train_matrix,...
    train_label_vector)
tic;
total_N = length(train_label_vector);
tol=10^(-6);

HSvm = zeros (total_N,total_N);
for i=1:total_N
    for j =1:total_N
        % (Gaussian RBF)Finding the training kernel matrix
        [krnl] = fn_rbf_kernel(train_matrix,i,train_matrix,j,Sigma_param);
        HSvm(i,j)= train_label_vector(i)*train_label_vector(j)*krnl;
    end
end

pos_point_index = find(train_label_vector==1);
neg_point_index = find(train_label_vector==-1);
% length_pos_points = length(pos_point_index);
% length_neg_points = length(neg_point_index);

%% LPSVM
% [SvmOut,bias,fval,exitflag,output,lambda]= fn_lpsvm...
%     (C_plus,C_minus,total_N,pos_point_index,neg_point_index,...
%     train_label_vector,train_matrix,tol,HSvm,MAxIter);
%% QPSVM
[SvmOut,bias,fval,exitflag,output,lambda]= fn_qpsvm...
    (C_plus,C_minus,total_N,pos_point_index,neg_point_index,...
    train_label_vector,train_matrix,tol,HSvm,Sigma_param);
%%
trainingTime=toc;
end