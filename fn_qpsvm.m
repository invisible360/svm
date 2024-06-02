function [SvmOut,bias,fval,exitflag,output,lambda]= fn_qpsvm ...
    (C_plus,C_minus,total_N,pos_point_index,neg_point_index,...
    train_label_vector,train_matrix,tol,HSvm,Sigma_param)

MAxIter = 10000;
fSvm= -ones(total_N,1); % Multiplying dual optimization problem by '-1'

%initialize Aeq, Beq in inequal constrain 
Aeq =train_label_vector'; % making transpose to match the dimension with alpha
beq =0;

%initialize lower bound(lb) and upper bound(ub) in linear constrain
lb = zeros(total_N,1);
%ub = ones(tot_num_input_patterns,1)*C; 
ub = ones(total_N,1);

ub(pos_point_index) = C_plus;
ub(neg_point_index) = C_minus;


%% using quadprog optimization solver in MATLAB

options = optimset('MaxIter',MAxIter);
%%%x = quadprog(H,f,A,b,Aeq,beq,lb,ub,[],options);
[x,fval,exitflag,output,lambda] = quadprog(HSvm,fSvm,[],[],Aeq,beq,lb,ub,[],options);

[SvmOut]=fn_SvCalculation (x,tol,train_matrix,train_label_vector);

[bias] = fn_qp_bias (SvmOut,C_plus,C_minus,tol,Sigma_param);
