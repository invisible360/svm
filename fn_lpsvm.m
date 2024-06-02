function [SvmOut,bias,fval,exitflag,output,lambda]=fn_lpsvm...
    (C_plus,C_minus,total_N,pos_point_index,neg_point_index,...
    train_label_vector,train_matrix,tol,HSvm,MAxIter)

wye = zeros(1,total_N);
for i=1:total_N
    wye(i) = train_label_vector(i);
end

M = ones(total_N,1)';
M(pos_point_index) = C_plus;
M(neg_point_index) = C_minus;
iden = eye(total_N);
%wye = train_label_vector(i);

f = [ones(1,total_N) M 0];
A = -[HSvm iden wye'];
b = -ones(total_N,1);
lb = [zeros(2*total_N,1);-inf];
ub = [];
%ub = [inf(1,tot_num_input_patterns) inf(1,tot_num_input_patterns) inf]';

Aeq = [];
beq = [];
x0 = [];

options = optimset('MaxIter',MAxIter);
clear x;
%HSvm,fSvm,[],[],Aeq,beq,lb,ub,[],options);
[x,fval,exitflag,output,lambda] = linprog(f,A,b,Aeq,beq,lb,ub,x0,options);

alpha= x(1:total_N);
[SvmOut]=fn_SvCalculation (alpha,tol,train_matrix,train_label_vector);
bias = x(end);
