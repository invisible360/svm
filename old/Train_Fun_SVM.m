function[SvmOut,bias,x,lambda] = Train_Fun_SVM(C_param...
    , Sigma_param,train_matrix,train_label,MAxIter)

%train_matrix dim = 320x2
%train_label dim = 320x1

tot_num_input_patterns = length(train_label);
tolerance = 10^-6;

gamma = 1/(2*(Sigma_param^2));
L = zeros(tot_num_input_patterns, tot_num_input_patterns);

        
%% loop for calculating a hesian symmetric matrix
for i=1:tot_num_input_patterns
    for j =1:tot_num_input_patterns
        L(i,j)= train_label(i)*(train_label(j)*...
            (exp(-gamma*((train_matrix(i,:)-train_matrix(j,:))*...
            (train_matrix(i,:)-train_matrix(j,:))'))));
    end
end

%% Initialization of parameters for solving LP Problem

%initialize f
fSvm= [ones(1,tot_num_input_patterns) C_param*ones(1,tot_num_input_patterns) 0]; % Multiplying dual optimization problem by '-1'

%initialize Aeq, Beq in inequal constrain 
Aeq =[]; 
beq =[];

%initialize lower bound(lb) and upper bound(ub) in linear constrain
lb = [zeros(1,tot_num_input_patterns) zeros(1,tot_num_input_patterns) -1*inf]';
ub = [inf(1,tot_num_input_patterns) inf(1,tot_num_input_patterns) inf]';

%initialize A and b in inequal constrain

A = -1*[L eye(tot_num_input_patterns) train_label];
b= -1*ones(tot_num_input_patterns,1);

%% using quadprog optimization solver in MATLAB
tic;
clear x
% options = optimset('MaxIter',MAxIter); 
%%%[x,fval,exitflag,output,lambda] = linprog(f,A,b,Aeq,beq,lb,ub,options);
[x,fval,exitflag,output,lambda] = linprog(fSvm,A,b,Aeq,beq,lb,ub);
toc;


alpha_for_sv_coeff = x(1:tot_num_input_patterns,1);
SVs_index = find (alpha_for_sv_coeff>tolerance);
SVs_coeff = alpha_for_sv_coeff(SVs_index);
total_SVs = length(SVs_index);
SVs_labels = train_label(SVs_index);
SVs_pattern_set = train_matrix(SVs_index,:);

% SVs.sv_alpha = SVs_index;
% SVs.alpha_for_sv_index = SVs_coeff;
% SVs.index_length = index_length;
% SVs.sv_index_labels = SVs_labels;
% SVs.sv_index_datasets = SVs_pattern_set;
% SVs.bias = bias;
% SVs.sigma = sigma;

% creating a SvmOut Structure
SVs.index = SVs_index;
SVs.SVs_coeff = SVs_coeff;
SVs.SVs_pattern_set = SVs_pattern_set;
SVs.labels = SVs_labels;
SVs.number = total_SVs; 
SvmOut = SVs;
bias = x(end);
%% SVMout Params
% SVs_lembda = x(1:tot_num_input_patterns,1);  % vlaues of all non-zero alpha
% SVs_index = find(SVs_lembda>tolerance); % all non-zero value of alpha %% SVs index
% SVs_coeff = SVs_lembda(SVs_lembda>tolerance);% vlaues of all non-zero alpha

% SVs_pattern_set = train_matrix(SVs_index,:); % indeces of SVs is used to select patterns from 320x2 train_matrix
% SVs_labels = train_label( SVs_index); % indeces of SVs is used to select labels from 320x1 train_matrix
% size_SVs_pattern_set = size(SVs_pattern_set);
% total_SVs = size_SVs_pattern_set(:,1);


