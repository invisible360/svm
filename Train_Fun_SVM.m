function[SvmOut,bias,x,lambda] = Train_Fun_SVM(C_param,...
    C_plus, C_minus, Sigma_param,train_matrix,train_label,MAxIter)

%train_matrix dim = 320x2
%train_label dim = 320x1

tot_num_input_patterns = length(train_label);% Toal number of input patterns...
...(here, 320). % dimension of label vector is = 320x1_z
%tolerance = 0;
tolerance = 10^-6;% for skipping zero!!_z

gamma = 1/(2*(Sigma_param^2));
clear L  
%% here,
%L = E(i=1:tot_num_input_patterns)E(j=1:tot_num_input_patterns) [yiyj(xi.xj)]
%from Dual objective function %y is label and x is train data
%For RBF, xi.xj=exp(-gamma(||xi-xj||^2))
        
%% loop for calculating a hesian symmetric matrix
for i=1:tot_num_input_patterns
    for j =1:tot_num_input_patterns
        % Finding the H matrix that to be used in QP problem
        L(i,j)= train_label(i)*(train_label(j)*...
            (exp(-gamma*((train_matrix(j,:)-train_matrix(i,:))*...
            (train_matrix(j,:)-train_matrix(i,:))'))));
    end
end
pos_point_index = train_label==1; % collection of positive class indices
neg_point_index = train_label==-1; % collection of negative class indices

%% Finding the ratio of positive and negative class in dataset
%length_pos_points = length(pos_point_index); %length of postive class indices
%length_neg_points = length(neg_point_index); %length of postive class indices
%ratio_neg_pos_points = length_neg_points/length_pos_points; ?????

%clear tau;
%tau = 1;
%a*100;%0.03;
%C_plus = C_minus*tau*ratio_neg_pos_points;

%% Initialization of parameters for solving QP Problem
clear f

%initialize f
fSvm= [ones(1,tot_num_input_patterns) C_param*ones(1,tot_num_input_patterns) 0]; % Multiplying dual optimization problem by '-1'

%initialize Aeq, Beq in inequal constrain 
Aeq =[]; 
beq =[];

%initialize lower bound(lb) and upper bound(ub) in linear constrain
lb = [zeros(1,tot_num_input_patterns) zeros(1,tot_num_input_patterns) -inf]';
ub = [inf(1,tot_num_input_patterns) inf(1,tot_num_input_patterns) inf]';

%initialize A and b in inequal constrain
A = -ones(tot_num_input_patterns,(tot_num_input_patterns*2)+1).*...
    cat(2,L,ones(tot_num_input_patterns,1),...
    train_label,zeros(tot_num_input_patterns,tot_num_input_patterns-1))
b= -ones(tot_num_input_patterns,1);

%% using quadprog optimization solver in MATLAB
tic;
clear x
options = optimset('MaxIter',MAxIter); 
%%%[x,fval,exitflag,output,lambda] = linprog(f,A,b,Aeq,beq,lb,ub,options);
[x,fval,exitflag,output,lambda] = linprog(fSvm,A,b,Aeq,beq,lb,ub,options);
toc;

%% SVMout Params
% SVs_index_row_vector = 1: tot_num_input_patterns; % all non-zero value of alpha %% SVs index
% SVs_index_col_vector = SVs_index_row_vector';
SVs_lembda = x(1:tot_num_input_patterns);  % vlaues of all non-zero alpha
SVs_index = find(SVs_lembda>tolerance); % all non-zero value of alpha %% SVs index
SVs_coeff = SVs_lembda(SVs_lembda>tolerance);% vlaues of all non-zero alpha

% zi_index_row_vector = tot_num_input_patterns+1 : tot_num_input_patterns*2; % all non-zero value of alpha %% SVs index
% zi_index = zi_index_row_vector';
% zi_coeff = x(tot_num_input_patterns+1:tot_num_input_patterns*2);
bias = x(2*tot_num_input_patterns+1:end);

SVs_pattern_set = train_matrix(SVs_index,:); % indeces of SVs is used to select patterns from 320x2 train_matrix
SVs_labels = train_label( SVs_index); % indeces of SVs is used to select labels from 320x1 train_matrix
size_SVs_pattern_set = size(SVs_pattern_set);
total_SVs = size_SVs_pattern_set(:,1);

% creating a SvmOut Structure
SVs.index = SVs_index;
SVs.SVs_coeff = SVs_coeff;
SVs.SVs_pattern_set = SVs_pattern_set;
SVs.labels = SVs_labels;
SVs.number = total_SVs; 
SvmOut = SVs;

%% 
% num_pos_SVs = length(find( SVs_labels==1));
% num_neg_SVs = length(find( SVs_labels==-1));
% 
% %%%FINDING bias (b); ****REMEMBER TO USE THE bias value found in this
% %%%way one has to write the decision function as: f(x) = w*x + b ; that
% %%%is , to use 'PLUS'bias for the function expression.%%%%%%%%%%%%%
% 
% pos_SVs_pattern_set_coef =  SVs_coeff( SVs_labels==1); % positive alpna value __x1
% neg_SVs_pattern_set_coef = SVs_coeff( SVs_labels==-1); % negative alpha value __x1
% pos_SVs_pattern_set =  SVs_pattern_set((find( SVs_labels==1)),:); % pattern set for positive alpha value __x2
% neg_SVs_pattern_set =  SVs_pattern_set((find( SVs_labels==-1)),:); % pattern set for negative alpha value ___x2
% 
% % excluding outliers or filtering inliers
% for_calculating_b_pos_SVs_pattern_set_coef=  pos_SVs_pattern_set_coef(pos_SVs_pattern_set_coef < C_param - tolerance);%excluding the outliers
% for_calculating_b_neg_SVs_pattern_set_coef = neg_SVs_pattern_set_coef(neg_SVs_pattern_set_coef < C_minus - tolerance);%excluding the outliers
% for_calculating_b_pos_SVs_pattern_set =  pos_SVs_pattern_set((find(pos_SVs_pattern_set_coef < C_plus - tolerance)),:);%excluding the outliers
% for_calculating_b_neg_SVs_pattern_set =  neg_SVs_pattern_set((find(neg_SVs_pattern_set_coef < C_minus - tolerance)),:);%excluding the outliers
% 
% clear pos_b neg_b
% 
% %%%%CONSIDERING THE POSITIVE INLIER SVs.
% 
% if (length(for_calculating_b_pos_SVs_pattern_set_coef)>=1)%In case if all of the SVs are outliers, the code should not stop
% 
%     for pos_i=1:length(for_calculating_b_pos_SVs_pattern_set_coef)
% 
%         if (for_calculating_b_pos_SVs_pattern_set_coef(pos_i)< C_plus - tolerance)% ensuring that, w and b is calculating for inliers
%             clear   pos_for_calculating_b_sum_i  pos_for_calculating_b_coef_label_ker_i
%             pos_for_calculating_b_sum_i = 0;
%             pos_for_calculating_b_coef_label_ker_i = 0;
% 
%             for j = 1:total_SVs
%                 pos_for_calculating_b_coef_label_ker_i =  SVs_coeff(j)*SVs_labels(j)*(exp(-gamma*((for_calculating_b_pos_SVs_pattern_set(pos_i,:) - SVs_pattern_set(j,:))*(for_calculating_b_pos_SVs_pattern_set(pos_i,:) - SVs_pattern_set(j,:))')));%FINDING $\alpha_j y_j k(x_i,x_j)$
%                 pos_for_calculating_b_sum_i = pos_for_calculating_b_sum_i + pos_for_calculating_b_coef_label_ker_i;%FINDING $ \sum_j \alpha_j y_j k(x_i,x_j)$
%             end % end of j loop
% 
% %             pos_b(pos_i) = 1- pos_for_calculating_b_sum_i;% $b = y_i - \sum_j \alpha_j y_j k(x_i,x_j)$
% 
%         else
% %             pos_b(pos_i) = 0; %???????? ei else a dhukbe kokhn?
%         end % end of nested if
%         %%%THIS IF BLOCK IS IN FACT NOT A MUST TO BE USED. ...
%         ...STILL IT IS USED TO CHECK THAT THE CALCULATIONS/PROCESS ...
%             ...ABOVE ARE CORRECT;IF ANY OF THESE pos_b(i) EQUALS ZERO, ...
%             ...THEN IT MEANS THERE COULD BE SOMETHING WRONG ABOVE (IN THE \alpha...
%             ...CO-EFFICIENTS FOR CALCULATING bias, rho)
%     end   %end of pos_i for 
% 
%     pos_term_rho = (sum(pos_b))/(length(find(for_calculating_b_pos_SVs_pattern_set_coef< C_plus - tolerance)));%TAKING THE AVERAGE OF THE BIAS VALUE DUE TO THE POSITIVE INLIER SVs. ???????
% 
% else
%     pos_term_rho = 0;%In case if all of the SVs are outliers, the code should not stop//
% end % end of if
% 
% 
% %%%%CONSIDERING THE NEGATIVE INLIER SVs.
% 
% if (length(for_calculating_b_neg_SVs_pattern_set_coef)>=1)%In case if all of the SVs are outliers, the code should not stop
% 
% 
%     for neg_i=1:length(for_calculating_b_neg_SVs_pattern_set_coef)
% 
%         if (  for_calculating_b_neg_SVs_pattern_set_coef(neg_i)< C_minus - tolerance)
%             clear   neg_for_calculating_b_sum_i  neg_for_calculating_b_coef_label_ker_i
%             neg_for_calculating_b_sum_i = 0;
%             neg_for_calculating_b_coef_label_ker_i = 0;
% 
%             for j = 1:total_SVs
% 
%                 neg_for_calculating_b_coef_label_ker_i =  SVs_coeff(j)*SVs_labels(j)*(exp(-gamma*((for_calculating_b_neg_SVs_pattern_set(neg_i,:) - SVs_pattern_set(j,:))*(for_calculating_b_neg_SVs_pattern_set(neg_i,:) - SVs_pattern_set(j,:))')));%FINDING $\alpha_j y_j k(x_i,x_j)$
%                 neg_for_calculating_b_sum_i = neg_for_calculating_b_sum_i + neg_for_calculating_b_coef_label_ker_i;%FINDING $ \sum_j \alpha_j y_j k(x_i,x_j)$
%             end% end of j loop
% 
% 
%             neg_b(neg_i) = -1 - neg_for_calculating_b_sum_i;% % $b = y_i - \sum_j \alpha_j y_j k(x_i,x_j)$
% 
%         else
%             neg_b(neg_i) = 0;
%         end %  end of if %%%THIS IF BLOCK IS IN FACT NOT A MUST TO BE USED. STILL IT IS USED TO CHECK THAT THE CALCULATIONS/PROCESS ABOVE ARE CORRECT;IF ANY OF THESE neg_b(i) EQUALS ZERO, THEN IT MEANS THERE COULD BE SOMETHING WRONG ABOVE (IN THE \alpha CO-EFFICIENTS FOR CALCULATING bias, rho)
%     end   %
%     %neg_b;
% 
%     neg_term_rho = (sum(neg_b))/(length(find(for_calculating_b_neg_SVs_pattern_set_coef< C_minus - tolerance)));% TAKING THE AVERAGE OF THE BIAS VALUE DUE TO THE NEGATIVE INLIER SVs.
% 
% else
%     neg_term_rho = 0;%In case if all of the SVs are outliers, the code should not stop
% end
% 
% 
% 
% if (abs(pos_term_rho)<= 0)
%     bias = neg_term_rho;
% elseif (abs(neg_term_rho)<= 0)
%     bias = pos_term_rho;
% else
%     bias = (pos_term_rho + neg_term_rho)/2; %%WHEN ALLTHE SVs ARE OUTLIERS, THIS BIAS VALUE BECOMES zero.
% end



%%NOTE!!%%**FOR nu-SVC*****IN LIBSVM, THEY USE(INTRODUCE) THE INLIER SVs' LABELS (THEY MULTIPLY IT
%%%WITH OTHERS)  TO CLACULATE THE 'pos_term_rho' (THEIR 'r_1') & 'neg_term_rho' (THEIR 'r_2')  AND DO NOT DIVIDE BY 2 BEFORE; WHEREAS, WE DO NOT USE THE INLIER SVs' LABELS FOR
%%%CALCULATING pos_term_rho' (THEIR 'r_1') & 'neg_term_rho' (THEIR 'r_2') . AS A RESULT, OUR 'neg_term_rho' (THEIR 'r_2')
%%%HAS OPPOSITE POLARITY OF THEIR, WHICH IS MEANINGFUL.SO, OUR EXPRESSION LOOKS A BIT DIFFERENT FROM THEM. THESE BOTH ARE THE SAME.


