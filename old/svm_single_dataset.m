clc; clear; close all;
%% loading banana benchmark
load benchmarks.mat banana; % load banana data
%% file selection

for a =1:3;
%% train data and lebel splitting by 400x2 and 400x1 dimentions respectively
train_inst_matrix = banana.x(banana.train(a,:),:); % 400x2 train data
train_inst_label_vector = banana.t(banana.train(a,:),:); % 400x1 label corresponding train data

test_inst_matrix = banana.x(banana.test(a,:),:);
test_inst_label_vector = banana.t(banana.test(a,:),:);
%% creating references of traning input data
train_matrix= train_inst_matrix;
train_label_vector = train_inst_label_vector;

test_matrix= test_inst_matrix;
test_label_vector = test_inst_label_vector;


%% creating random ins_matrix and label vector using randperm()
%rand_inst_matrix_index = randperm(size(inst_matrix,1)); %taking random index number...
...in a row matrix format from the size of inst_matrix

%rand_inst_matrix = inst_matrix(rand_inst_matrix_index,:);
%rand_label_vector = label_vector(rand_inst_matrix_index); 

%% preparing train data for Cross Validation
tr_data_set_length = size(train_matrix,1);
fold = 5;
cv_each_set_length = floor(tr_data_set_length/fold); %% dividing train data into 5 fold

% creating C and Sigma Array
CP=2.^(0:2:20); %%C_param
SP = 2.^(-2:2:8);%%Sigma_param
% How to understand that, what dimension of matrix we take for C and Sigma????
% Here it is 11x6

accuracy=zeros(length(CP),length(SP));% Initializing the accuracy matrix

MAxIter = 10000;

%% CV Looping Starts
for c=1:length(CP)
    for s=1:length(SP)
       C_param=CP(c);
       Sigma_param=SP(s);
       ind = 1:size(train_matrix,1); % creating 1x400 matrix from 1 to 400
       
            for i = 1:fold
                test_ind{i}  = (i-1)*cv_each_set_length+1:i*cv_each_set_length; % 1x80
                train_ind{i} = setdiff(ind, test_ind{i}); % 1x320
                
                % splitting the x_train, y_train and x_test, y_test for
                % cross validation
                
                %train_matrix = rand_inst_matrix(train_ind{i},:);% x_train 320x2
                %train_label = rand_label_vector(train_ind{i}); %y_train 320x1
                %test_matrix = rand_inst_matrix(test_ind{i},:); %x_test 80x2
                %test_label = rand_label_vector(test_ind{i}); %y_test 80x1
                
                cv_train_matrix = train_matrix(train_ind{i},:);% x_train 320x2
                cv_train_label = train_label_vector(train_ind{i}); %y_train 320x1
                cv_test_matrix = train_matrix(test_ind{i},:); %x_test 80x2
                cv_test_label = train_label_vector(test_ind{i}); %y_test 80x1
  
                % skipping the machine from biasing behaviour and to keep it as synchronous
                C_plus = C_param;
                C_minus = C_param;
  
                clear SvmOut bias lambda SVs_set SVs_coeff SVs_labels
                
                %% training function
                [SvmOut,bias,alpha,lambda] = Train_Fun_SVM(C_plus,...
                    C_minus,Sigma_param,cv_train_matrix,cv_train_label,MAxIter);
         
                test_set = cv_test_matrix;
                actual_labels = cv_test_label;
                support_vect_pattern_set = SvmOut.SVs_pattern_set;
                SVs_coeff = SvmOut.SVs_coeff;
                SVs_labels = SvmOut.labels;
                gamma = 1/(2*Sigma_param^2);
                
                %% testing function
                [FP_rate,FN_rate,TotEr_rate,TruePred_rate,tot_test_pattern]...
                    = Test_Fun_SVM(test_set,actual_labels,...
                    support_vect_pattern_set,SVs_coeff,SVs_labels,gamma,bias);

                acc(i) = TruePred_rate;
                clear inst_matrix  test_ind   train_ind
            end
        accuracy(c,s) = mean(acc);
    end %%End of CV s loop
end %%End of CV c loop

%% best params:
 [temp,c]=max(accuracy);%[Y,I] = MAX(X) returns the indices of the maximum values in vector I.% 'temp' contains the maximum value of each of the columns;'c' contains the row-indices of these values in their columns (of accuracy matrix)
 [temp,s]=max(temp);%this 'temp' contains the maximum value (element) of the last 'temp'(row-array);g contains the position of this value in that array. 
 c=c(s); % 'c' takes the mentioned index at the g-th position of the last 'c'-array.At this index of the CP- array gives the maximum accuracy.
 c_param =CP(c); % 'c_param' takes the value from the 'CP' array that gives the maximum accuracy.
 Sigma_param=SP(s);% 'g_param' takes the value from the 'GP' array that gives the maximum accuracy.
  From_CV_opt_c = c_param
  From_CV_opt_s = Sigma_param

  
  [SvmOut,bias,alpha,lambda] = Train_Fun_SVM(From_CV_opt_c,...
                    From_CV_opt_c,From_CV_opt_s,train_matrix,train_label_vector,MAxIter);
                
 test_set = test_matrix;
 actual_labels = test_label_vector;
 support_vect_pattern_set = SvmOut.SVs_pattern_set;
 SVs_coeff = SvmOut.SVs_coeff;
 SVs_labels = SvmOut.labels;
 gamma = 1/(2*From_CV_opt_s^2);
 
 [FP_rate,FN_rate,TotEr_rate,TruePred_rate,tot_test_pattern]...
                    = Test_Fun_SVM(test_set,actual_labels,...
                    support_vect_pattern_set,SVs_coeff,SVs_labels,gamma,bias);

                final_acc(a) = TruePred_rate;
end
                