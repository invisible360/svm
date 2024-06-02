function[FP_rate,FN_rate,TotEr_rate,TruePred_rate,...
    tot_test_pattern] = Test_Fun_SVM...
    (test_set,actual_labels,support_vect_pattern_set,SVs_coeff,...
    SVs_labels,gamm,bias)

p = 0;
q =0;
r=0;
X2 =test_set;
for q=1:size(X2,1)%length(X2)
    r=r+1;
    train_sum = 0.0;
    train_each_term =0;
    tic;
    for p= 1:size(support_vect_pattern_set,1)
        train_vect_differ = (support_vect_pattern_set(p,:) - X2(q,:));
        train_vect_distance = train_vect_differ*train_vect_differ';
        train_expon_term = exp(-gamm*train_vect_distance);
        train_each_term = (SVs_coeff(p))*(SVs_labels(p))*train_expon_term; %it is positive or negative if the corresponding SVs are from positive or negative classes respectively
        train_sum = train_sum + train_each_term;
    end
    decisn_func_train_data(r) = train_sum + bias;% decision function value for each of the test data.This value leads to a prediction about the corresponding test data to be in the positive or negative class if its(this function's) value is positive or negative respectively.
   toc; 
 %   t(r)=toc;
end
%aver_time_classif_each_pattern = mean(t);% Gives the average time to classify each novel pattern (using the above support vectors or kernels)
%tot_time_classif_all_patterns = sum(t);

tot_test_pattern = size(X2,1);
z = decisn_func_train_data;
predict_labels =(sign(z))';


dif_predic_actual_labels = predict_labels - actual_labels;
num_flase_pos = length(find(dif_predic_actual_labels == 2)); %positive er jonno false
num_flase_neg = length(find(dif_predic_actual_labels == -2));
num_wrong_prediction = num_flase_pos + num_flase_neg;
num_correct_prediction = length(find(dif_predic_actual_labels == 0));

FP_rate = (num_flase_pos*100)/length(find(actual_labels==-1)); 
FN_rate = (num_flase_neg*100)/length(find(actual_labels==1));
TotEr_rate = (num_wrong_prediction*100)/tot_test_pattern;
TruePred_rate = (num_correct_prediction*100)/tot_test_pattern;
