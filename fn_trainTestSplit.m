function [train_matrix,train_label_vector,test_matrix,test_label_vector] = ...
    fn_trainTestSplit (dataset_name, num_of_dataset, num_of_file)

train_matrix = dataset_name(num_of_dataset).x(dataset_name(num_of_dataset).train(num_of_file,:),:);
train_label_vector = dataset_name(num_of_dataset).t(dataset_name(num_of_dataset).train(num_of_file,:),:);
test_matrix = dataset_name(num_of_dataset).x(dataset_name(num_of_dataset).test(num_of_file,:),:);
test_label_vector = dataset_name(num_of_dataset).t(dataset_name(num_of_dataset).test(num_of_file,:),:);


% train_matrix= train_inst_matrix{num_of_file,num_of_dataset};
% train_label_vector= train_inst_label_vector{num_of_file,num_of_dataset};
% test_matrix=test_inst_matrix{num_of_file,num_of_dataset};
% test_label_vector=test_inst_label_vector{num_of_file,num_of_dataset};
