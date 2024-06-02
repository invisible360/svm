function [bias] = fn_qp_bias ...
    (SvmOut,C_plus,C_minus,tol,Sigma_param)

SVs_coeff = SvmOut.SVs_coeff;
SVs_labels = SvmOut.labels;
SVs_pattern_set = SvmOut.SVs_pattern_set;
total_SVs = SvmOut.number;

pos_SVs_pattern_set_coef =  SVs_coeff( SVs_labels==1);
neg_SVs_pattern_set_coef = SVs_coeff( SVs_labels==-1);
pos_SVs_pattern_set =  SVs_pattern_set((find( SVs_labels==1)),:);
neg_SVs_pattern_set =  SVs_pattern_set((find( SVs_labels==-1)),:);

% excluding outliers or filtering inliers
for_calculating_b_pos_SVs_pattern_set_coef=  pos_SVs_pattern_set_coef(pos_SVs_pattern_set_coef < C_plus - tol);
for_calculating_b_neg_SVs_pattern_set_coef = neg_SVs_pattern_set_coef(neg_SVs_pattern_set_coef < C_minus - tol);
for_calculating_b_pos_SVs_pattern_set =  pos_SVs_pattern_set((find(pos_SVs_pattern_set_coef < C_plus - tol)),:);
for_calculating_b_neg_SVs_pattern_set =  neg_SVs_pattern_set((find(neg_SVs_pattern_set_coef < C_minus - tol)),:);

clear pos_b neg_b

%%%%CONSIDERING THE POSITIVE INLIER SVs.

if (length(for_calculating_b_pos_SVs_pattern_set_coef)>=1)%In case if all of the SVs are outliers, the code should not stop
    
    for pos_i=1:length(for_calculating_b_pos_SVs_pattern_set_coef)
        
        if (  for_calculating_b_pos_SVs_pattern_set_coef(pos_i)< C_plus - tol)% ensuring that, w and b is calculating for inliers
            clear   pos_for_calculating_b_sum_i  pos_for_calculating_b_coef_label_ker_i
            pos_for_calculating_b_sum_i = 0;
            pos_for_calculating_b_coef_label_ker_i = 0;
            
            for j = 1:total_SVs
                
                [krnl] = fn_rbf_kernel(for_calculating_b_pos_SVs_pattern_set,pos_i,SVs_pattern_set,j,Sigma_param);
                pos_for_calculating_b_coef_label_ker_i =  SVs_coeff(j)*SVs_labels(j)*krnl;
                pos_for_calculating_b_sum_i = pos_for_calculating_b_sum_i + pos_for_calculating_b_coef_label_ker_i;
            end % end of j loop
            
            pos_b(pos_i) = 1- pos_for_calculating_b_sum_i;% $b = y_i - \sum_j \alpha_j y_j k(x_i,x_j)$
            
        else
            pos_b(pos_i) = 0; %???????? ei else a dhukbe kokhn?
        end % end of nested if
    end   %end of pos_i for
    
    pos_term_rho = (sum(pos_b))/(length(find(for_calculating_b_pos_SVs_pattern_set_coef< C_plus - tol)));%TAKING THE AVERAGE OF THE BIAS VALUE DUE TO THE POSITIVE INLIER SVs. ???????
    
else
    pos_term_rho = 0;%In case if all of the SVs are outliers, the code should not stop//
end % end of if


%%%%CONSIDERING THE NEGATIVE INLIER SVs.

if (length(for_calculating_b_neg_SVs_pattern_set_coef)>=1)%In case if all of the SVs are outliers, the code should not stop
    
    
    for neg_i=1:length(for_calculating_b_neg_SVs_pattern_set_coef)
        
        if (  for_calculating_b_neg_SVs_pattern_set_coef(neg_i)< C_minus - tol)
            clear   neg_for_calculating_b_sum_i  neg_for_calculating_b_coef_label_ker_i
            neg_for_calculating_b_sum_i = 0;
            neg_for_calculating_b_coef_label_ker_i = 0;
            
            for j = 1:total_SVs
                [krnl] = fn_rbf_kernel(for_calculating_b_neg_SVs_pattern_set,neg_i,SVs_pattern_set,j,Sigma_param);
                neg_for_calculating_b_coef_label_ker_i =  SVs_coeff(j)*SVs_labels(j)*krnl;%FINDING $\alpha_j y_j k(x_i,x_j)$
                neg_for_calculating_b_sum_i = neg_for_calculating_b_sum_i + neg_for_calculating_b_coef_label_ker_i;%FINDING $ \sum_j \alpha_j y_j k(x_i,x_j)$
            end% end of j loop
            
            
            neg_b(neg_i) = -1 - neg_for_calculating_b_sum_i;% % $b = y_i - \sum_j \alpha_j y_j k(x_i,x_j)$
            
        else
            neg_b(neg_i) = 0;
        end %  end of if %%%THIS IF BLOCK IS IN FACT NOT A MUST TO BE USED. STILL IT IS USED TO CHECK THAT THE CALCULATIONS/PROCESS ABOVE ARE CORRECT;IF ANY OF THESE neg_b(i) EQUALS ZERO, THEN IT MEANS THERE COULD BE SOMETHING WRONG ABOVE (IN THE \alpha CO-EFFICIENTS FOR CALCULATING bias, rho)
    end   %
    %neg_b;
    
    neg_term_rho = (sum(neg_b))/(length(find(for_calculating_b_neg_SVs_pattern_set_coef< C_minus - tol)));% TAKING THE AVERAGE OF THE BIAS VALUE DUE TO THE NEGATIVE INLIER SVs.
    
else
    neg_term_rho = 0;%In case if all of the SVs are outliers, the code should not stop
end



if (abs(pos_term_rho)<= 0)
    bias = neg_term_rho;
elseif (abs(neg_term_rho)<= 0)
    bias = pos_term_rho;
else
    bias = (pos_term_rho + neg_term_rho)/2; %%WHEN ALLTHE SVs ARE OUTLIERS, THIS BIAS VALUE BECOMES zero.
end

