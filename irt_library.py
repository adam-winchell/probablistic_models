import pickle
import numpy as np
from sklearn.metrics import mean_squared_error
import pystan
import os
import utils
import stan_utility
import random
from sklearn.metrics import roc_auc_score


def stan_irt_gaussian_hierarchical(x_train, x_test, y_train, y_test, fold_num, model_name, num_draws=2000,
                                      debug=False):
    model_code = """
        data {
            int<lower=0> num_students_total;     
            int<lower=0> num_students_train;  
            int<lower=0> num_students_test;        
            int<lower=0> num_questions;             
            int<lower=0> num_train;
            int<lower=0> num_predictors;
            int<lower=0> num_test;

            int<lower=1> student_idx_train[num_train];
            int<lower=1> question_idx_train[num_train];
            int<lower=1, upper=2> question_type_train[num_train];
            int<lower=1, upper=2> experiment_version_train[num_train];

            int<lower=1> student_idx_test[num_test];
            int<lower=1> question_idx_test[num_test];
            int<lower=1, upper=2> question_type_test[num_test];
            int<lower=1> highlight_idx_test[num_test];
            int<lower=1, upper=2> experiment_version_test[num_test];

            int<lower=0,upper=1> quiz_responses[num_train];  

            vector[num_predictors] highlights[num_students_train];
            vector[num_predictors] highlights_test[num_students_test];
        }
        parameters { 
            vector[num_questions] difficulties_raw;
            vector[num_students_total] abilities_raw;

            real<lower=0> sigma_d;
            real<lower=0> sigma_a;
            vector<lower=0>[num_questions] sigma_weights;

            vector[2] question_type_offset;
            
            vector[2] experiment_version_offset;

            vector[num_predictors] weights_raw[num_questions];
        }
        transformed parameters {
            vector[num_train] highlight_effect;
            vector[num_predictors] weights[num_questions];
            vector[num_questions] difficulties;
            vector[num_students_total] abilities;
            
            difficulties = 0 + sigma_d * difficulties_raw;  //non-centered paramaterization
            abilities = 0 + sigma_a * abilities_raw; 
            
            for (row in 1:num_questions){
                weights[row] = sigma_weights[row] * weights_raw[row];
            }
            
            for(i in 1:num_train){
                highlight_effect[i] = dot_product( highlights[student_idx_train[i]], weights[question_idx_train[i]] );
            } 
            
           
        }
        model {
            abilities_raw ~ normal(0, 1);
            sigma_a ~ normal (0, 2.5) ; 

            difficulties_raw ~ normal(0, 1); 
            sigma_d ~ normal (0, 2.5) ;
            
            sigma_weights ~ normal(0, 2.5);
            
            for (row in 1:num_questions){
                weights_raw[row] ~ normal(0, 1);
            }

            question_type_offset ~ normal (0,1);   
            
            experiment_version_offset ~ normal(0, 1);

            quiz_responses ~ bernoulli_logit ( abilities[student_idx_train] + difficulties[question_idx_train] + question_type_offset[question_type_train] + highlight_effect + experiment_version_offset[experiment_version_train]) ;    
        }

        generated quantities {  
            vector[num_test] predictions;

            for(i in 1:num_test) { 
                predictions[i] = bernoulli_logit_rng( abilities[student_idx_test[i]] + difficulties[question_idx_test[i]]  + question_type_offset[question_type_test[i]] + dot_product(highlights_test[highlight_idx_test[i]],weights[question_idx_test[i]] ) + experiment_version_offset[experiment_version_test[i]] );    
            }
        }
        """
    end_of_predictors = int(x_train.shape[1] - y_train.shape[1] - y_train.shape[
        1])  # used to grab the question type data -- we subtract y_train.shape[1] twice to obtain question_type and experiment_version

    extra_predictors_train = x_train[:, end_of_predictors:].astype(int)
    experiment_version_train = np.ravel(extra_predictors_train[:, y_train.shape[1]:])
    question_type_train = np.ravel(extra_predictors_train[:, :y_train.shape[1]])
    x_train = x_train[:, :end_of_predictors]

    extra_predictors_test = x_test[:, end_of_predictors:].astype(int)
    experiment_version_test = np.ravel(extra_predictors_test[:, y_train.shape[1]:])
    question_type_test = np.ravel(extra_predictors_test[:, :y_train.shape[1]])
    x_test = x_test[:, :end_of_predictors]

    num_students = y_train.shape[0] + y_test.shape[0]  # need to create abilities for those students in the test set
    num_questions = 12

    student_idx_train = np.ravel(np.tile(np.arange(y_train.shape[0]), (y_train.shape[1], 1)).T)
    question_idx_train = np.ravel(np.tile(np.arange(y_train.shape[1]), (y_train.shape[0], 1))) % num_questions

    question_idx_test = np.ravel(np.tile(np.arange(y_test.shape[1]), (y_test.shape[0], 1))) % num_questions
    student_idx_test = np.ravel(np.tile(np.arange(num_students), (y_test.shape[1], 1)).T)[
                       student_idx_train.shape[0]:]

    highlight_idx_test = student_idx_test - x_train.shape[
        0]  # i can reuse student_idx_train for highlight_idx, but for highlights test we have to shift the numbers over to start at 0

    quiz_responses = np.ravel(y_train).astype(int)
    quiz_responses_test = np.ravel(y_test).astype(int)

    model_data = {
        'num_students_total': num_students,
        'num_students_train': x_train.shape[0],
        'num_students_test': x_test.shape[0],
        'num_questions': num_questions,
        'num_train': quiz_responses.shape[0],
        'student_idx_train': student_idx_train + 1,  # stan counts starting at 1
        'question_idx_train': question_idx_train + 1,
        'quiz_responses': quiz_responses,
        'num_predictors': x_train.shape[1],
        'highlights': x_train,
        'question_idx_test': question_idx_test + 1,
        'student_idx_test': student_idx_test + 1,
        'highlights_test': x_test,
        'num_test': quiz_responses_test.shape[0],
        'question_type_train': question_type_train + 1,
        'question_type_test': question_type_test + 1,
        'highlight_idx_test': highlight_idx_test + 1,
        'experiment_version_train': experiment_version_train + 1,
        'experiment_version_test': experiment_version_test + 1
    }

    if os.name != 'nt':
        os.environ['STAN_NUM_THREADS'] = "16"
        extra_compile_args = ['-pthread', '-DSTAN_THREADS']

    if fold_num == 1:
        if os.name == 'nt':
            sm = pystan.StanModel(model_code=model_code)
        else:
            sm = pystan.StanModel(model_code=model_code, extra_compile_args=extra_compile_args)

        pickle.dump(sm, open(model_name + '.pkl', 'wb'))

    else:
        sm = pickle.load(open(model_name + '.pkl', 'rb'))

    if os.name == 'nt':
        fit = sm.sampling(data=model_data, iter=num_draws, chains=4, seed=42, n_jobs=1,
                          control={'adapt_delta': 0.999, 'stepsize': 0.001,
                                   'max_treedepth': 20})  # must set n_jobs=1 for windows
    else:
        fit = sm.sampling(data=model_data, iter=num_draws, chains=4, seed=42, n_jobs=-1,
                          control={'adapt_delta': 0.999, 'stepsize': 0.001, 'max_treedepth': 20})

    if debug:
        stan_utility.check_all_diagnostics(fit)

    # fig = fit.plot(pars=('sigma_weights'))
    # plt.savefig('temp')

    predictions = fit['predictions']
    predictions = predictions[int(predictions.shape[0] / 2):]
    predictions = np.reshape(np.mean(predictions, axis=0), newshape=y_test.shape)

    auc = roc_auc_score(y_test, predictions, average='macro')

    prediction = np.sum(predictions, axis=1)  # get the overall quiz score per participant
    error = mean_squared_error(prediction, np.sum(y_test, axis=1))

    utils.pickle_out(model_name, {'extract': fit.extract(), 'sampler_params': fit.get_sampler_params()},
                     add_current_time=True)

    return error, auc


def stan_irt_gaussian(x_train, x_test, y_train, y_test, fold_num, model_name, num_draws=2000, debug=False):
    model_code = """
    data {
        int<lower=0> num_students_total;     
        int<lower=0> num_students_train;  
        int<lower=0> num_students_test;        
        int<lower=0> num_questions;             
        int<lower=0> num_train;
        int<lower=0> num_predictors;
        int<lower=0> num_test;
        
        int<lower=1> student_idx_train[num_train];
        int<lower=1> question_idx_train[num_train];
        int<lower=1, upper=2> question_type_train[num_train];
        int<lower=1, upper=2> experiment_version_train[num_train];
        
        int<lower=1> student_idx_test[num_test];
        int<lower=1> question_idx_test[num_test];
        int<lower=1, upper=2> question_type_test[num_test];
        int<lower=1> highlight_idx_test[num_test];
        int<lower=1, upper=2> experiment_version_test[num_test];
        
        int<lower=0,upper=1> quiz_responses[num_train];  
                
        vector[num_predictors] highlights[num_students_train];
        vector[num_predictors] highlights_test[num_students_test];
    }
    parameters { 
        vector[num_questions] difficulties_raw;
        vector[num_students_total] abilities_raw;

        real<lower=0> sigma_d;
        real<lower=0> sigma_a;

        vector[2] question_type_offset;
        
        vector[2] experiment_version_offset;
        
        vector[num_predictors] weights[num_questions];
    }
    transformed parameters {
        vector[num_train] highlight_effect;
        vector[num_questions] difficulties;
        vector[num_students_total] abilities;
        
        difficulties = 0 + sigma_d * difficulties_raw;  //non-centered paramaterization
        abilities = 0 + sigma_a * abilities_raw; 
        
        for(i in 1:num_train){
            highlight_effect[i] = dot_product( highlights[student_idx_train[i]], weights[question_idx_train[i]] );
        } 
    }
    model {
        abilities_raw ~ normal(0, 1);
        sigma_a ~ normal (0, 2.5) ; 

        difficulties_raw ~ normal(0, 1); 
        sigma_d ~ normal (0, 2.5) ;
        
        for (row in 1:num_questions){
            weights[row] ~ normal(0,1);
        }
        
        question_type_offset ~ normal (0,1);   
        
        experiment_version_offset ~ normal(0, 1);
        
        quiz_responses ~ bernoulli_logit ( abilities[student_idx_train] + difficulties[question_idx_train] + question_type_offset[question_type_train] + highlight_effect + experiment_version_offset[experiment_version_train]) ;    
    }

    generated quantities {  //used for prediction
        vector[num_test] predictions;

        for(i in 1:num_test) { 
            predictions[i] = bernoulli_logit_rng( abilities[student_idx_test[i]] + difficulties[question_idx_test[i]]  + question_type_offset[question_type_test[i]] + dot_product(highlights_test[highlight_idx_test[i]],weights[question_idx_test[i]] ) + experiment_version_offset[experiment_version_test[i]]);    
        }
    }
    """
    end_of_predictors = int(x_train.shape[1] - y_train.shape[1] - y_train.shape[
        1])  # used to grab the question type data -- we subtract y_train.shape[1] twice to obtain question_type and experiment_version

    extra_predictors_train = x_train[:, end_of_predictors:].astype(int)
    experiment_version_train = np.ravel(extra_predictors_train[:, y_train.shape[1]:])
    question_type_train = np.ravel(extra_predictors_train[:, :y_train.shape[1]])
    x_train = x_train[:, :end_of_predictors]

    extra_predictors_test = x_test[:, end_of_predictors:].astype(int)
    experiment_version_test = np.ravel(extra_predictors_test[:, y_train.shape[1]:])
    question_type_test = np.ravel(extra_predictors_test[:, :y_train.shape[1]])
    x_test = x_test[:, :end_of_predictors]

    num_students = y_train.shape[0] + y_test.shape[0]  # need to create abilities for those students in the test set
    num_questions = 12

    student_idx_train = np.ravel(np.tile(np.arange(y_train.shape[0]), (y_train.shape[1], 1)).T)
    question_idx_train = np.ravel(np.tile(np.arange(y_train.shape[1]), (y_train.shape[0], 1))) % num_questions

    question_idx_test = np.ravel(np.tile(np.arange(y_test.shape[1]), (y_test.shape[0], 1))) % num_questions
    student_idx_test = np.ravel(np.tile(np.arange(num_students), (y_test.shape[1], 1)).T)[
                       student_idx_train.shape[0]:]

    highlight_idx_test = student_idx_test - x_train.shape[
        0]  # i can reuse student_idx_train for highlight_idx, but for highlights test we have to shift the numbers over to start at 0

    quiz_responses = np.ravel(y_train).astype(int)
    quiz_responses_test = np.ravel(y_test).astype(int)

    model_data = {
        'num_students_total': num_students,
        'num_students_train': x_train.shape[0],
        'num_students_test': x_test.shape[0],
        'num_questions': num_questions,
        'num_train': quiz_responses.shape[0],
        'student_idx_train': student_idx_train + 1,  # stan counts starting at 1
        'question_idx_train': question_idx_train + 1,
        'quiz_responses': quiz_responses,
        'num_predictors': x_train.shape[1],
        'highlights': x_train,
        'question_idx_test': question_idx_test + 1,
        'student_idx_test': student_idx_test + 1,
        'highlights_test': x_test,
        'num_test': quiz_responses_test.shape[0],
        'question_type_train': question_type_train + 1,
        'question_type_test': question_type_test + 1,
        'highlight_idx_test': highlight_idx_test + 1,
        'experiment_version_train': experiment_version_train + 1,
        'experiment_version_test': experiment_version_test + 1
    }

    if os.name != 'nt':
        os.environ['STAN_NUM_THREADS'] = "16"
        extra_compile_args = ['-pthread', '-DSTAN_THREADS']

    if fold_num == 1:
        if os.name == 'nt':
            sm = pystan.StanModel(model_code=model_code)
        else:
            sm = pystan.StanModel(model_code=model_code, extra_compile_args=extra_compile_args)

        pickle.dump(sm, open(model_name + '.pkl', 'wb'))

    else:
        sm = pickle.load(open(model_name + '.pkl', 'rb'))



    if os.name == 'nt':
        fit = sm.sampling(data=model_data, iter=num_draws, chains=4, seed=42, n_jobs=1)  # must set n_jobs=1 for windows
    else:
        fit = sm.sampling(data=model_data, iter=num_draws, chains=4, seed=42, n_jobs=-1)

    if debug:
        stan_utility.check_all_diagnostics(fit)

    predictions = fit['predictions']
    predictions = predictions[int(predictions.shape[0] / 2):]
    predictions = np.reshape(np.mean(predictions, axis=0), newshape=y_test.shape)

    auc = roc_auc_score(y_test, predictions, average='macro')

    prediction = np.sum(predictions, axis=1)  # get the overall quiz score per participant
    error = mean_squared_error(prediction, np.sum(y_test, axis=1))

    utils.pickle_out(model_name, {'extract': fit.extract(), 'sampler_params': fit.get_sampler_params()},
                     add_current_time=True)

    return error, auc





def stan_irt_uniform(x_train, x_test, y_train, y_test, fold_num, model_name, num_draws=2000, debug=False):
    model_code = """
    data {
        int<lower=0> num_students_total;     
        int<lower=0> num_students_train;  
        int<lower=0> num_students_test;        
        int<lower=0> num_questions;             
        int<lower=0> num_train;
        int<lower=0> num_predictors;
        int<lower=0> num_test;

        int<lower=1> student_idx_train[num_train];
        int<lower=1> question_idx_train[num_train];
        int<lower=1, upper=2> question_type_train[num_train];
        int<lower=1, upper=2> experiment_version_train[num_train];

        int<lower=1> student_idx_test[num_test];
        int<lower=1> question_idx_test[num_test];
        int<lower=1, upper=2> question_type_test[num_test];
        int<lower=1> highlight_idx_test[num_test];
        int<lower=1, upper=2> experiment_version_test[num_test];

        int<lower=0,upper=1> quiz_responses[num_train];  

        vector[num_predictors] highlights[num_students_train];
        vector[num_predictors] highlights_test[num_students_test];
    }
    parameters { 
        vector[num_questions] difficulties_raw;
        vector[num_students_total] abilities_raw;

        real<lower=0> sigma_d;
        real<lower=0> sigma_a;

        vector[2] question_type_offset;

        vector[2] experiment_version_offset;

        vector[num_predictors] weights[num_questions];
    }
    transformed parameters {
        vector[num_train] highlight_effect;
        vector[num_questions] difficulties;
        vector[num_students_total] abilities;

        difficulties = 0 + sigma_d * difficulties_raw;  //non-centered paramaterization
        abilities = 0 + sigma_a * abilities_raw; 
        
        for(i in 1:num_train){
            highlight_effect[i] = dot_product( highlights[student_idx_train[i]], weights[question_idx_train[i]] );
        } 
    }
    model {
        abilities_raw ~ normal(0, 1);
        sigma_a ~ normal (0, 2.5) ; 

        difficulties_raw ~ normal(0, 1); 
        sigma_d ~ normal (0, 2.5) ;

        for (row in 1:num_questions){
            weights[row] ~ uniform(0,2.5);
        }

        

        question_type_offset ~ normal (0,1);   

        experiment_version_offset ~ normal(0, 1);

        quiz_responses ~ bernoulli_logit ( abilities[student_idx_train] + difficulties[question_idx_train] + question_type_offset[question_type_train] + highlight_effect + experiment_version_offset[experiment_version_train]) ;    
    }

    generated quantities {  //used for prediction
        vector[num_test] predictions;

        for(i in 1:num_test) { 
            predictions[i] = bernoulli_logit_rng( abilities[student_idx_test[i]] + difficulties[question_idx_test[i]]  + question_type_offset[question_type_test[i]] + dot_product(highlights_test[highlight_idx_test[i]],weights[question_idx_test[i]] ) + experiment_version_offset[experiment_version_test[i]]);    
        }
    }
    """
    end_of_predictors = int(x_train.shape[1] - y_train.shape[1] - y_train.shape[
        1])  # used to grab the question type data -- we subtract y_train.shape[1] twice to obtain question_type and experiment_version

    extra_predictors_train = x_train[:, end_of_predictors:].astype(int)
    experiment_version_train = np.ravel(extra_predictors_train[:, y_train.shape[1]:])
    question_type_train = np.ravel(extra_predictors_train[:, :y_train.shape[1]])
    x_train = x_train[:, :end_of_predictors]

    extra_predictors_test = x_test[:, end_of_predictors:].astype(int)
    experiment_version_test = np.ravel(extra_predictors_test[:, y_train.shape[1]:])
    question_type_test = np.ravel(extra_predictors_test[:, :y_train.shape[1]])
    x_test = x_test[:, :end_of_predictors]

    num_students = y_train.shape[0] + y_test.shape[0]  # need to create abilities for those students in the test set
    num_questions = 12

    student_idx_train = np.ravel(np.tile(np.arange(y_train.shape[0]), (y_train.shape[1], 1)).T)
    question_idx_train = np.ravel(np.tile(np.arange(y_train.shape[1]), (y_train.shape[0], 1))) % num_questions

    question_idx_test = np.ravel(np.tile(np.arange(y_test.shape[1]), (y_test.shape[0], 1))) % num_questions
    student_idx_test = np.ravel(np.tile(np.arange(num_students), (y_test.shape[1], 1)).T)[
                       student_idx_train.shape[0]:]

    highlight_idx_test = student_idx_test - x_train.shape[
        0]  # i can reuse student_idx_train for highlight_idx, but for highlights test we have to shift the numbers over to start at 0

    quiz_responses = np.ravel(y_train).astype(int)
    quiz_responses_test = np.ravel(y_test).astype(int)

    model_data = {
        'num_students_total': num_students,
        'num_students_train': x_train.shape[0],
        'num_students_test': x_test.shape[0],
        'num_questions': num_questions,
        'num_train': quiz_responses.shape[0],
        'student_idx_train': student_idx_train + 1,  # stan counts starting at 1
        'question_idx_train': question_idx_train + 1,
        'quiz_responses': quiz_responses,
        'num_predictors': x_train.shape[1],
        'highlights': x_train,
        'question_idx_test': question_idx_test + 1,
        'student_idx_test': student_idx_test + 1,
        'highlights_test': x_test,
        'num_test': quiz_responses_test.shape[0],
        'question_type_train': question_type_train + 1,
        'question_type_test': question_type_test + 1,
        'highlight_idx_test': highlight_idx_test + 1,
        'experiment_version_train': experiment_version_train + 1,
        'experiment_version_test': experiment_version_test + 1
    }

    if os.name != 'nt':
        os.environ['STAN_NUM_THREADS'] = "16"
        extra_compile_args = ['-pthread', '-DSTAN_THREADS']

    if fold_num == 1:
        if os.name == 'nt':
            sm = pystan.StanModel(model_code=model_code)
        else:
            sm = pystan.StanModel(model_code=model_code, extra_compile_args=extra_compile_args)

        pickle.dump(sm, open(model_name + '.pkl', 'wb'))

    else:
        sm = pickle.load(open(model_name + '.pkl', 'rb'))

    if os.name == 'nt':
        fit = sm.sampling(data=model_data, iter=num_draws, chains=4, seed=42, n_jobs=1)  # must set n_jobs=1 for windows
    else:
        fit = sm.sampling(data=model_data, iter=num_draws, chains=4, seed=42, n_jobs=-1)

    if debug:
        stan_utility.check_all_diagnostics(fit)

    predictions = fit['predictions']
    predictions = predictions[int(predictions.shape[0] / 2):]
    predictions = np.reshape(np.mean(predictions, axis=0), newshape=y_test.shape)

    auc = roc_auc_score(y_test, predictions, average='macro')

    prediction = np.sum(predictions, axis=1)  # get the overall quiz score per participant
    error = mean_squared_error(prediction, np.sum(y_test, axis=1))

    utils.pickle_out(model_name, {'extract': fit.extract(), 'sampler_params': fit.get_sampler_params()},
                     add_current_time=True)

    return error, auc

def stan_irt_horseshoe_alt(x_train, x_test, y_train, y_test, fold_num, model_name, num_draws=2000,
                                      debug=False):
    model_code = """
    data {
        int<lower=0> num_students_total;     
        int<lower=0> num_students_train;  
        int<lower=0> num_students_test;        
        int<lower=0> num_questions;             
        int<lower=0> num_train;
        int<lower=0> num_predictors;
        int<lower=0> num_test;

        int<lower=1> student_idx_train[num_train];
        int<lower=1> question_idx_train[num_train];
        int<lower=1, upper=2> question_type_train[num_train];
        int<lower=1, upper=2> experiment_version_train[num_train];


        int<lower=1> student_idx_test[num_test];
        int<lower=1> question_idx_test[num_test];
        int<lower=1, upper=2> question_type_test[num_test];
        int<lower=1> highlight_idx_test[num_test];
        int<lower=1, upper=2> experiment_version_test[num_test];

        int<lower=0,upper=1> quiz_responses[num_train];  

        vector[num_predictors] highlights[num_students_train];
        vector[num_predictors] highlights_test[num_students_test];

        real<lower=1> nu; //degrees of freedom for the half t-priors 
        real < lower =0> scale_global ; // scale for the half -t prior for tau
    }
    parameters { 
        vector[num_questions] difficulties_raw;
        vector[num_students_total] abilities_raw;

        real<lower=0> sigma_d;
        real<lower=0> sigma_a;

        vector[2] question_type_offset;
        
        vector[2] experiment_version_offset;


        // auxiliary horseshoe variables that define the global and local parameters
        vector [num_predictors] z[num_questions];
        real < lower =0> r1_global ;
        real < lower =0> r2_global ;
        vector < lower =0 >[num_predictors] r1_local[num_questions] ;
        vector < lower =0 >[num_predictors] r2_local[num_questions] ;
    }
    transformed parameters {
        real < lower =0> tau; // global shrinkage parameter
        vector < lower =0 >[num_predictors] lambda[num_questions] ; // local shrinkage parameter
        vector [num_predictors] weights[num_questions] ; // regression coefficients
        vector[num_train] highlight_effect;
        vector[num_questions] difficulties;
        vector[num_students_total] abilities;
        
        difficulties = 0 + sigma_d * difficulties_raw;  //non-centered paramaterization
        abilities = 0 + sigma_a * abilities_raw; 

        tau = r1_global * sqrt ( r2_global ) * scale_global;

        for (row in 1:num_questions){
            lambda[row] = r1_local[row] .* sqrt ( r2_local[row] );
            weights[row] = z[row] .* lambda[row] * tau ;
        }     

        for(i in 1:num_train){
            highlight_effect[i] = dot_product( highlights[student_idx_train[i]], weights[question_idx_train[i]] );
        }   

    }
    model {
        // half -t prior for tau
        r1_global ~ normal (0.0 , 1.0 );
        r2_global ~ inv_gamma (0.5, 0.5);

        abilities_raw ~ normal(0, 1);
        sigma_a ~ normal (0, 2.5) ; 

        difficulties_raw ~ normal(0, 1); 
        sigma_d ~ normal (0, 2.5) ;

        // half -t priors for lambdas
        for (row in 1:num_questions){
            z[row] ~ normal (0, 1);
            r1_local[row] ~ normal (0.0 , 1.0);
            r2_local[row] ~ inv_gamma (0.5* nu , 0.5* nu );
        }

        question_type_offset ~ normal (0,1);   
        
        question_type_offset ~ normal (0, 1); 

        quiz_responses ~ bernoulli_logit (abilities[student_idx_train] + difficulties[question_idx_train] + question_type_offset[question_type_train] + highlight_effect + experiment_version_offset[experiment_version_train]) ;    
    }

    generated quantities {  //used for prediction
        vector[num_test] predictions;

        for(i in 1:num_test) { 
            predictions[i] = bernoulli_logit_rng( abilities[student_idx_test[i]] + difficulties[question_idx_test[i]]  + question_type_offset[question_type_test[i]] + dot_product(highlights_test[highlight_idx_test[i]],weights[question_idx_test[i]] )+ experiment_version_offset[experiment_version_test[i]] );    
        }
    }
    """
    end_of_predictors = int(x_train.shape[1] - y_train.shape[1] - y_train.shape[
        1])  # used to grab the question type data -- we subtract y_train.shape[1] twice to obtain question_type and experiment_version

    extra_predictors_train = x_train[:, end_of_predictors:].astype(int)
    experiment_version_train = np.ravel(extra_predictors_train[:, y_train.shape[1]:])
    question_type_train = np.ravel(extra_predictors_train[:, :y_train.shape[1]])
    x_train = x_train[:, :end_of_predictors]

    extra_predictors_test = x_test[:, end_of_predictors:].astype(int)
    experiment_version_test = np.ravel(extra_predictors_test[:, y_train.shape[1]:])
    question_type_test = np.ravel(extra_predictors_test[:, :y_train.shape[1]])
    x_test = x_test[:, :end_of_predictors]

    num_students = y_train.shape[0] + y_test.shape[0]  # need to create abilities for those students in the test set
    num_questions = 12

    student_idx_train = np.ravel(np.tile(np.arange(y_train.shape[0]), (y_train.shape[1], 1)).T)
    question_idx_train = np.ravel(np.tile(np.arange(y_train.shape[1]), (y_train.shape[0], 1))) % num_questions

    question_idx_test = np.ravel(np.tile(np.arange(y_test.shape[1]), (y_test.shape[0], 1))) % num_questions
    student_idx_test = np.ravel(np.tile(np.arange(num_students), (y_test.shape[1], 1)).T)[
                       student_idx_train.shape[0]:]

    highlight_idx_test = student_idx_test - x_train.shape[
        0]  # i can reuse student_idx_train for highlight_idx, but for highlights test we have to shift the numbers over to start at 0

    quiz_responses = np.ravel(y_train).astype(int)
    quiz_responses_test = np.ravel(y_test).astype(int)

    effective_predictors_guess = 12
    scale_global = (effective_predictors_guess * 2) / (
        (x_train.shape[1] - effective_predictors_guess) * np.sqrt(x_train.shape[0]))

    model_data = {
        'num_students_total': num_students,
        'num_students_train': x_train.shape[0],
        'num_students_test': x_test.shape[0],
        'num_questions': num_questions,
        'num_train': quiz_responses.shape[0],
        'student_idx_train': student_idx_train + 1,  # stan counts starting at 1
        'question_idx_train': question_idx_train + 1,
        'quiz_responses': quiz_responses,
        'num_predictors': x_train.shape[1],
        'highlights': x_train,
        'question_idx_test': question_idx_test + 1,
        'student_idx_test': student_idx_test + 1,
        'highlights_test': x_test,
        'num_test': quiz_responses_test.shape[0],
        'question_type_train': question_type_train + 1,
        'question_type_test': question_type_test + 1,
        'highlight_idx_test': highlight_idx_test + 1,
        'nu': 3,
        'scale_global': scale_global,
        'experiment_version_train': experiment_version_train + 1,
        'experiment_version_test': experiment_version_test + 1
    }

    if fold_num == 1:
        sm = pystan.StanModel(model_code=model_code)

    else:
        sm = pickle.load(open(model_name + '.pkl', 'rb'))

    pickle.dump(sm, open(model_name + '.pkl', 'wb'))

    if os.name == 'nt':
        fit = sm.sampling(data=model_data, iter=num_draws, chains=4, seed=42, n_jobs=1,
                          control={'adapt_delta': 0.999, 'stepsize': 0.001,
                                   'max_treedepth': 20})  # must set n_jobs=1 for windows
    else:
        # num_chains = multiprocessing.cpu_count()
        num_chains = 4
        fit = sm.sampling(data=model_data, iter=num_draws, chains=num_chains, seed=42, n_jobs=-1,
                          control={'adapt_delta': 0.999, 'stepsize': 0.001, 'max_treedepth': 20})

    if debug:
        stan_utility.check_all_diagnostics(fit)

    predictions = fit['predictions']
    predictions = predictions[int(predictions.shape[0] / 2):]
    predictions = np.reshape(np.mean(predictions, axis=0), newshape=y_test.shape)

    auc = roc_auc_score(y_test, predictions, average='macro')

    prediction = np.sum(predictions, axis=1)  # get the overall quiz score per participant
    error = mean_squared_error(prediction, np.sum(y_test, axis=1))

    utils.pickle_out(model_name, {'extract': fit.extract(), 'sampler_params': fit.get_sampler_params()},
                     add_current_time=True)

    return error, auc


def stan_irt_regularized_horseshoe(x_train, x_test, y_train, y_test, fold_num, model_name, num_draws=2000,
                                              debug=False):
    model_code = """
    data {
        int<lower=0> num_students_total;     
        int<lower=0> num_students_train;  
        int<lower=0> num_students_test;        
        int<lower=0> num_questions;             
        int<lower=0> num_train;
        int<lower=0> num_predictors;
        int<lower=0> num_test;

        int<lower=1> student_idx_train[num_train];
        int<lower=1> question_idx_train[num_train];
        int<lower=1, upper=2> question_type_train[num_train];
        int<lower=1, upper=2> experiment_version_train[num_train];

        int<lower=1> student_idx_test[num_test];
        int<lower=1> question_idx_test[num_test];
        int<lower=1, upper=2> question_type_test[num_test];
        int<lower=1> highlight_idx_test[num_test];
        int<lower=1, upper=2> experiment_version_test[num_test];

        int<lower=0,upper=1> quiz_responses[num_train];  

        vector[num_predictors] highlights[num_students_train];
        vector[num_predictors] highlights_test[num_students_test];
        
        real < lower =0> scale_global ;// scale for the half -t prior for tau
        real < lower =1> nu_global ; // degrees of freedom for the half -t priors for tau
        real < lower =1> nu_local ; // degrees of freedom for the half -t priors for lambdas
        real < lower =0> slab_scale ; // slab scale for the regularized horseshoe
        real < lower =0> slab_df ; // slab degrees of freedom for the regularized horseshoe
    }
    parameters { 
        vector[num_questions] difficulties_raw;
        vector[num_students_total] abilities_raw;
        
        real<lower=0> sigma_d;
        real<lower=0> sigma_a;

        vector[2] question_type_offset;

        vector[2] experiment_version_offset;
        
        // auxiliary horseshoe+ variables that define the global and local parameters
        vector [num_predictors] z[num_questions];        
        real < lower =0> aux1_global ;
        real < lower =0> aux2_global ;
        vector < lower =0 >[num_predictors] aux1_local[num_questions];
        vector < lower =0 >[num_predictors] aux2_local[num_questions];
        real < lower =0> caux ;
    }
    transformed parameters {
        real < lower =0> tau; // global shrinkage parameter
        vector < lower =0 >[num_predictors] lambda[num_questions] ; // local shrinkage parameter
        vector < lower =0 >[num_predictors] lambda_tilde[num_questions] ; // ’truncated ’ local shrinkage parameter
        real < lower =0> c; // slab scale
        vector [num_predictors] weights[num_questions] ; // regression coefficients
        vector[num_train] highlight_effect; //vectorization helper
        vector[num_questions] difficulties;
        vector[num_students_total] abilities;
        
        difficulties = 0 + sigma_d * difficulties_raw;  //non-centered paramaterization
        abilities = 0 + sigma_a * abilities_raw; 

        
        tau = aux1_global * sqrt ( aux2_global ) * scale_global ;
        c = slab_scale * sqrt ( caux );
        
        
        for (row in 1:num_questions){
            lambda[row] = aux1_local[row] .* sqrt ( aux2_local[row] );
            lambda_tilde[row] = sqrt ( c^2 * square ( lambda[row] ) ./ (c^2 + tau ^2* square ( lambda[row] )) );
            weights[row] = z[row] .* lambda_tilde[row] * tau ;
        }     
        
        for(i in 1:num_train){
            highlight_effect[i] = dot_product( highlights[student_idx_train[i]], weights[question_idx_train[i]] );
        }   
        
    }
    model {
        // half -t priors for lambdas and tau , and inverse - gamma for c^2
        aux1_global ~ normal (0, 1);
        aux2_global ~ inv_gamma (0.5* nu_global , 0.5* nu_global );
        caux ~ inv_gamma (0.5* slab_df , 0.5* slab_df );
        
        // half -t priors for lambdas
        for (row in 1:num_questions){
            z[row] ~ normal (0, 1);
            aux1_local[row] ~ normal (0.0 , 1.0); //J. Piironen and A. Vehtari / Sparsity and regularization in the horseshoe prior 31
            aux2_local[row] ~ inv_gamma (0.5* nu_local , 0.5* nu_local );
        }
        

        abilities_raw ~ normal(0, 1);
        sigma_a ~ normal (0, 2.5) ; 

        difficulties_raw ~ normal(0, 1); 
        sigma_d ~ normal (0, 2.5) ;
        
        question_type_offset ~ normal (0,1);   
        
        experiment_version_offset ~ normal(0, 1);

        quiz_responses ~ bernoulli_logit (abilities[student_idx_train] + difficulties[question_idx_train] + question_type_offset[question_type_train] + highlight_effect+ experiment_version_offset[experiment_version_train]) ;    
    }

    generated quantities {  //used for prediction
        vector[num_test] predictions;

        for(i in 1:num_test) { 
            predictions[i] = bernoulli_logit_rng( abilities[student_idx_test[i]] + difficulties[question_idx_test[i]]  + question_type_offset[question_type_test[i]] + dot_product(highlights_test[highlight_idx_test[i]],weights[question_idx_test[i]] ) + experiment_version_offset[experiment_version_test[i]]);    
        }
    }
    """
    end_of_predictors = int(x_train.shape[1] - y_train.shape[1] - y_train.shape[
        1])  # used to grab the question type data -- we subtract y_train.shape[1] twice to obtain question_type and experiment_version

    extra_predictors_train = x_train[:, end_of_predictors:].astype(int)
    experiment_version_train = np.ravel(extra_predictors_train[:, y_train.shape[1]:])
    question_type_train = np.ravel(extra_predictors_train[:, :y_train.shape[1]])
    x_train = x_train[:, :end_of_predictors]

    extra_predictors_test = x_test[:, end_of_predictors:].astype(int)
    experiment_version_test = np.ravel(extra_predictors_test[:, y_train.shape[1]:])
    question_type_test = np.ravel(extra_predictors_test[:, :y_train.shape[1]])
    x_test = x_test[:, :end_of_predictors]

    num_students = y_train.shape[0] + y_test.shape[0]  # need to create abilities for those students in the test set
    num_questions = 12

    student_idx_train = np.ravel(np.tile(np.arange(y_train.shape[0]), (y_train.shape[1], 1)).T)
    question_idx_train = np.ravel(np.tile(np.arange(y_train.shape[1]), (y_train.shape[0], 1))) % num_questions

    question_idx_test = np.ravel(np.tile(np.arange(y_test.shape[1]), (y_test.shape[0], 1))) % num_questions
    student_idx_test = np.ravel(np.tile(np.arange(num_students), (y_test.shape[1], 1)).T)[
                       student_idx_train.shape[0]:]

    highlight_idx_test = student_idx_test - x_train.shape[
        0]  # i can reuse student_idx_train for highlight_idx, but for highlights test we have to shift the numbers over to start at 0

    quiz_responses = np.ravel(y_train).astype(int)
    quiz_responses_test = np.ravel(y_test).astype(int)

    effective_predictors_guess = 12
    scale_global = (effective_predictors_guess * 2) / (
        (x_train.shape[1] - effective_predictors_guess) * np.sqrt(x_train.shape[0]))

    model_data = {
        'num_students_total': num_students,
        'num_students_train': x_train.shape[0],
        'num_students_test': x_test.shape[0],
        'num_questions': num_questions,
        'num_train': quiz_responses.shape[0],
        'student_idx_train': student_idx_train + 1,  # stan counts starting at 1
        'question_idx_train': question_idx_train + 1,
        'quiz_responses': quiz_responses,
        'num_predictors': x_train.shape[1],
        'highlights': x_train,
        'question_idx_test': question_idx_test + 1,
        'student_idx_test': student_idx_test + 1,
        'highlights_test': x_test,
        'num_test': quiz_responses_test.shape[0],
        'question_type_train': question_type_train + 1,
        'question_type_test': question_type_test + 1,
        'highlight_idx_test': highlight_idx_test + 1,
        'scale_global': scale_global,
        'nu_local': 3,
        'nu_global': 3,
        'slab_scale': 2,  # scale for relevant weights
        'slab_df': 8,  # effective degrees of freedom for relevant weights
        'experiment_version_train': experiment_version_train + 1,
        'experiment_version_test': experiment_version_test + 1
    }

    if fold_num == 1:
        sm = pystan.StanModel(model_code=model_code)

    else:
        sm = pickle.load(open(model_name + '.pkl', 'rb'))

    pickle.dump(sm, open(model_name + '.pkl', 'wb'))

    if os.name == 'nt':
        fit = sm.sampling(data=model_data, iter=num_draws, chains=4, seed=42, n_jobs=1,
                          control={'adapt_delta': 0.999, 'stepsize': 0.001,
                                   'max_treedepth': 20})  # must set n_jobs=1 for windows
    else:
        # num_chains = multiprocessing.cpu_count()
        num_chains = 4
        fit = sm.sampling(data=model_data, iter=num_draws, chains=num_chains, seed=42, n_jobs=-1,
                          control={'adapt_delta': 0.999, 'stepsize': 0.001, 'max_treedepth': 20})

    if debug:
        stan_utility.check_all_diagnostics(fit)

    predictions = fit['predictions']
    predictions = predictions[int(predictions.shape[0] / 2):]
    predictions = np.reshape(np.mean(predictions, axis=0), newshape=y_test.shape)

    auc = roc_auc_score(y_test, predictions, average='macro')

    prediction = np.sum(predictions, axis=1)  # get the overall quiz score per participant
    error = mean_squared_error(prediction, np.sum(y_test, axis=1))

    utils.pickle_out(model_name, {'extract': fit.extract(), 'sampler_params': fit.get_sampler_params()},
                     add_current_time=True)

    return error, auc


def stan_irt(x_train, x_test, y_train, y_test, fold_num, model_name, num_draws=2000, debug=False):
    model_code = """
    data {
        int<lower=0> num_students;             
        int<lower=0> num_questions;             
        int<lower=0> num_train;
        int<lower=0> num_test;

        int<lower=1,upper=num_students> student_idx_train[num_train];
        int<lower=1,upper=num_students> question_idx_train[num_train];
        int<lower=1, upper=2> question_type_train[num_train];
        int<lower=1, upper=2> experiment_version_train[num_train];

        int<lower=1,upper=num_students> student_idx_test[num_test];
        int<lower=1,upper=num_students> question_idx_test[num_test];
        int<lower=1, upper=2> question_type_test[num_test];
        int<lower=1, upper=2> experiment_version_test[num_test];

        int<lower=0,upper=1> quiz_responses[num_train];  
    }
    parameters { 
        vector[num_questions] difficulties_raw;
        real<lower=0> sigma_d;

        vector[num_students] abilities_raw;
        real<lower=0> sigma_a;

        vector[2] question_type_offset;
        
        vector[2] experiment_version_offset;
    }
    transformed parameters {
        vector[num_questions] difficulties;
        vector[num_students] abilities;

        difficulties = 0 + sigma_d * difficulties_raw;
        abilities = 0 + sigma_a * abilities_raw;     // a non-centered parameterization
    }
    model {

        abilities_raw ~ normal(0, 1);
        sigma_a ~ normal (0, 2.5) ; 

        difficulties_raw ~ normal(0, 1); 
        sigma_d ~ normal (0, 2.5) ;

        question_type_offset ~ normal (0, 2.5);  
        
        experiment_version_offset ~ normal(0, 2.5);

        quiz_responses ~ bernoulli_logit ( abilities[student_idx_train] + difficulties[question_idx_train] + question_type_offset[question_type_train] + experiment_version_offset[experiment_version_train]) ;    
    }

    generated quantities {  //used for prediction
        vector[num_test] predictions;

        for(i in 1:num_test) { 
            predictions[i] = bernoulli_logit_rng( abilities[student_idx_test[i]] + difficulties[question_idx_test[i]]  + question_type_offset[question_type_test[i]] + experiment_version_offset[experiment_version_test[i]]);    
        }
    }
    """

    end_of_predictors = int(x_train.shape[1] - y_train.shape[1] - y_train.shape[
        1])  # used to grab the question type data -- we subtract y_train.shape[1] twice to obtain question_type and experiment_version

    extra_predictors_train = x_train[:, end_of_predictors:].astype(int)
    experiment_version_train = np.ravel(extra_predictors_train[:, y_train.shape[1]:])
    question_type_train = np.ravel(extra_predictors_train[:, :y_train.shape[1]])
    x_train = x_train[:, :end_of_predictors]

    extra_predictors_test = x_test[:, end_of_predictors:].astype(int)
    experiment_version_test = np.ravel(extra_predictors_test[:, y_train.shape[1]:])
    question_type_test = np.ravel(extra_predictors_test[:, :y_train.shape[1]])
    x_test = x_test[:, :end_of_predictors]

    num_students = y_train.shape[0] + y_test.shape[0]  # need to create abilities for those students in the test set
    num_questions = 12

    student_idx_train = np.ravel(np.tile(np.arange(y_train.shape[0]), (y_train.shape[1], 1)).T)
    question_idx_train = np.ravel(np.tile(np.arange(y_train.shape[1]), (
        y_train.shape[0], 1))) % num_questions  # we mod 12 because mc and fb questions are the same

    question_idx_test = np.ravel(np.tile(np.arange(y_test.shape[1]), (y_test.shape[0], 1))) % num_questions
    student_idx_test = np.ravel(np.tile(np.arange(num_students), (y_test.shape[1], 1)).T)[
                       student_idx_train.shape[0]:]

    quiz_responses = np.ravel(y_train).astype(int)
    quiz_responses_test = np.ravel(y_test).astype(int)

    model_data = {
        'num_students': num_students,
        'num_questions': num_questions,
        'num_train': quiz_responses.shape[0],
        'student_idx_train': student_idx_train + 1,  # stan counts starting at 1
        'question_idx_train': question_idx_train + 1,
        'quiz_responses': quiz_responses,
        'question_idx_test': question_idx_test + 1,
        'student_idx_test': student_idx_test + 1,
        'num_test': quiz_responses_test.shape[0],
        'question_type_train': question_type_train + 1,
        'question_type_test': question_type_test + 1,
        'experiment_version_train': experiment_version_train + 1,
        'experiment_version_test': experiment_version_test + 1
    }

    if fold_num == 1:
        sm = pystan.StanModel(model_code=model_code)

    else:
        sm = pickle.load(open(model_name + '.pkl', 'rb'))

    pickle.dump(sm, open(model_name + '.pkl', 'wb'))

    if os.name == 'nt':
        fit = sm.sampling(data=model_data, iter=num_draws, chains=4, seed=42, n_jobs=1)  # must set n_jobs=1 for windows
    else:
        fit = sm.sampling(data=model_data, iter=num_draws, chains=4, seed=42, n_jobs=-1)

    # fig = fit.plot(pars=('mu_a','mu_d','sigma_a','sigma_d'))
    # plt.show()
    # print(fit)
    if debug:
        stan_utility.check_all_diagnostics(fit)

    predictions = fit['predictions']
    predictions = predictions[int(predictions.shape[0] / 2):]
    predictions = np.reshape(np.mean(predictions, axis=0), newshape=y_test.shape)

    auc = roc_auc_score(y_test, predictions, average='macro')

    prediction = np.sum(predictions, axis=1)  # get the overall quiz score per participant
    error = mean_squared_error(prediction, np.sum(y_test, axis=1))

    utils.pickle_out(model_name, {'extract': fit.extract(), 'sampler_params': fit.get_sampler_params()},
                     add_current_time=True)

    return error, auc


def stan_irt_predict_highlights(x_train, x_test, y_train, y_test, fold_num, model_name, num_draws=2000, debug=False):

    def remove_entries_by_passage(sentences):
        random.seed(42)

        passage1 = sentences[:, list(range(27))].astype(float)
        passage2 = sentences[:, list(range(27, 69))].astype(float)
        passage3 = sentences[:, list(range(69, 117))].astype(float)

        for student in range(passage1.shape[0]):
            num = random.randint(0, 2)
            if num == 0:
                passage1[student] = np.array([np.nan] * passage1.shape[1])
            elif num == 1:
                passage2[student] = np.array([np.nan] * passage2.shape[1])
            else:
                passage3[student] = np.array([np.nan] * passage3.shape[1])

        passages = np.hstack((passage1, passage2, passage3))
        indices = np.argwhere(np.isnan(passages))

        return passages, indices


    model_code = """
    data {
        int<lower=0> num_students;             
        int<lower=0> num_questions;             
        int<lower=0> num_train;
        int<lower=0> num_test;

        int<lower=1,upper=num_students> student_idx_train[num_train];
        int<lower=1,upper=num_students> question_idx_train[num_train];
        int<lower=1, upper=2> experiment_version_train[num_train];

        int<lower=1,upper=num_students> student_idx_test[num_test];
        int<lower=1,upper=num_students> question_idx_test[num_test];
        int<lower=1, upper=2> experiment_version_test[num_test];

        int<lower=0,upper=1> quiz_responses[num_train];  
    }
    parameters { 
        vector[num_questions] difficulties_raw;
        real<lower=0> sigma_d;

        vector[num_students] abilities_raw;
        real<lower=0> sigma_a;

        vector[2] experiment_version_offset;
    }
    transformed parameters {
        vector[num_questions] difficulties;
        vector[num_students] abilities;

        difficulties = 0 + sigma_d * difficulties_raw;
        abilities = 0 + sigma_a * abilities_raw;     // a non-centered parameterization
    }
    model {

        abilities_raw ~ normal(0, 1);
        sigma_a ~ normal (0, 2.5) ; 

        difficulties_raw ~ normal(0, 1); 
        sigma_d ~ normal (0, 2.5) ;

        experiment_version_offset ~ normal(0, 2.5);

        quiz_responses ~ bernoulli_logit ( abilities[student_idx_train] + difficulties[question_idx_train] + experiment_version_offset[experiment_version_train]) ;    
    }

    generated quantities {  //used for prediction
        vector[num_test] predictions;

        for(i in 1:num_test) { 
            predictions[i] = bernoulli_logit_rng( abilities[student_idx_test[i]] + difficulties[question_idx_test[i]]  + experiment_version_offset[experiment_version_test[i]] );    
        }
    }
    """

    end_of_predictors = int(x_train.shape[1] - y_train.shape[1] - y_train.shape[
        1])  # used to grab the question type data -- we subtract y_train.shape[1] twice to obtain question_type and experiment_version

    extra_predictors_train = x_train[:, end_of_predictors:].astype(int)
    experiment_version_train = extra_predictors_train[:, y_train.shape[1]:]
    question_type_train = np.ravel(extra_predictors_train[:, :y_train.shape[1]])
    x_train = x_train[:, :end_of_predictors]

    extra_predictors_test = x_test[:, end_of_predictors:].astype(int)
    experiment_version_test = extra_predictors_test[:, y_train.shape[1]:]

    question_type_test = np.ravel(extra_predictors_test[:, :y_train.shape[1]])
    x_test = x_test[:, :end_of_predictors]


    experiment_version_train = np.ravel(np.array([[experiment_version_train[idx][0]]*x_train.shape[1] for idx in range(experiment_version_train.shape[0])]))

    experiment_version_test = np.array([[experiment_version_test[idx][0]]*x_test.shape[1] for idx in range(experiment_version_test.shape[0])])


    num_students = x_train.shape[0] + x_test.shape[0]  # need to create abilities for those students in the test set
    num_questions = x_train.shape[1]

    student_idx_train = np.ravel(np.tile(np.arange(x_train.shape[0]), (x_train.shape[1], 1)).T)
    question_idx_train = np.ravel(np.tile(np.arange(x_train.shape[1]), (x_train.shape[0], 1)))

    question_idx_test = np.tile(np.arange(x_test.shape[1]), (x_test.shape[0], 1))
    student_idx_test = np.ravel((np.tile(np.arange(num_students), (x_test.shape[1], 1)).T))[student_idx_train.shape[0]:]

    quiz_responses = np.ravel(x_train).astype(int)

    x_test_removed, indices = remove_entries_by_passage(x_test)

    x_test_not_removed = []
    question_idx_test_temp = []
    student_idx_test_temp = []
    experiment_version_test_temp = []


    for idx, x in np.ndenumerate(x_test_removed):
        if not np.isnan(x):
            x_test_not_removed.append(x_test[idx[0]][idx[1]])
            question_idx_test_temp.append(question_idx_test[idx[0]][idx[1]])
            student_idx_test_temp.append(student_idx_test[idx[0]])
            experiment_version_test_temp.append(experiment_version_test[idx[0]][idx[1]])

    question_idx_test_not_removed = np.array(question_idx_test_temp)
    student_idx_test_not_removed = np.array(student_idx_test_temp)
    experiment_version_test_not_removed = np.array(experiment_version_test_temp)
    x_test_not_removed = np.array(x_test_not_removed).astype(int)

    question_idx_test_temp = []
    student_idx_test_temp = []
    experiment_version_test_temp = []
    quiz_responses_test_temp = []

    for idx in indices:
        question_idx_test_temp.append(question_idx_test[idx[0]][idx[1]])
        student_idx_test_temp.append(student_idx_test[idx[0]])
        experiment_version_test_temp.append(experiment_version_test[idx[0]][idx[1]])
        quiz_responses_test_temp.append(x_test[idx[0]][idx[1]])

    question_idx_test = np.array(question_idx_test_temp)
    student_idx_test = np.array(student_idx_test_temp)
    experiment_version_test = np.array(experiment_version_test_temp)
    quiz_responses_test = np.array(quiz_responses_test_temp).astype(int)


    quiz_responses = np.concatenate((quiz_responses, x_test_not_removed))
    question_idx_train = np.concatenate((question_idx_train, question_idx_test_not_removed))
    student_idx_train = np.concatenate((student_idx_train, student_idx_test_not_removed))
    experiment_version_train = np.concatenate((experiment_version_train, experiment_version_test_not_removed))


    model_data = {
        'num_students': num_students,
        'num_questions': num_questions,
        'num_train': quiz_responses.shape[0],
        'student_idx_train': student_idx_train + 1,  # stan counts starting at 1
        'question_idx_train': question_idx_train + 1,
        'quiz_responses': quiz_responses,
        'question_idx_test': question_idx_test + 1,
        'student_idx_test': student_idx_test + 1,
        'num_test': quiz_responses_test.shape[0],
        'experiment_version_train': experiment_version_train + 1,
        'experiment_version_test': experiment_version_test + 1
    }

    if fold_num == 1:
        sm = pystan.StanModel(model_code=model_code)

    else:
        sm = pickle.load(open(model_name + '.pkl', 'rb'))

    pickle.dump(sm, open(model_name + '.pkl', 'wb'))

    if os.name == 'nt':
        fit = sm.sampling(data=model_data, iter=num_draws, chains=4, seed=42, n_jobs=1)  # must set n_jobs=1 for windows
    else:
        fit = sm.sampling(data=model_data, iter=num_draws, chains=4, seed=42, n_jobs=-1)

    # fig = fit.plot(pars=('mu_a','mu_d','sigma_a','sigma_d'))
    # plt.show()
    # print(fit)
    if debug:
        stan_utility.check_all_diagnostics(fit)

    predictions = fit['predictions']
    predictions = predictions[int(predictions.shape[0] / 2):]
    predictions = np.reshape(np.mean(predictions, axis=0), newshape=quiz_responses_test.shape)

    auc = roc_auc_score(quiz_responses_test, predictions, average='micro')

    # prediction = np.sum(predictions, axis=1)  # get the overall quiz score per participant
    # error = mean_squared_error(prediction, np.sum(x_test, axis=1))

    utils.pickle_out(model_name, {'extract': fit.extract(), 'sampler_params': fit.get_sampler_params()},
                     add_current_time=True)

    return 1, auc


